"""
Finetune an existing Mini-Engram with multi-fact-in-the-loss.

Per step (after batch loaded as x, y):
  1. Pick K positions per sequence (uniformly from a valid range that has
     enough left-context for the N-gram).
  2. For each (b, p):
        - rows = trigger_global_rows(eng, x[b:b+1], last_engram_layer, p)
        - g = sample a random target token id
        - marker = UNEMBED_P(g) = scale * (W_V^+ @ U_g), reshaped to
                                   [total_heads, embed_per_head]
        - save originals at rows; embedding.weight.data[rows] = marker
        - y[b, p] = g  (synthetic target replaces the natural next token)
  3. Forward + standard NTP cross_entropy loss against the modified y.
  4. Backward.
  5. Zero gradients on the union of overridden row indices (we don't
     train the synthetic markers themselves).
  6. Restore the original row values from saved copies.
  7. Optimizer step.

The standard NTP loss at the K injected positions now requires the model
to produce the synthetic gold; everywhere else the natural next-token
objective is untouched. K positions per batch is O(0.1%) of all training
tokens, so the natural LM signal is essentially preserved.

Usage:
  python -m scripts.engram_finetune_mf \\
    --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12_w1280_optimal \\
    --out-tag engram_d12_w1280_mf \\
    --num-iterations 1500 --K 10 --device-batch-size 8 \\
    --total-batch-size 131072 --max-seq-len 1024
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import gc, json, time, argparse, random
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit, tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, COMPUTE_DTYPE
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--out-tag", required=True)
    p.add_argument("--num-iterations", type=int, default=1500)
    p.add_argument("--device-batch-size", type=int, default=8)
    p.add_argument("--total-batch-size", type=int, default=131072)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--K", type=int, default=10,
                    help="number of fact-injection positions per sequence per step")
    p.add_argument("--scale", type=float, default=20.0,
                    help="marker magnitude (matches inference-time insertion)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--warmdown-ratio", type=float, default=0.5)
    p.add_argument("--final-lr-frac", type=float, default=0.05)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-tokens", type=int, default=524288)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device_type = autodetect_device_type()
    ddp, ddp_rank, *_, device = compute_init(device_type)
    master = ddp_rank == 0
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    # ----- Load model -----
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Loading checkpoint: {args.ckpt_dir}")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.train()
    eng = model.engram
    assert eng is not None, "checkpoint must be an Engram-pretrained model"
    last_engram_layer = max(config.engram_layer_ids)
    tbl = eng.tables[str(last_engram_layer)]
    layer_mod = eng.layers_module[str(last_engram_layer)]
    embed_per_head = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    Wv = layer_mod.value_proj.weight.data.float()  # [n_embd, total_heads * embed_per_head]
    Wv_pinv = torch.linalg.pinv(Wv).to(device)
    # U: unembedding matrix [vocab, n_embd]; will be used to build marker
    print0(f"Engram last layer={last_engram_layer} total_heads={total_heads} "
            f"embed_per_head={embed_per_head}")
    print0(f"Wv shape={tuple(Wv.shape)}  Wv_pinv shape={tuple(Wv_pinv.shape)}")

    # ----- Optimizer (small finetune LR; we don't want to disturb pretrained weights too much) -----
    optimizer = model.setup_optimizer(
        unembedding_lr=args.lr * 0.1,
        embedding_lr=args.lr * 0.1,
        matrix_lr=args.lr,
        scalar_lr=args.lr * 5,
        weight_decay=args.weight_decay,
    )
    for g_ in optimizer.param_groups:
        g_["initial_lr"] = g_["lr"]

    orig_model = model
    if not args.no_compile:
        model = torch.compile(model, dynamic=False)

    # ----- Data -----
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
    )
    x, y, _ = next(train_loader)

    tokens_per_micro = args.device_batch_size * args.max_seq_len
    assert args.total_batch_size % tokens_per_micro == 0
    grad_accum = args.total_batch_size // tokens_per_micro
    print0(f"Tokens/micro: {tokens_per_micro:,}  grad_accum: {grad_accum}")
    print0(f"K injections per micro: {args.K * args.device_batch_size} "
            f"= {args.K * args.device_batch_size / tokens_per_micro:.2%} of tokens")

    # ----- LR schedule -----
    def get_lr_mult(it):
        warm = args.warmup_steps
        warmdown = round(args.warmdown_ratio * args.num_iterations)
        if it < warm:
            return (it + 1) / warm
        elif it <= args.num_iterations - warmdown:
            return 1.0
        progress = (args.num_iterations - it) / warmdown
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

    # ----- Marker construction -----
    # Get the unembedding matrix U [vocab, n_embd]
    U = orig_model.lm_head.weight  # [vocab, n_embd]
    def make_marker(gold_id: int) -> torch.Tensor:
        # UNEMBED_P: marker = scale * Wv_pinv @ U_g, reshaped to [total_heads, embed_per_head]
        u_g = U[gold_id].detach().float()
        m = (Wv_pinv @ u_g) * args.scale  # [total_heads * embed_per_head]
        return m.view(total_heads, embed_per_head).to(tbl.embedding.weight.dtype)

    # Earliest position allowing the N-gram lookup to have left-context
    min_pos = config.engram_max_ngram_size
    # Latest position where we still have a next-token target (y[b, T-1] is the last valid target)
    max_pos = args.max_seq_len - 1

    # ----- Output dirs -----
    base_dir = get_base_dir()
    ckpt_dir = os.path.join(base_dir, "engram_runs", args.out_tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(ckpt_dir, "train_log.jsonl")
    log_fh = open(log_path, "w") if master else None

    # ----- Training loop -----
    step = 0
    total_time = 0.0
    ema = 0.0
    while True:
        last = step == args.num_iterations
        if args.eval_every > 0 and (last or (step > 0 and step % args.eval_every == 0)):
            model.eval()
            with torch.no_grad():
                val_loader = build_val_loader()
                eval_steps = max(1, args.eval_tokens // tokens_per_micro)
                val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"step {step:05d} | val_bpb: {val_bpb:.6f}")
            if master:
                log_fh.write(json.dumps({"step": step, "val_bpb": float(val_bpb), "type": "eval"}) + "\n")
                log_fh.flush()
            model.train()

        if last:
            if master:
                torch.save(orig_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
                with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                    cfg_d = {**vars(args)}
                    cfg_d.update({
                        "sequence_len": config.sequence_len, "vocab_size": config.vocab_size,
                        "n_layer": config.n_layer, "n_head": config.n_head,
                        "n_kv_head": config.n_kv_head, "n_embd": config.n_embd,
                        "window_pattern": config.window_pattern,
                        "engram_layer_ids": list(config.engram_layer_ids),
                        "engram_max_ngram_size": config.engram_max_ngram_size,
                        "engram_vocab_per_ngram": config.engram_vocab_per_ngram,
                        "engram_n_head_per_ngram": config.engram_n_head_per_ngram,
                        "engram_n_embed_per_ngram": config.engram_n_embed_per_ngram,
                    })
                    json.dump(cfg_d, f, indent=2)
                print0(f"Saved final checkpoint to {ckpt_dir}/model.pt")
            break

        synchronize()
        t0 = time.time()
        for _ms in range(grad_accum):
            B = x.shape[0]
            # Pick K positions per sequence, sample synthetic gold per (b, p)
            picks = []  # list of (b, p, g)
            all_rows = []
            all_orig = []
            for b in range(B):
                for _ in range(args.K):
                    p = rng.randint(min_pos, max_pos)
                    g_id = rng.randrange(vocab_size)
                    picks.append((b, p, g_id))
                    rows = trigger_global_rows(eng, x[b:b+1], last_engram_layer, p)
                    orig = tbl.embedding.weight.data[rows].detach().clone()
                    marker = make_marker(g_id)
                    write_marker(eng, last_engram_layer, rows, marker)
                    all_rows.append(rows)
                    all_orig.append((rows, orig))

            # Modify y at injected positions
            y_mod = y.clone()
            for (b, p, g_id) in picks:
                y_mod[b, p] = g_id

            # Forward + backward (NTP loss)
            loss = model(x, y_mod)
            train_loss = loss.detach().float().item()
            (loss / grad_accum).backward()

            # Zero grads on overridden rows (do not train synthetic markers)
            if tbl.embedding.weight.grad is not None and all_rows:
                idx = torch.cat(all_rows)
                tbl.embedding.weight.grad.index_fill_(0, idx, 0.0)

            # Restore originals
            for rows, orig in all_orig:
                restore_rows(eng, last_engram_layer, rows, orig)

            x, y, _ = next(train_loader)

        lrm = get_lr_mult(step)
        for g_ in optimizer.param_groups:
            g_["lr"] = g_["initial_lr"] * lrm
        optimizer.step()
        model.zero_grad(set_to_none=True)
        synchronize()
        dt = time.time() - t0
        if step > 5: total_time += dt
        ema = 0.9 * ema + 0.1 * train_loss
        pct = 100 * step / args.num_iterations
        tps = int(args.total_batch_size / dt)
        print0(f"step {step:05d}/{args.num_iterations:05d} ({pct:.1f}%) "
                f"| loss {train_loss:.4f} ema {ema/(1-0.9**(step+1)):.4f} "
                f"| dt {dt*1000:.0f}ms | tok/s {tps:,} | lrm {lrm:.2f}")
        if master and step % 10 == 0:
            log_fh.write(json.dumps({"step": step, "train_loss": train_loss,
                                        "dt_ms": dt*1000, "tok_per_sec": tps,
                                        "lrm": lrm, "type": "train"}) + "\n")
            log_fh.flush()
        if step == 0:
            gc.collect(); gc.freeze(); gc.disable()
        step += 1

    if master:
        log_fh.close()
    print0(f"Multi-fact finetune done. Total wall time {total_time/60:.2f}m")
    compute_cleanup()


if __name__ == "__main__":
    main()
