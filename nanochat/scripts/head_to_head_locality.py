"""
Head-to-head locality control: POLAR-class LoRA vs Engram-row insertion on
Mini-Engram-d20, with the SAME base model and SAME user fact set. The thesis
prediction:

    A) NO_EDIT       :  val_bpb_delta = 0
    B) LoRA          :  val_bpb_delta > 0   (global edit harms unrelated text)
    C) ENGRAM_J_OPT  :  val_bpb_delta ~ 0   (local edit, gate doesn't fire)

`val_bpb_delta` on a held-out ClimbMix shard is the *cleanest* locality test:
it measures whether the per-user edit contaminates the model's behaviour on
text that has nothing to do with the user's facts. Indirect-reasoning probes
require the base to actually reason; a 1.22B *base* LM can't do that
reliably, so the BPB delta is a more honest substitute that directly tests
the architectural claim.

We also report per-fact direct-recall to confirm both methods actually
memorise the facts (otherwise the delta isn't fair).

Usage:
  python -m scripts.head_to_head_locality \\
       --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \\
       --user-dir /home/ubuntu/user-as-lora/data/users \\
       --n-users 20 \\
       --eval-tokens 524288 \\
       --out /home/ubuntu/user-as-engram/results/head_to_head_d20.json
"""
import os, sys, json, argparse, time, random
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_token_bytes

from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, make_marker_UNEMBED_P,
)
from scripts.sft_baseline import attach_lora, detach_lora


# -----------------------------------------------------------------------------
# User-as-LoRA → Engram corpus conversion
# -----------------------------------------------------------------------------

def user_to_facts(user_json: dict, tokenizer):
    """Convert one user-as-lora user into Engram-style (prompt, gold) facts.

    For each fact we pick the canonical paraphrase that *ends with* the
    answer string, peel off the answer, and use the prefix as the trigger.
    Single-token-gold convention (tokeniser-dependent first BPE token).
    """
    facts = []
    for f in user_json["facts"]:
        ans = str(f["answer"])
        if isinstance(f["answer"], list):
            ans = ", ".join(str(x) for x in f["answer"])
        # Find a paraphrase that ends with the answer
        prefix = None
        for p in f["paraphrases"]:
            p = p.rstrip(".")
            if p.endswith(ans):
                prefix = p[: -len(ans)].rstrip()
                break
        if prefix is None:
            # Fallback: construct one from the key
            prefix = f"My {f['key'].replace('_', ' ')} is"
        # Gold = first BPE token of " <ans>" (note leading space, matters)
        gold_text = " " + ans.split()[0] if " " in ans else " " + ans
        gold_ids = tokenizer.encode(gold_text)
        if not gold_ids:
            continue
        facts.append({
            "key": f["key"],
            "prompt": prefix,
            "gold_text": gold_text,
            "gold_id": gold_ids[0],
        })
    return facts


# -----------------------------------------------------------------------------
# Common eval primitives
# -----------------------------------------------------------------------------

@torch.no_grad()
def direct_recall(model, tokenizer, facts, device):
    bos = tokenizer.get_bos_token_id()
    n_top1 = n_top5 = 0
    for f in facts:
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]
        gold = f["gold_id"]
        rank = int((logits > logits[gold]).sum().item())
        if rank == 0: n_top1 += 1
        if rank < 5: n_top5 += 1
    return n_top1, n_top5, len(facts)


def measure_val_bpb(model, tokenizer, device, eval_tokens, device_bs, max_seq_len, token_bytes):
    eval_steps = max(1, eval_tokens // (device_bs * max_seq_len))
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, device_bs, max_seq_len, split="val", device=device,
    )
    return float(evaluate_bpb(model, loader, eval_steps, token_bytes))


# -----------------------------------------------------------------------------
# Condition B: POLAR-class LoRA training on Mini-Engram
# -----------------------------------------------------------------------------

def train_lora_on_user(model, tokenizer, facts, device, rank=64, alpha=128, steps=1500, lr=5e-4):
    """Attach rank-64 LoRA, train on (prompt, gold) pairs. Returns handles for detach()."""
    examples = []
    bos = tokenizer.get_bos_token_id()
    for f in facts:
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        examples.append((ids, f["gold_id"]))

    # Snapshot grads
    grad_snap = []
    for p in model.parameters():
        grad_snap.append(p.requires_grad)
        p.requires_grad_(False)

    handles = attach_lora(model, rank=rank, alpha=alpha)
    lora_params = []
    for _, _, lora, _ in handles:
        for p in lora.parameters():
            p.requires_grad_(True)
            lora_params.append(p)

    optim = torch.optim.Adam(lora_params, lr=lr)
    t0 = time.time()
    for step in range(steps):
        i = torch.randint(0, len(examples), (1,)).item()
        ids, gold = examples[i]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        logits = model(x)
        loss = F.cross_entropy(
            logits[0, -1, :].unsqueeze(0).float(),
            torch.tensor([gold], device=device),
        )
        grads = torch.autograd.grad(loss, lora_params, create_graph=False)
        for p, g in zip(lora_params, grads):
            if p.grad is None: p.grad = g.clone()
            else: p.grad.copy_(g)
        optim.step()
        optim.zero_grad()
    return handles, grad_snap, time.time() - t0


def undo_lora(handles, grad_snap, model):
    detach_lora(handles)
    for p, rg in zip(model.parameters(), grad_snap):
        p.requires_grad_(rg)


# -----------------------------------------------------------------------------
# Condition C: Engram-row Joint OPT on Mini-Engram
# -----------------------------------------------------------------------------

def train_engram_joint_opt(model, config, tokenizer, facts, device,
                           steps=1500, lr=0.5, init_scale=20.0):
    """Run Joint OPT on the user's fact set. Modifies the Engram table in-place;
    returns (restore_fn, train_time_s)."""
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    tbl = eng.tables[str(last_layer)]
    # CUSOLVER occasionally fails under GPU memory pressure; CPU pinv is
    # tiny (matrix is ~256x256) and orders of magnitude more reliable here.
    Wv_pinv = torch.linalg.pinv(
        eng.layers_module[str(last_layer)].value_proj.weight.data.float().cpu()
    ).to(eng.layers_module[str(last_layer)].value_proj.weight.device)
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    bos = tokenizer.get_bos_token_id()

    fact_data = []
    all_global_rows_set = set()
    for f in facts:
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos, user_salt=0)
        init_marker = make_marker_UNEMBED_P(
            model, eng, last_layer, f["gold_id"], idx, trig_pos,
            init_scale, total_heads, embed_dim, Wv_pinv=Wv_pinv,
        )
        fact_data.append({
            "prompt_ids": ids, "gold_id": f["gold_id"],
            "global_rows": global_rows, "init_marker": init_marker,
        })
        all_global_rows_set.update(global_rows.tolist())

    all_rows_list = sorted(all_global_rows_set)
    all_rows_t = torch.tensor(all_rows_list, dtype=torch.long, device=device)
    saved_originals = tbl.embedding.weight.data[all_rows_t].clone()
    addr_to_leaf = {a: i for i, a in enumerate(all_rows_list)}

    init_stack = torch.zeros(len(all_rows_list), embed_dim, device=device)
    counts = torch.zeros(len(all_rows_list), device=device)
    for fd in fact_data:
        for i, a in enumerate(fd["global_rows"].tolist()):
            li = addr_to_leaf[a]
            init_stack[li] += fd["init_marker"][i]
            counts[li] += 1
    counts = counts.clamp_min(1)
    init_stack = init_stack / counts.unsqueeze(-1)
    row_leaves = init_stack.detach().clone().to(tbl.embedding.weight.dtype).requires_grad_(True)

    grad_snap = []
    for p in model.parameters():
        grad_snap.append(p.requires_grad)
        p.requires_grad_(False)

    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = 0

    addr_to_leaf_t = torch.full((tbl.embedding.weight.size(0),), -1,
                                dtype=torch.long, device=device)
    for a, li in addr_to_leaf.items():
        addr_to_leaf_t[a] = li

    def hook(module, inputs, output):
        in_idx = inputs[0]
        leaf_idx = addr_to_leaf_t[in_idx]
        mask = (leaf_idx >= 0)
        if not mask.any():
            return output
        out = output.clone() if not output.requires_grad else output
        flat_idx = leaf_idx.clamp_min(0)
        added = row_leaves[flat_idx].to(out.dtype)
        m = mask.unsqueeze(-1).to(out.dtype)
        return out * (1 - m) + added * m

    handle = tbl.embedding.register_forward_hook(hook)
    optim = torch.optim.Adam([row_leaves], lr=lr)
    t0 = time.time()
    try:
        for step in range(steps):
            i = torch.randint(0, len(fact_data), (1,)).item()
            fd = fact_data[i]
            x = torch.tensor([fd["prompt_ids"]], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :]
            loss = F.cross_entropy(
                logits.unsqueeze(0).float(),
                torch.tensor([fd["gold_id"]], device=device),
            )
            grads = torch.autograd.grad(loss, [row_leaves])
            if row_leaves.grad is None: row_leaves.grad = grads[0].clone()
            else: row_leaves.grad.copy_(grads[0])
            optim.step()
            optim.zero_grad()
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = row_leaves.detach().to(tbl.embedding.weight.dtype)
    except Exception:
        # Restore even on error
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = saved_originals
        handle.remove()
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)
        raise

    train_time = time.time() - t0

    def restore():
        handle.remove()
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = saved_originals
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)

    return restore, train_time, len(all_rows_list)


# -----------------------------------------------------------------------------
# Main per-user loop
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--user-dir", default="/home/ubuntu/user-as-lora/data/users")
    p.add_argument("--n-users", type=int, default=20)
    p.add_argument("--eval-tokens", type=int, default=524_288)
    p.add_argument("--device-bs", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-steps", type=int, default=1500)
    p.add_argument("--lora-lr", type=float, default=5e-4)
    p.add_argument("--engram-steps", type=int, default=1500)
    p.add_argument("--engram-lr", type=float, default=0.5)
    p.add_argument("--out", default="/home/ubuntu/user-as-engram/results/head_to_head_d20.json")
    p.add_argument("--smoke", action="store_true",
                   help="Run only 1 user with truncated steps/tokens for end-to-end validation.")
    args = p.parse_args()

    if args.smoke:
        args.n_users = 1
        args.eval_tokens = 16384
        args.lora_steps = 50
        args.engram_steps = 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    print(f"Loading model from {args.ckpt_dir} ...")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.eval()

    # Baseline val_bpb (single measurement reused for every user's delta).
    print(f"\nBaseline val_bpb on {args.eval_tokens} tokens ...")
    t0 = time.time()
    baseline_bpb = measure_val_bpb(
        model, tokenizer, device, args.eval_tokens,
        args.device_bs, args.max_seq_len, token_bytes,
    )
    print(f"  baseline_bpb = {baseline_bpb:.4f}  ({time.time()-t0:.0f}s)")

    user_files = sorted(Path(args.user_dir).glob("u*.json"))[: args.n_users]
    print(f"\nEvaluating {len(user_files)} users.")

    out = {
        "config": vars(args),
        "model_ckpt": args.ckpt_dir,
        "baseline_bpb": baseline_bpb,
        "per_user": [],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for uidx, upath in enumerate(user_files):
        uid = upath.stem
        with open(upath) as f:
            user_json = json.load(f)
        facts = user_to_facts(user_json, tokenizer)
        print(f"\n=== [{uidx+1}/{len(user_files)}] {uid}  ({len(facts)} facts) ===")

        # ---------------- A) NO_EDIT direct recall (single measurement) -----
        t0 = time.time()
        a_t1, a_t5, ntot = direct_recall(model, tokenizer, facts, device)
        a_time = time.time() - t0
        print(f"  [A] no_edit: direct top1={a_t1}/{ntot} top5={a_t5}/{ntot}  ({a_time:.0f}s)")

        # ---------------- B) LoRA -------------------------------------------
        t0 = time.time()
        handles, grad_snap, lora_train_s = train_lora_on_user(
            model, tokenizer, facts, device,
            rank=args.lora_rank, alpha=args.lora_alpha,
            steps=args.lora_steps, lr=args.lora_lr,
        )
        b_t1, b_t5, _ = direct_recall(model, tokenizer, facts, device)
        b_bpb = measure_val_bpb(
            model, tokenizer, device, args.eval_tokens,
            args.device_bs, args.max_seq_len, token_bytes,
        )
        undo_lora(handles, grad_snap, model)
        b_time = time.time() - t0
        print(f"  [B] LoRA   : direct top1={b_t1}/{ntot} top5={b_t5}/{ntot} "
              f"val_bpb={b_bpb:.4f} (Δ={b_bpb-baseline_bpb:+.4f})  ({b_time:.0f}s)")

        # ---------------- C) Engram Joint OPT --------------------------------
        t0 = time.time()
        restore, eng_train_s, n_rows = train_engram_joint_opt(
            model, config, tokenizer, facts, device,
            steps=args.engram_steps, lr=args.engram_lr,
        )
        c_t1, c_t5, _ = direct_recall(model, tokenizer, facts, device)
        c_bpb = measure_val_bpb(
            model, tokenizer, device, args.eval_tokens,
            args.device_bs, args.max_seq_len, token_bytes,
        )
        restore()
        c_time = time.time() - t0
        print(f"  [C] Engram : direct top1={c_t1}/{ntot} top5={c_t5}/{ntot} "
              f"val_bpb={c_bpb:.4f} (Δ={c_bpb-baseline_bpb:+.4f})  "
              f"rows={n_rows}  ({c_time:.0f}s)")

        out["per_user"].append({
            "uid": uid, "n_facts": ntot,
            "no_edit": {"top1": a_t1, "top5": a_t5,
                         "bpb": baseline_bpb, "bpb_delta": 0.0},
            "lora": {"top1": b_t1, "top5": b_t5,
                      "bpb": b_bpb, "bpb_delta": b_bpb - baseline_bpb,
                      "train_s": lora_train_s},
            "engram_joint_opt": {"top1": c_t1, "top5": c_t5,
                                  "bpb": c_bpb, "bpb_delta": c_bpb - baseline_bpb,
                                  "train_s": eng_train_s, "n_rows": n_rows},
        })
        # Save incrementally so partial results survive interruption.
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    # ----------- Aggregate across users -----------
    per = out["per_user"]
    if per:
        def mean(xs): return sum(xs) / len(xs)
        agg = {
            "n_users": len(per),
            "no_edit_top1_mean": mean([u["no_edit"]["top1"] / u["n_facts"] for u in per]),
            "lora_top1_mean": mean([u["lora"]["top1"] / u["n_facts"] for u in per]),
            "engram_top1_mean": mean([u["engram_joint_opt"]["top1"] / u["n_facts"] for u in per]),
            "lora_bpb_delta_mean": mean([u["lora"]["bpb_delta"] for u in per]),
            "engram_bpb_delta_mean": mean([u["engram_joint_opt"]["bpb_delta"] for u in per]),
            "lora_bpb_delta_pos_frac": mean([1.0 if u["lora"]["bpb_delta"] > 0 else 0.0 for u in per]),
            "engram_bpb_delta_pos_frac": mean([1.0 if u["engram_joint_opt"]["bpb_delta"] > 0 else 0.0 for u in per]),
        }
        out["agg"] = agg
        print(f"\n========== AGGREGATE (n={agg['n_users']}) ==========")
        print(f"direct recall mean top1 — no_edit: {agg['no_edit_top1_mean']:.3f}, "
              f"LoRA: {agg['lora_top1_mean']:.3f}, "
              f"Engram: {agg['engram_top1_mean']:.3f}")
        print(f"val_bpb Δ mean — LoRA: {agg['lora_bpb_delta_mean']:+.4f}, "
              f"Engram: {agg['engram_bpb_delta_mean']:+.4f}")
        print(f"val_bpb Δ > 0 fraction — LoRA: {agg['lora_bpb_delta_pos_frac']:.2f}, "
              f"Engram: {agg['engram_bpb_delta_pos_frac']:.2f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
