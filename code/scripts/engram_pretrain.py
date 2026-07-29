"""
Mini-Engram pretraining driver.
Implements the matched pretraining runs in “Experiments” (sec:experiments).

Usage (from $USER_AS_ENGRAM_ROOT/nanochat/, with NANOCHAT_BASE_DIR set):
  python -m scripts.engram_pretrain --depth 8 --num-iterations 500 --device-batch-size 8 --total-batch-size 65536 --engram off  --run base
  python -m scripts.engram_pretrain --depth 8 --num-iterations 500 --device-batch-size 8 --total-batch-size 65536 --engram on   --run engram

Pared-down version of scripts/base_train.py with explicit Engram on/off, fixed
hyperparameters, and a model-tag suffix so runs land in different checkpoint
directories. Single GPU only (no DDP).
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import torch

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_with_state_bos_bestfit, tokenizing_distributed_data_loader_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--engram", choices=["on", "off"], default="off")
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--aspect-ratio", type=int, default=64)
parser.add_argument("--head-dim", type=int, default=64)
parser.add_argument("--max-seq-len", type=int, default=1024)
parser.add_argument("--window-pattern", type=str, default="L", help="L (full) avoids SDPA's poor sliding-window perf on Blackwell")
parser.add_argument("--num-iterations", type=int, default=500)
parser.add_argument("--device-batch-size", type=int, default=8)
parser.add_argument("--total-batch-size", type=int, default=131072)  # 128k tokens / step
parser.add_argument("--embedding-lr", type=float, default=0.3)
parser.add_argument("--unembedding-lr", type=float, default=0.008)
parser.add_argument("--matrix-lr", type=float, default=0.02)
parser.add_argument("--scalar-lr", type=float, default=0.5)
parser.add_argument("--weight-decay", type=float, default=0.05)
parser.add_argument("--warmup-steps", type=int, default=20)
parser.add_argument("--warmdown-ratio", type=float, default=0.5)
parser.add_argument("--final-lr-frac", type=float, default=0.05)
parser.add_argument("--eval-every", type=int, default=100)
parser.add_argument("--eval-tokens", type=int, default=1_048_576)  # 1M tokens for val
# Engram hyperparameters
parser.add_argument("--engram-layer-ids", type=int, nargs="*", default=[2, 5])
parser.add_argument("--engram-vocab-per-ngram", type=int, default=50_000)
parser.add_argument("--engram-n-head", type=int, default=8)
parser.add_argument("--engram-n-embed", type=int, default=128)  # total per N-gram width, split across heads
parser.add_argument("--engram-max-ngram", type=int, default=3)
parser.add_argument("--run", type=str, default="dummy")
parser.add_argument("--model-tag", type=str, default=None)
parser.add_argument("--no-compile", action="store_true")
# FP8 training (port from base_train.py — Blackwell + H100+ only)
parser.add_argument("--fp8", action="store_true", help="enable FP8 training (requires CUDA SM 89+)")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 scaling recipe")
args = parser.parse_args()

device_type = autodetect_device_type()
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master = ddp_rank == 0
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")
print0(f"Device: {device}  DDP: {ddp}  world: {ddp_world_size}")

# -----------------------------------------------------------------------------
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

base_dim = args.depth * args.aspect_ratio
model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
num_heads = max(model_dim // args.head_dim, 1)
print0(f"Model dim: {model_dim}, num_heads: {num_heads}")

cfg_kwargs = dict(
    sequence_len=args.max_seq_len,
    vocab_size=vocab_size,
    n_layer=args.depth,
    n_head=num_heads, n_kv_head=num_heads,
    n_embd=model_dim,
    window_pattern=args.window_pattern,
)
if args.engram == "on":
    cfg_kwargs.update(
        engram_layer_ids=tuple(args.engram_layer_ids),
        engram_max_ngram_size=args.engram_max_ngram,
        engram_vocab_per_ngram=args.engram_vocab_per_ngram,
        engram_n_head_per_ngram=args.engram_n_head,
        engram_n_embed_per_ngram=args.engram_n_embed,
    )

config = GPTConfig(**cfg_kwargs)
print0(f"GPTConfig: {asdict(config)}")

with torch.device("meta"):
    model_meta = GPT(config)
model = model_meta
model.to_empty(device=device)
model.init_weights()
if args.engram == "on":
    base_dir = get_base_dir()
    model.attach_engram(tokenizer, base_dir=base_dir)
    print0(f"Engram attached at layers {args.engram_layer_ids}")
    print0(f"Engram total slots: {model.engram.hash_mapping.total_slots():,}")
    print0(f"Engram total params: {model.engram.total_engram_params():,}")

# -----------------------------------------------------------------------------
# FP8 training (port from base_train.py). Convert qualifying nn.Linear modules
# AFTER attach_engram so Engram K/V projections also get the FP8 path if they qualify.
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn
        def fp8_module_filter(mod, fqn):
            if not isinstance(mod, nn.Linear): return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0: return False
            if min(mod.in_features, mod.out_features) < 128: return False
            return True
        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        print0(f"FP8 enabled ({args.fp8_recipe}) - converted {num_fp8}/{num_linear} Linears, skipped {num_linear-num_fp8} (too small)")

param_counts = model.num_scaling_params()
print0("Param counts:")
for k, v in param_counts.items():
    print0(f"  {k}: {v:,}")
total_params = param_counts['total']
print0(f"TOTAL PARAMS: {total_params:,}")

# -----------------------------------------------------------------------------
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    scalar_lr=args.scalar_lr,
    weight_decay=args.weight_decay,
)
for g in optimizer.param_groups:
    g["initial_lr"] = g["lr"]

# -----------------------------------------------------------------------------
orig_model = model
if not args.no_compile:
    model = torch.compile(model, dynamic=False)
else:
    pass

# -----------------------------------------------------------------------------
# Dataloader
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device,
)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
    tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device,
)
x, y, _ = next(train_loader)

# -----------------------------------------------------------------------------
def get_lr_mult(it):
    warm = args.warmup_steps
    warmdown = round(args.warmdown_ratio * args.num_iterations)
    if it < warm:
        return (it + 1) / warm
    elif it <= args.num_iterations - warmdown:
        return 1.0
    else:
        progress = (args.num_iterations - it) / warmdown
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

tokens_per_micro = args.device_batch_size * args.max_seq_len
assert args.total_batch_size % tokens_per_micro == 0, \
    f"total_batch_size {args.total_batch_size} must be divisible by tokens_per_micro {tokens_per_micro}"
grad_accum = args.total_batch_size // tokens_per_micro
print0(f"Tokens/micro: {tokens_per_micro:,}  grad_accum: {grad_accum}")

base_dir = get_base_dir()
tag = args.model_tag if args.model_tag else (f"engram_d{args.depth}" if args.engram == "on" else f"base_d{args.depth}")
ckpt_dir = os.path.join(base_dir, "engram_runs", tag)
os.makedirs(ckpt_dir, exist_ok=True)
log_path = os.path.join(ckpt_dir, "train_log.jsonl")
log_fh = open(log_path, "w") if master else None
print0(f"Logging to {log_path}")

step = 0
total_time = 0.0
ema = 0.0
while True:
    last = step == args.num_iterations
    if args.eval_every > 0 and (last or step % args.eval_every == 0):
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
        # final checkpoint
        if master:
            torch.save(orig_model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
            with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                json.dump({**vars(args), **asdict(config)}, f, indent=2)
            print0(f"Saved final checkpoint to {ckpt_dir}/model.pt")
        break

    synchronize()
    t0 = time.time()
    for ms in range(grad_accum):
        loss = model(x, y)
        train_loss = loss.detach().float().item()
        loss = loss / grad_accum
        loss.backward()
        x, y, _ = next(train_loader)
    lrm = get_lr_mult(step)
    for g in optimizer.param_groups:
        g["lr"] = g["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    dt = time.time() - t0
    if step > 5: total_time += dt
    ema = 0.9 * ema + 0.1 * train_loss
    pct = 100 * step / args.num_iterations
    tps = int(args.total_batch_size / dt)
    print0(f"step {step:05d}/{args.num_iterations:05d} ({pct:.1f}%) | loss {train_loss:.4f} ema {ema/(1-0.9**(step+1)):.4f} | dt {dt*1000:.0f}ms | tok/s {tps:,} | lrm {lrm:.2f}")
    if master and step % 10 == 0:
        log_fh.write(json.dumps({"step": step, "train_loss": train_loss, "dt_ms": dt*1000, "tok_per_sec": tps, "lrm": lrm, "type": "train"}) + "\n")
        log_fh.flush()
    if step == 0:
        gc.collect(); gc.freeze(); gc.disable()
    step += 1

if master:
    log_fh.close()
print0(f"Training done. Total wall time {total_time/60:.2f}m")
compute_cleanup()
