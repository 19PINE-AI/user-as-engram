"""
Phases 2 + 3 of the layered architecture experiment.

For each test user, evaluate 6 conditions on Mini-Engram-d20:
  A) NO_EDIT                 — baseline
  B) per-user LoRA (rank-64) — POLAR-class on Mini-Engram (global edit)
  C) per-user Engram J-OPT   — local edit baseline
  D) (B) + (C)               — combination A: both per-user
  E) shared LoRA only        — does the meta-skill help on its own?
  F) shared LoRA + (C)       — combination B: layered design

Metrics:
  - direct top-1/top-5 on user's facts (completion-style prompts)
  - indirect top-1/top-5 (completion-style indirect probes)
  - val_bpb on held-out ClimbMix shard (locality test)

The independent-training assumption (per-user LoRA, per-user Engram, and
shared LoRA are each trained alone) tests whether the substrates COMPOSE
without joint optimisation. A future v2 could test joint training.

Usage (full Phase 2 at single rank):
  python -m scripts.layered_architecture \\
      --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d20_w1536_optimal \\
      --shared-lora-dir $NANOCHAT_BASE_DIR/shared_lora_d20/r16 \\
      --user-dir /home/ubuntu/user-as-lora/data/users \\
      --test-uids u000 u001 ... u019 \\
      --eval-tokens 524288 \\
      --out /home/ubuntu/user-as-engram/results/layered_d20_r16_full.json

Phase 3 ablation (subset of users at additional ranks):
  ... --test-uids u000 u001 u002 u003 u004  --shared-lora-dir .../r4   --out .../layered_d20_r4_abl.json
  ... --test-uids u000 u001 u002 u003 u004  --shared-lora-dir .../r64  --out .../layered_d20_r64_abl.json
"""
import os, sys, json, argparse, time, random
from pathlib import Path

import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb

from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, make_marker_UNEMBED_P,
)
from scripts.sft_baseline import attach_lora, detach_lora
from scripts.head_to_head_locality import (
    user_to_facts, direct_recall, measure_val_bpb,
    train_lora_on_user, undo_lora, train_engram_joint_opt,
)


# -----------------------------------------------------------------------------
# Shared-LoRA load/attach helpers
# -----------------------------------------------------------------------------

def attach_shared_lora(model, state_path, rank, alpha=None):
    """Attach a fresh LoRA, then load shared-LoRA weights from state_path."""
    if alpha is None:
        alpha = 2 * rank
    handles = attach_lora(model, rank=rank, alpha=alpha)
    state = torch.load(state_path, map_location="cuda" if torch.cuda.is_available() else "cpu")
    loaded = 0
    for name, _, lora, _ in handles:
        sd = lora.state_dict()
        for k in sd.keys():
            full_key = f"{name}.{k}"
            if full_key in state:
                sd[k] = state[full_key].to(sd[k].device).to(sd[k].dtype)
                loaded += 1
        lora.load_state_dict(sd)
    print(f"  loaded {loaded} shared-LoRA tensors from {state_path}")
    return handles


def lora_freeze(handles):
    """Set requires_grad=False on all LoRA params (we never train shared LoRA)."""
    for _, _, lora, _ in handles:
        for p in lora.parameters():
            p.requires_grad_(False)


# -----------------------------------------------------------------------------
# Indirect-reasoning probes (completion format on Mini-Engram base)
# -----------------------------------------------------------------------------

def render_indirect_prompts(user_json, n_max=20):
    """Make completion-format probes from indirect_qa entries.

    Format: 'Q: <question>\\nA:'  → model generates → check gold appears.
    """
    probes = []
    for iq in user_json["indirect_qa"][:n_max]:
        prompt = f"Q: {iq['question']}\nA:"
        gold = str(iq["answer"])
        probes.append({"prompt": prompt, "gold": gold})
    return probes


@torch.no_grad()
def indirect_accuracy(model, tokenizer, probes, device, max_new_tokens=16):
    bos = tokenizer.get_bos_token_id()
    n_top1 = n_any = 0
    for p in probes:
        ids = tokenizer.encode(p["prompt"], prepend=bos)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        # Greedy generate max_new_tokens
        for _ in range(max_new_tokens):
            logits = model(x)[0, -1, :]
            nxt = int(logits.argmax().item())
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        tail = tokenizer.decode(x[0, len(ids):].tolist())
        gold_l = p["gold"].lower().strip()
        tail_l = tail.lower().strip()
        # top-1: first token matches first token of gold
        gold_first = tokenizer.encode(" " + p["gold"].split()[0])[0] if p["gold"].split() else None
        if gold_first is not None:
            # Look at first generated token
            first_gen = x[0, len(ids)].item()
            if first_gen == gold_first:
                n_top1 += 1
        # any: gold string appears anywhere in tail
        if gold_l in tail_l:
            n_any += 1
    return n_top1, n_any, len(probes)


# -----------------------------------------------------------------------------
# Condition runners
# -----------------------------------------------------------------------------

def run_condition(label, eval_fn, *,
                  attach_per_user_lora=False, attach_shared_lora_h=None,
                  apply_per_user_engram=False,
                  model=None, config=None, tokenizer=None, facts=None,
                  user_probes=None, device=None,
                  lora_rank=64, lora_alpha=128, lora_steps=1500, lora_lr=5e-4,
                  engram_steps=1500, engram_lr=0.5,
                  eval_tokens=524288, device_bs=8, max_seq_len=1024, token_bytes=None):
    """Apply the named edit combo, run eval_fn(), restore. Returns dict."""
    state = {"label": label}
    t0 = time.time()
    per_user_handles = per_user_snap = None
    engram_restore = None

    try:
        # ----- attach edits -----
        if attach_per_user_lora:
            per_user_handles, per_user_snap, lora_s = train_lora_on_user(
                model, tokenizer, facts, device,
                rank=lora_rank, alpha=lora_alpha, steps=lora_steps, lr=lora_lr,
            )
            state["lora_train_s"] = lora_s
        if apply_per_user_engram:
            engram_restore, eng_s, n_rows = train_engram_joint_opt(
                model, config, tokenizer, facts, device,
                steps=engram_steps, lr=engram_lr,
            )
            state["engram_train_s"] = eng_s
            state["engram_n_rows"] = n_rows

        # ----- run evals -----
        d1, d5, dt = direct_recall(model, tokenizer, facts, device)
        state["direct_top1"] = d1
        state["direct_top5"] = d5
        state["direct_total"] = dt

        i1, ia, it = indirect_accuracy(model, tokenizer, user_probes, device)
        state["indirect_top1"] = i1
        state["indirect_any"] = ia
        state["indirect_total"] = it

        bpb = measure_val_bpb(model, tokenizer, device, eval_tokens,
                              device_bs, max_seq_len, token_bytes)
        state["val_bpb"] = bpb
    finally:
        # ----- restore -----
        if engram_restore is not None:
            engram_restore()
        if per_user_handles is not None:
            undo_lora(per_user_handles, per_user_snap, model)

    state["wall_s"] = time.time() - t0
    return state


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--shared-lora-dir", required=True,
                   help="Directory with lora_state.pt + meta.json from train_shared_lora.py")
    p.add_argument("--user-dir", default="/home/ubuntu/user-as-lora/data/users")
    p.add_argument("--test-uids", nargs="+",
                   default=[f"u{i:03d}" for i in range(20)])
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-steps", type=int, default=1500)
    p.add_argument("--lora-lr", type=float, default=5e-4)
    p.add_argument("--engram-steps", type=int, default=1500)
    p.add_argument("--engram-lr", type=float, default=0.5)
    p.add_argument("--eval-tokens", type=int, default=262144)
    p.add_argument("--device-bs", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true",
                   help="One user, truncated training, small val")
    args = p.parse_args()

    if args.smoke:
        args.test_uids = args.test_uids[:1]
        args.lora_steps = 100
        args.engram_steps = 100
        args.eval_tokens = 16384

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)

    print(f"Loading model from {args.ckpt_dir}")
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    model.eval()

    # Load shared-LoRA meta to discover its rank
    with open(Path(args.shared_lora_dir) / "meta.json") as f:
        shared_meta = json.load(f)
    shared_rank = shared_meta["rank"]
    shared_alpha = shared_meta.get("alpha", 2 * shared_rank)
    shared_state_path = Path(args.shared_lora_dir) / "lora_state.pt"
    print(f"Shared LoRA: rank={shared_rank} alpha={shared_alpha}")

    # ---- BASELINE val_bpb (condition A → all users; condition E → same regardless of user) ----
    print(f"\n[A] Baseline val_bpb ({args.eval_tokens} tokens) ...")
    baseline_bpb = measure_val_bpb(model, tokenizer, device, args.eval_tokens,
                                   args.device_bs, args.max_seq_len, token_bytes)
    print(f"  baseline_bpb = {baseline_bpb:.4f}")

    # ---- Shared-LoRA-only val_bpb (used by conditions E and F) ----
    print(f"\n[shared-LoRA only] Measuring val_bpb with shared LoRA attached ...")
    sl_handles = attach_shared_lora(model, shared_state_path, shared_rank, shared_alpha)
    lora_freeze(sl_handles)
    shared_only_bpb = measure_val_bpb(model, tokenizer, device, args.eval_tokens,
                                      args.device_bs, args.max_seq_len, token_bytes)
    detach_lora(sl_handles)
    print(f"  shared_only_bpb = {shared_only_bpb:.4f}  "
          f"(Δ vs base = {shared_only_bpb - baseline_bpb:+.4f})")

    out = {
        "config": vars(args),
        "shared_lora_meta": shared_meta,
        "baseline_bpb": baseline_bpb,
        "shared_only_bpb": shared_only_bpb,
        "per_user": [],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    user_files = [Path(args.user_dir) / f"{u}.json" for u in args.test_uids]
    for uidx, upath in enumerate(user_files):
        uid = upath.stem
        with open(upath) as f:
            uj = json.load(f)
        facts = user_to_facts(uj, tokenizer)
        probes = render_indirect_prompts(uj)
        print(f"\n=== [{uidx+1}/{len(user_files)}] {uid}  "
              f"({len(facts)} facts, {len(probes)} indirect) ===")

        user_results = {"uid": uid, "n_facts": len(facts), "n_probes": len(probes)}

        # ---- A: NO_EDIT ----
        t0 = time.time()
        d1, d5, dt = direct_recall(model, tokenizer, facts, device)
        i1, ia, it = indirect_accuracy(model, tokenizer, probes, device)
        user_results["A_no_edit"] = {
            "direct_top1": d1, "direct_top5": d5, "direct_total": dt,
            "indirect_top1": i1, "indirect_any": ia, "indirect_total": it,
            "val_bpb": baseline_bpb, "val_bpb_delta": 0.0,
            "wall_s": time.time() - t0,
        }
        print(f"  [A] direct={d1}/{dt}  indirect_any={ia}/{it}  Δbpb=+0.0000  "
              f"({user_results['A_no_edit']['wall_s']:.0f}s)")

        # ---- B: per-user LoRA ----
        b = run_condition("B_per_user_lora",
                          eval_fn=None,
                          attach_per_user_lora=True,
                          model=model, config=config, tokenizer=tokenizer,
                          facts=facts, user_probes=probes, device=device,
                          lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                          lora_steps=args.lora_steps, lora_lr=args.lora_lr,
                          engram_steps=args.engram_steps, engram_lr=args.engram_lr,
                          eval_tokens=args.eval_tokens, device_bs=args.device_bs,
                          max_seq_len=args.max_seq_len, token_bytes=token_bytes)
        b["val_bpb_delta"] = b["val_bpb"] - baseline_bpb
        user_results["B_per_user_lora"] = b
        print(f"  [B] direct={b['direct_top1']}/{dt}  indirect_any={b['indirect_any']}/{it}  "
              f"Δbpb={b['val_bpb_delta']:+.4f}  ({b['wall_s']:.0f}s)")

        # ---- C: per-user Engram ----
        c = run_condition("C_per_user_engram",
                          eval_fn=None,
                          apply_per_user_engram=True,
                          model=model, config=config, tokenizer=tokenizer,
                          facts=facts, user_probes=probes, device=device,
                          engram_steps=args.engram_steps, engram_lr=args.engram_lr,
                          eval_tokens=args.eval_tokens, device_bs=args.device_bs,
                          max_seq_len=args.max_seq_len, token_bytes=token_bytes)
        c["val_bpb_delta"] = c["val_bpb"] - baseline_bpb
        user_results["C_per_user_engram"] = c
        print(f"  [C] direct={c['direct_top1']}/{dt}  indirect_any={c['indirect_any']}/{it}  "
              f"Δbpb={c['val_bpb_delta']:+.4f}  ({c['wall_s']:.0f}s)")

        # ---- D: per-user LoRA + per-user Engram (combination A) ----
        d_ = run_condition("D_per_user_lora_plus_engram",
                           eval_fn=None,
                           attach_per_user_lora=True,
                           apply_per_user_engram=True,
                           model=model, config=config, tokenizer=tokenizer,
                           facts=facts, user_probes=probes, device=device,
                           lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                           lora_steps=args.lora_steps, lora_lr=args.lora_lr,
                           engram_steps=args.engram_steps, engram_lr=args.engram_lr,
                           eval_tokens=args.eval_tokens, device_bs=args.device_bs,
                           max_seq_len=args.max_seq_len, token_bytes=token_bytes)
        d_["val_bpb_delta"] = d_["val_bpb"] - baseline_bpb
        user_results["D_per_user_lora_plus_engram"] = d_
        print(f"  [D] direct={d_['direct_top1']}/{dt}  indirect_any={d_['indirect_any']}/{it}  "
              f"Δbpb={d_['val_bpb_delta']:+.4f}  ({d_['wall_s']:.0f}s)")

        # ---- E: shared LoRA only ----
        t0 = time.time()
        sl = attach_shared_lora(model, shared_state_path, shared_rank, shared_alpha)
        lora_freeze(sl)
        try:
            d1e, d5e, _ = direct_recall(model, tokenizer, facts, device)
            i1e, iae, _ = indirect_accuracy(model, tokenizer, probes, device)
            # val_bpb reused from shared_only_bpb (doesn't depend on user)
            user_results["E_shared_lora_only"] = {
                "direct_top1": d1e, "direct_top5": d5e,
                "indirect_top1": i1e, "indirect_any": iae,
                "val_bpb": shared_only_bpb,
                "val_bpb_delta": shared_only_bpb - baseline_bpb,
                "wall_s": time.time() - t0,
            }
            print(f"  [E] direct={d1e}/{dt}  indirect_any={iae}/{it}  "
                  f"Δbpb={shared_only_bpb-baseline_bpb:+.4f}  "
                  f"({user_results['E_shared_lora_only']['wall_s']:.0f}s)")
        finally:
            detach_lora(sl)

        # ---- F: shared LoRA + per-user Engram (combination B = layered) ----
        t0 = time.time()
        sl = attach_shared_lora(model, shared_state_path, shared_rank, shared_alpha)
        lora_freeze(sl)
        try:
            f_restore, f_eng_s, f_n_rows = train_engram_joint_opt(
                model, config, tokenizer, facts, device,
                steps=args.engram_steps, lr=args.engram_lr,
            )
            try:
                d1f, d5f, _ = direct_recall(model, tokenizer, facts, device)
                i1f, iaf, _ = indirect_accuracy(model, tokenizer, probes, device)
                f_bpb = measure_val_bpb(model, tokenizer, device, args.eval_tokens,
                                        args.device_bs, args.max_seq_len, token_bytes)
                user_results["F_layered"] = {
                    "direct_top1": d1f, "direct_top5": d5f,
                    "indirect_top1": i1f, "indirect_any": iaf,
                    "val_bpb": f_bpb, "val_bpb_delta": f_bpb - baseline_bpb,
                    "engram_train_s": f_eng_s, "engram_n_rows": f_n_rows,
                    "wall_s": time.time() - t0,
                }
                print(f"  [F] direct={d1f}/{dt}  indirect_any={iaf}/{it}  "
                      f"Δbpb={f_bpb-baseline_bpb:+.4f}  ({user_results['F_layered']['wall_s']:.0f}s)")
            finally:
                f_restore()
        finally:
            detach_lora(sl)

        out["per_user"].append(user_results)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)

    # ----- Aggregate -----
    per = out["per_user"]
    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    if per:
        agg = {"n_users": len(per)}
        for k in ("A_no_edit", "B_per_user_lora", "C_per_user_engram",
                  "D_per_user_lora_plus_engram", "E_shared_lora_only", "F_layered"):
            tots_d = [u[k]["direct_total"] if "direct_total" in u[k] else u["n_facts"] for u in per]
            tots_i = [u[k]["indirect_total"] if "indirect_total" in u[k] else u["n_probes"] for u in per]
            agg[f"{k}_direct_top1"] = mean([u[k]["direct_top1"] / max(t,1) for u, t in zip(per, tots_d)])
            agg[f"{k}_direct_top5"] = mean([u[k]["direct_top5"] / max(t,1) for u, t in zip(per, tots_d)])
            agg[f"{k}_indirect_top1"] = mean([u[k]["indirect_top1"] / max(t,1) for u, t in zip(per, tots_i)])
            agg[f"{k}_indirect_any"] = mean([u[k]["indirect_any"] / max(t,1) for u, t in zip(per, tots_i)])
            agg[f"{k}_val_bpb_delta"] = mean([u[k]["val_bpb_delta"] for u in per])
            agg[f"{k}_val_bpb_delta_gt0"] = mean([1.0 if u[k]["val_bpb_delta"] > 0 else 0.0 for u in per])
        out["agg"] = agg
        print(f"\n========== AGGREGATE n={agg['n_users']} ==========")
        for k in ("A_no_edit", "B_per_user_lora", "C_per_user_engram",
                  "D_per_user_lora_plus_engram", "E_shared_lora_only", "F_layered"):
            print(f"  {k:36s} direct_top1={agg[k+'_direct_top1']:.3f}  "
                  f"indirect_any={agg[k+'_indirect_any']:.3f}  "
                  f"Δbpb={agg[k+'_val_bpb_delta']:+.4f}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
