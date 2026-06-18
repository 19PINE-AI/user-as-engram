"""Test OPT at higher step count on 100-fact density regime."""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import os, json, argparse, time
from pathlib import Path
import torch

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, write_marker, restore_rows,
    make_marker_OPT,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default=f"{UAE_ROOT}/data/corpora_xxl.json")
    p.add_argument("--out", default=f"{UAE_ROOT}/results/opt_strong_density.json")
    p.add_argument("--n-facts", type=int, default=100)
    p.add_argument("--opt-steps", type=int, default=60)
    p.add_argument("--opt-lr", type=float, default=1.0)
    p.add_argument("--scale", type=float, default=50.0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()
    model, config = load_model(args.ckpt_dir, tokenizer, device)
    eng = model.engram
    last_layer = max(config.engram_layer_ids)
    Wv_pinv = torch.linalg.pinv(eng.layers_module[str(last_layer)].value_proj.weight.data.float())
    embed_dim = eng.embed_per_head
    total_heads = config.engram_n_head_per_ngram * (config.engram_max_ngram_size - 1)
    bos = tokenizer.get_bos_token_id()

    with open(args.corpus) as f:
        corpora = json.load(f)
    seen = {}
    for f in corpora["user_facts"]:
        seen[f["trigger"]] = f
        if len(seen) >= args.n_facts: break
    facts = list(seen.values())[:args.n_facts]
    print(f"OPT-{args.opt_steps} on {len(facts)} facts (lr={args.opt_lr}, scale={args.scale})")

    # Build all overrides (write all at once, eval all)
    tbl = eng.tables[str(last_layer)]
    all_writes = []
    t0 = time.time()
    for i, f in enumerate(facts):
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        gold_id = tokenizer.encode(f["gold"])[0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        global_rows = trigger_global_rows(eng, idx, last_layer, trig_pos, user_salt=0)
        marker = make_marker_OPT(model, eng, last_layer, gold_id, idx, trig_pos, args.scale,
                                  total_heads, embed_dim, Wv_pinv=Wv_pinv,
                                  n_steps=args.opt_steps, lr=args.opt_lr)
        all_writes.append({"prompt_ids": ids, "gold_id": gold_id, "global_rows": global_rows, "marker": marker})
        if (i+1) % 25 == 0:
            print(f"  built {i+1}/{len(facts)}  elapsed {time.time()-t0:.0f}s")
    build_time = time.time() - t0
    print(f"All overrides built in {build_time:.0f}s")

    # Apply all
    saved = []
    for o in all_writes:
        saved.append(tbl.embedding.weight.data[o["global_rows"]].clone())
        write_marker(eng, last_layer, o["global_rows"], o["marker"])
    try:
        # Eval
        n_top1 = n_top5 = 0
        for o in all_writes:
            x = torch.tensor([o["prompt_ids"]], dtype=torch.long, device=device)
            with torch.no_grad():
                lg = model(x)[0, -1, :]
            r = int((lg > lg[o["gold_id"]]).sum().item())
            if r == 0: n_top1 += 1
            if r < 5: n_top5 += 1
    finally:
        for o, originals in zip(all_writes, saved):
            restore_rows(eng, last_layer, o["global_rows"], originals)

    print(f"\nOPT-{args.opt_steps} on {len(facts)} simultaneous facts:")
    print(f"  top-1: {n_top1}/{len(facts)} = {n_top1/len(facts):.1%}")
    print(f"  top-5: {n_top5}/{len(facts)} = {n_top5/len(facts):.1%}")
    print(f"  build time: {build_time:.0f}s ({build_time/len(facts)*1000:.0f}ms/fact)")

    out = {"config": vars(args), "n": len(facts),
            "top1": n_top1/len(facts), "top5": n_top5/len(facts),
            "build_time_s": build_time}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
