"""Run only the judge phase of judge_layered, given an existing predictions.json."""
import json, argparse, sys
from pathlib import Path
sys.path.insert(0, ".")
from scripts.judge_layered import judge_all
import torch

p = argparse.ArgumentParser()
p.add_argument("--predictions", required=True,
               help="Path to layered_judge_d??.predictions.json")
p.add_argument("--judge-model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--out", required=True)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
with open(args.predictions) as f:
    src = json.load(f)
preds = src["predictions"]

preds = judge_all(args.judge_model, preds, device)

# Aggregate per condition
by_cond = {}
for p_ in preds:
    c = p_["condition"]
    if c not in by_cond: by_cond[c] = {"correct": 0, "total": 0}
    for a in p_["answers"]:
        by_cond[c]["total"] += 1
        if a["correct"]: by_cond[c]["correct"] += 1
agg = {c: {"correct": v["correct"], "total": v["total"],
            "accuracy": v["correct"] / max(v["total"], 1)}
        for c, v in by_cond.items()}

out = {"config": src["config"], "predictions": preds, "agg": agg}
with open(args.out, "w") as f: json.dump(out, f, indent=2)

print("\n========== LLM-judge aggregate ==========")
for c, v in agg.items():
    print(f"  {c:36s}  {v['accuracy']*100:.1f}%  ({v['correct']}/{v['total']})")
print(f"\nWrote {args.out}")
