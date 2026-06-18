"""Per-schema diagnostic on trace_v2 Stage A runs.

Aggregates held-indirect eval across all 10 users for each split (within_schema,
cross_schema, cross_schema_soft) and reports per-schema accuracy with and
without the adapter, plus the adapter lift (with - without).

Where does cross-schema collapse? If the failure is concentrated in a subset
of schemas, the generalization gap has structure we can target. If it's
uniform, the problem is architectural (adapter adds nothing, full stop).
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "stage_a_recite_v2"
TAG = "qwen2p5_3b"
UIDS = [f"u{i:03d}" for i in range(10)]
SPLITS = ["within_schema", "cross_schema", "cross_schema_soft"]


def load_split(split: str):
    """Return {schema: [(ok_with, ok_without), ...]} aggregated over users."""
    per_schema = defaultdict(list)
    total = 0
    for uid in UIDS:
        path = RESULTS / f"{TAG}_{split}" / uid / "eval_details.json"
        d = json.loads(path.read_text())
        held_w = d["with_adapter"]["held_indirect_details"]
        held_wo = d["without_adapter"]["held_indirect_details"]
        # same order (same question list)
        assert len(held_w) == len(held_wo), f"{uid} {split}"
        for w, wo in zip(held_w, held_wo):
            sch = w.get("schema") or "UNKNOWN"
            per_schema[sch].append((bool(w["ok"]), bool(wo["ok"])))
            total += 1
    return per_schema, total


def fmt(p):
    return f"{p:.3f}"


def main():
    for split in SPLITS:
        per_schema, total = load_split(split)
        print(f"\n=== {split}  (n={total} held questions, 10 users) ===")
        print(f"{'schema':<18} {'n':>4} {'with':>7} {'without':>8} {'lift':>7}")
        rows = []
        for sch, pairs in per_schema.items():
            n = len(pairs)
            w = sum(1 for x, _ in pairs if x) / n
            wo = sum(1 for _, y in pairs if y) / n
            rows.append((sch, n, w, wo, w - wo))
        rows.sort(key=lambda r: -r[1])
        gw = 0; gwo = 0; n_all = 0
        for sch, n, w, wo, lift in rows:
            print(f"{sch:<18} {n:>4} {fmt(w):>7} {fmt(wo):>8} {lift:>+7.3f}")
            gw += w * n; gwo += wo * n; n_all += n
        print(f"{'TOTAL':<18} {n_all:>4} {fmt(gw/n_all):>7} {fmt(gwo/n_all):>8} {(gw-gwo)/n_all:>+7.3f}")


if __name__ == "__main__":
    main()
