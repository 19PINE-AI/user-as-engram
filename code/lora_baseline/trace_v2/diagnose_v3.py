"""Per-schema diagnostic on v3 (expanded-generator) Stage A runs.

Handles a single split at a time; the v3 run only targets within_schema.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "stage_a_recite_v3"

UIDS = [f"u{i:03d}" for i in range(10)]


def _load_variant_map(uid: str) -> dict:
    u = json.loads((REPO / "data" / "users" / f"{uid}.json").read_text())
    return {q["question"]: q.get("variant") for q in u["indirect_qa"]}


def diagnose(tag_dir: Path):
    per_schema = defaultdict(list)
    per_variant = defaultdict(list)
    total = 0
    for uid in UIDS:
        variant_map = _load_variant_map(uid)
        path = tag_dir / uid / "eval_details.json"
        d = json.loads(path.read_text())
        held_w = d["with_adapter"]["held_indirect_details"]
        held_wo = d["without_adapter"]["held_indirect_details"]
        for w, wo in zip(held_w, held_wo):
            sch = w.get("schema") or "UNKNOWN"
            var = variant_map.get(w["q"], "")
            per_schema[sch].append((bool(w["ok"]), bool(wo["ok"])))
            if var:
                per_variant[(sch, var)].append((bool(w["ok"]), bool(wo["ok"])))
            total += 1
    print(f"=== {tag_dir.name}  (n={total} held questions, {len(UIDS)} users) ===")
    print(f"{'schema':<18} {'n':>4} {'with':>7} {'without':>8} {'lift':>7}")
    gw = 0; gwo = 0; n_all = 0
    for sch, pairs in sorted(per_schema.items(), key=lambda kv: -len(kv[1])):
        n = len(pairs)
        w = sum(1 for x, _ in pairs if x) / n
        wo = sum(1 for _, y in pairs if y) / n
        print(f"{sch:<18} {n:>4} {w:>7.3f} {wo:>8.3f} {w - wo:>+7.3f}")
        gw += w * n; gwo += wo * n; n_all += n
    print(f"{'TOTAL':<18} {n_all:>4} {gw/n_all:>7.3f} {gwo/n_all:>8.3f} {(gw-gwo)/n_all:>+7.3f}")
    print()
    print("--- per-variant (where variant present) ---")
    print(f"{'schema':<10} {'variant':<26} {'n':>4} {'with':>7} {'without':>8} {'lift':>7}")
    rows = []
    for (sch, var), pairs in per_variant.items():
        n = len(pairs)
        w = sum(1 for x, _ in pairs if x) / n
        wo = sum(1 for _, y in pairs if y) / n
        rows.append((sch, var, n, w, wo, w - wo))
    rows.sort(key=lambda r: (r[0], -r[3]))
    for sch, var, n, w, wo, lift in rows:
        print(f"{sch:<10} {var:<26} {n:>4} {w:>7.3f} {wo:>8.3f} {lift:>+7.3f}")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "qwen2p5_3b_r64_within_schema"
    diagnose(RESULTS / tag)


if __name__ == "__main__":
    main()
