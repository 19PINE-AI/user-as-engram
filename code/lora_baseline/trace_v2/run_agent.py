"""Orchestrate agent trajectory generation across users × indirect_qa.

Writes per-user JSONL of raw trajectories plus an aggregate filter report.
Does NOT yet rewrite — that's a separate step downstream.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .agent import AgentConfig, run_agent
from .filter import evaluate, short_summary


REPO = Path(__file__).resolve().parents[2]
USERS_DIR = REPO / "data" / "users"
OUT_ROOT = REPO / "data" / "trace_v2"


def run_for_user(uid: str, cfg: AgentConfig, out_dir: Path, n_attempts: int = 1, verbose: bool = False) -> list[dict]:
    user = json.loads((USERS_DIR / f"{uid}.json").read_text())
    results: list[dict] = []
    out_path = out_dir / f"{uid}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = 0
    total = 0
    with out_path.open("w") as f:
        for q in user["indirect_qa"]:
            total += 1
            best = None
            best_fr = None
            for attempt in range(n_attempts):
                traj = run_agent(
                    uid=uid,
                    question=q["question"],
                    gold=str(q["answer"]),
                    schema=q["schema"],
                    required_fact_keys=q["required_fact_keys"],
                    cfg=cfg,
                )
                fr = evaluate(traj)
                if verbose:
                    print("  ", short_summary(traj, fr))
                if fr.accepted:
                    best = traj
                    best_fr = fr
                    break
                if best is None or (fr.answer_correct and not best_fr.answer_correct):
                    best = traj
                    best_fr = fr
            assert best is not None
            record = {
                "trajectory": best.to_dict(),
                "filter": {
                    "accepted": best_fr.accepted,
                    "reasons": best_fr.reasons,
                    "retrieved": sorted(best_fr.retrieved),
                    "missing_required": sorted(best_fr.missing_required),
                    "num_recalls": best_fr.num_recalls,
                    "answer_correct": best_fr.answer_correct,
                },
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            results.append(record)
            if best_fr.accepted:
                accepted += 1
    print(f"[{uid}] accepted={accepted}/{total}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", nargs="+", default=[f"u{i:03d}" for i in range(10)])
    ap.add_argument("--out_dir", default=str(OUT_ROOT / "trajectories_raw"))
    ap.add_argument("--base_url", default="http://127.0.0.1:8002/v1")
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_turns", type=int, default=10)
    ap.add_argument("--n_attempts", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = AgentConfig(
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_turns=args.max_turns,
    )
    out_dir = Path(args.out_dir)

    t0 = time.time()
    agg_accepted = 0
    agg_total = 0
    by_schema = {}
    for uid in args.uids:
        print(f"=== {uid} ===")
        results = run_for_user(uid, cfg, out_dir, n_attempts=args.n_attempts, verbose=args.verbose)
        for r in results:
            agg_total += 1
            sch = r["trajectory"]["schema"]
            by_schema.setdefault(sch, {"acc": 0, "tot": 0})
            by_schema[sch]["tot"] += 1
            if r["filter"]["accepted"]:
                agg_accepted += 1
                by_schema[sch]["acc"] += 1

    summary = {
        "n_users": len(args.uids),
        "accepted": agg_accepted,
        "total": agg_total,
        "rate": round(agg_accepted / max(agg_total, 1), 3),
        "by_schema": by_schema,
        "wall_seconds": round(time.time() - t0, 1),
        "cfg": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
