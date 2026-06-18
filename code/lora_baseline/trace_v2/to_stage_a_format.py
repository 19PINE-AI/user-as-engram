"""Convert v2 rewriter output (data/trace_v2/traces/uXXX.jsonl) into the
Stage A recite trainer's expected trace format (data/traces_v2/uXXX.jsonl).

Rewriter format (per line):
  {uid, schema, question, answer, gold, required_fact_keys, think, sft_text}

Stage A recite format (per line):
  {question, trace}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            q = rec.get("question")
            trace = rec.get("sft_text")
            if not q or not trace:
                continue
            fout.write(json.dumps({"question": q, "trace": trace}) + "\n")
            written += 1
    return total, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/trace_v2/traces")
    ap.add_argument("--out_dir", default="data/traces_v2")
    ap.add_argument("--uids", nargs="+", default=[f"u{i:03d}" for i in range(20)])
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    in_dir = (repo / args.in_dir).resolve()
    out_dir = (repo / args.out_dir).resolve()
    counts = {}
    for uid in args.uids:
        src = in_dir / f"{uid}.jsonl"
        if not src.exists():
            print(f"missing: {src}")
            continue
        total, written = convert_file(src, out_dir / f"{uid}.jsonl")
        counts[uid] = {"total": total, "written": written}
        print(f"[{uid}] total={total} written={written}")
    print(json.dumps({"out_dir": str(out_dir), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
