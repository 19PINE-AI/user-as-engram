"""
Evaluate the EngramServer on a multi-user workload.

Procedure:
  1. Register N users (each with K facts; train Joint OPT).
  2. Optionally register an org with corporate facts.
  3. Issue M requests by sampling random (user, fact) pairs and the user's
     own question. Measure end-to-end latency, recall, and verify
     cross-user privacy by mixing in a few "wrong-user" probes.

Metrics:
  - per-request latency: apply / forward / restore
  - throughput: requests / second
  - own-user fact recall: top-1, top-5
  - cross-user leak rate: when serving user U with user V's question, is
    the answer ever V's gold (it must not be — overrides are restored
    between users)

Usage:
  python -m scripts.eval_serving \\
      --ckpt-dir $NANOCHAT_BASE_DIR/engram_runs/engram_d12 \\
      --corpus /home/ubuntu/user-as-engram/data/corpora_xxl.json \\
      --n-users 30 --facts-per-user 50 --n-requests 600
"""
import os, json, argparse, time, random
from pathlib import Path
import torch

from scripts.engram_server import EngramServer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--corpus", default="/home/ubuntu/user-as-engram/data/corpora_xxl.json")
    p.add_argument("--out", default="/home/ubuntu/user-as-engram/results/serving_eval.json")
    p.add_argument("--n-users", type=int, default=30)
    p.add_argument("--facts-per-user", type=int, default=50)
    p.add_argument("--n-requests", type=int, default=600)
    p.add_argument("--cross-user-frac", type=float, default=0.2,
                    help="fraction of requests where we deliberately query under WRONG user (privacy probe)")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--scale", type=float, default=20.0)
    args = p.parse_args()

    rng = random.Random(42)
    print(f"=== EngramServer eval ===")
    print(f"ckpt: {args.ckpt_dir}")
    server = EngramServer(args.ckpt_dir)

    with open(args.corpus) as f:
        corpora = json.load(f)
    multi_users = corpora["multi_users"]
    if args.n_users > len(multi_users):
        raise ValueError(f"Only {len(multi_users)} users in corpus")

    # Subset and dedupe each user's facts to args.facts_per_user
    test_users = []
    for u in multi_users[:args.n_users]:
        seen = {}
        for f in u["facts"]:
            seen[f["trigger"]] = f
            if len(seen) >= args.facts_per_user: break
        test_users.append({"user_id": u["user_id"], "facts": list(seen.values())[:args.facts_per_user]})

    # ----- Register users -----
    print(f"\nRegistering {len(test_users)} users (Joint OPT, steps={args.steps})...")
    register_t0 = time.time()
    register_meta = []
    for u in test_users:
        info = server.register_user(str(u["user_id"]), u["facts"],
                                       training='joint_opt', steps=args.steps,
                                       lr=args.lr, scale=args.scale)
        register_meta.append(info)
    register_total = time.time() - register_t0
    avg_train_time = sum(m["train_time_s"] for m in register_meta) / len(register_meta)
    avg_size_kb = sum(m["size_kb"] for m in register_meta) / len(register_meta)
    print(f"All users registered in {register_total:.0f}s "
          f"(avg train {avg_train_time:.1f}s, avg storage {avg_size_kb:.1f} KB/user)")

    # ----- Issue requests -----
    print(f"\nIssuing {args.n_requests} requests...")
    request_log = []
    own_top1 = own_top5 = 0
    own_total = 0
    leak_count = 0
    leak_total = 0
    t0 = time.time()
    for i in range(args.n_requests):
        # Pick a user to QUERY about (fact owner)
        fact_user = rng.choice(test_users)
        fact = rng.choice(fact_user["facts"])
        gold_first = server.tokenizer.encode(fact["gold"])[0]

        if rng.random() < args.cross_user_frac:
            # Privacy probe: serve under a DIFFERENT user's overrides; their
            # answer should NOT be fact_user's gold
            wrong_user = rng.choice([u for u in test_users if u["user_id"] != fact_user["user_id"]])
            served_uid = str(wrong_user["user_id"])
            mode = "cross_user_probe"
        else:
            served_uid = str(fact_user["user_id"])
            mode = "own_query"

        t_req = time.time()
        out = server.serve(served_uid, fact["prompt"], max_tokens=1)
        elapsed_ms = (time.time() - t_req) * 1000

        is_match = (out["top1_id"] == gold_first)
        is_top5 = (gold_first in out["top5_ids"])

        if mode == "own_query":
            own_total += 1
            if is_match: own_top1 += 1
            if is_top5: own_top5 += 1
        else:
            leak_total += 1
            # If the wrong user's overrides somehow give us fact_user's gold, that's a leak
            if is_match: leak_count += 1

        request_log.append({
            "i": i, "mode": mode,
            "served_uid": served_uid,
            "fact_user": fact_user["user_id"],
            "prompt": fact["prompt"], "expected_gold": fact["gold"],
            "got_top1": out["top1_text"],
            "is_match": int(is_match), "is_top5": int(is_top5),
            "elapsed_ms": elapsed_ms, "timings": out["timings"],
            "n_active_maps": out["n_active_maps"],
        })

        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            tps = (i+1) / elapsed
            print(f"  [{i+1}/{args.n_requests}]  own top-1: {own_top1}/{own_total}={own_top1/max(own_total,1):.1%}"
                  f"  leaks: {leak_count}/{leak_total}={leak_count/max(leak_total,1):.4f}"
                  f"  throughput: {tps:.1f} req/s")

    total_t = time.time() - t0

    # Latency stats
    own_lat = [r["elapsed_ms"] for r in request_log if r["mode"] == "own_query"]
    own_apply = [r["timings"]["apply_ms"] for r in request_log if r["mode"] == "own_query"]
    own_forward = [r["timings"]["forward_ms"] for r in request_log if r["mode"] == "own_query"]
    own_restore = [r["timings"]["restore_ms"] for r in request_log if r["mode"] == "own_query"]

    def percentile(xs, p):
        xs_sorted = sorted(xs)
        return xs_sorted[min(int(p * len(xs_sorted)), len(xs_sorted) - 1)]

    summary = {
        "config": vars(args),
        "register": {
            "total_time_s": register_total,
            "avg_user_train_s": avg_train_time,
            "avg_user_storage_kb": avg_size_kb,
            "n_users": len(test_users),
        },
        "throughput": {
            "total_requests": args.n_requests,
            "total_time_s": total_t,
            "requests_per_second": args.n_requests / total_t,
        },
        "latency_ms": {
            "p50": percentile(own_lat, 0.5),
            "p90": percentile(own_lat, 0.9),
            "p99": percentile(own_lat, 0.99),
            "apply_p50": percentile(own_apply, 0.5),
            "apply_p99": percentile(own_apply, 0.99),
            "forward_p50": percentile(own_forward, 0.5),
            "restore_p50": percentile(own_restore, 0.5),
        },
        "recall": {
            "own_top1": own_top1,
            "own_top5": own_top5,
            "own_total": own_total,
            "own_top1_rate": own_top1 / max(own_total, 1),
            "own_top5_rate": own_top5 / max(own_total, 1),
        },
        "privacy": {
            "leak_count": leak_count,
            "leak_total": leak_total,
            "leak_rate": leak_count / max(leak_total, 1),
        },
    }
    print(f"\n=== Serving evaluation summary ===")
    print(f"  registration: {len(test_users)} users in {register_total:.0f}s ({avg_train_time:.1f}s/user, {avg_size_kb:.1f} KB/user)")
    print(f"  throughput: {summary['throughput']['requests_per_second']:.1f} req/s")
    print(f"  latency p50/p90/p99: {summary['latency_ms']['p50']:.1f} / {summary['latency_ms']['p90']:.1f} / {summary['latency_ms']['p99']:.1f} ms")
    print(f"  apply latency p50/p99: {summary['latency_ms']['apply_p50']:.2f} / {summary['latency_ms']['apply_p99']:.2f} ms")
    print(f"  recall own-fact: top-1 {own_top1}/{own_total}={own_top1/max(own_total,1):.1%}  top-5 {own_top5}/{own_total}={own_top5/max(own_total,1):.1%}")
    print(f"  cross-user leak: {leak_count}/{leak_total}={leak_count/max(leak_total,1):.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "requests": request_log[:200]}, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
