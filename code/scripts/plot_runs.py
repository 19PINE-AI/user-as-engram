"""
Plot training and evaluation curves for base vs Engram runs.
Reads engram_runs/{base_d*,engram_d*}/train_log.jsonl and produces
- a loss curve comparison
- a val_bpb curve comparison
"""
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path):
    train_steps, train_losses = [], []
    eval_steps, eval_bpb = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if row["type"] == "train":
                train_steps.append(row["step"]); train_losses.append(row["train_loss"])
            elif row["type"] == "eval":
                eval_steps.append(row["step"]); eval_bpb.append(row["val_bpb"])
    return (train_steps, train_losses), (eval_steps, eval_bpb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-log", required=True)
    p.add_argument("--engram-log", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    (b_ts, b_ls), (b_es, b_bpb) = load_log(args.base_log)
    (e_ts, e_ls), (e_es, e_bpb) = load_log(args.engram_log)

    # Train loss
    plt.figure(figsize=(10, 5))
    plt.plot(b_ts, b_ls, alpha=0.5, label="base train loss")
    plt.plot(e_ts, e_ls, alpha=0.5, label="engram train loss")
    plt.xlabel("step"); plt.ylabel("train loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title("Training loss: base vs Engram (Mini-Engram pilot)")
    plt.tight_layout(); plt.savefig(out / "train_loss.png", dpi=120); plt.close()

    # Val bpb
    plt.figure(figsize=(10, 5))
    plt.plot(b_es, b_bpb, marker="o", label="base val bpb")
    plt.plot(e_es, e_bpb, marker="s", label="engram val bpb")
    plt.xlabel("step"); plt.ylabel("val bpb (lower=better)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.title("Validation bits-per-byte: base vs Engram (Mini-Engram pilot)")
    plt.tight_layout(); plt.savefig(out / "val_bpb.png", dpi=120); plt.close()

    # Print final numbers
    if b_bpb and e_bpb:
        print(f"Final base val_bpb:    {b_bpb[-1]:.4f}")
        print(f"Final engram val_bpb:  {e_bpb[-1]:.4f}")
        print(f"Delta (base - engram): {b_bpb[-1] - e_bpb[-1]:+.4f}")
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()
