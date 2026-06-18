"""Plot LogitLens KL by layer for base vs Engram."""
import os
UAE_ROOT = os.environ.get("USER_AS_ENGRAM_ROOT") or (
    os.path.dirname(os.environ["NANOCHAT_BASE_DIR"]) if os.environ.get("NANOCHAT_BASE_DIR")
    else os.getcwd())
import json
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mechanistic", default=f"{UAE_ROOT}/results/mechanistic_d8.json")
    p.add_argument("--out-dir", default=f"{UAE_ROOT}/results/figs")
    args = p.parse_args()

    with open(args.mechanistic) as f:
        d = json.load(f)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: LogitLens KL by layer
    eng_kl = d["engram"]["logitlens_kl"]
    base_kl = d.get("base", {}).get("logitlens_kl") if "base" in d else None
    eng_layers = d["engram"]["engram_layers"]
    layers = list(range(len(eng_kl)))

    plt.figure(figsize=(8, 5))
    if base_kl is not None:
        plt.plot(layers, base_kl, marker='o', label='base', color='#888')
    plt.plot(layers, eng_kl, marker='s', label='engram', color='#c33')
    for el in eng_layers:
        plt.axvline(x=el, linestyle='--', alpha=0.4, color='#33c', label=f'Engram inserted at L{el}' if el == eng_layers[0] else None)
    plt.xlabel("Layer index")
    plt.ylabel("KL(layer logits || final logits)")
    plt.title("LogitLens KL — base vs Engram (Mini-Engram d8)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "logitlens_kl.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    # Plot 2: Insertion attribution per layer per position
    attr = d["engram"]["insertion_attribution"]
    per_layer_per_pos = attr["per_layer_per_pos"]
    trig_pos = attr["trig_pos"]
    n_layers = len(per_layer_per_pos)
    n_pos = len(per_layer_per_pos[0])

    import numpy as np
    arr = np.array(per_layer_per_pos)  # [n_layers, n_pos]
    plt.figure(figsize=(10, 4))
    im = plt.imshow(arr, aspect='auto', cmap='hot', interpolation='nearest')
    plt.colorbar(im, label="L2 of (after - before) at each (layer, position)")
    plt.axvline(x=trig_pos, linestyle='--', color='#0c0', label=f'trigger position {trig_pos}')
    plt.xlabel("Token position")
    plt.ylabel("Layer index")
    plt.title("Insertion attribution: per-layer per-position L2 change\n(prompt: 'Vandelay\\'s customer support email starts with')")
    plt.legend(loc='upper left')
    plt.tight_layout()
    out_path = out_dir / "insertion_attribution.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
