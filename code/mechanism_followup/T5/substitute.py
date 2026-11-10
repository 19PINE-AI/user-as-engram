"""Gao-style substitution: replace a neuron's activation on each sim window with the MiMo-simulated
activation (scaled to the neuron's max), and measure next-token KL vs the clean model on positions
from the window start to the end of the sequence. Baselines: zero-ablation and mean-ablation of the
same positions. damage_avoided = 1 - KL_sim / KL_zero.
Output: <dir>/subst.json
"""
import argparse, json, os
import torch
import torch.nn.functional as F
from audit_lib import cap_gpu, load_model, MLPTap, logits_of

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--state", default=None)
ap.add_argument("--dir", required=True)
ap.add_argument("--gb", type=float, default=2.5)
args = ap.parse_args()
cap_gpu(args.gb)
model, cfg, tok = load_model(args.ckpt, state_path=args.state)
tap = MLPTap(model)
neurons = json.load(open(os.path.join(args.dir, "neurons.json")))
sims = json.load(open(os.path.join(args.dir, "sims.json")))
sim_seqs = torch.load(os.path.join(args.dir, "sim_seqs.pt"))
W = 32
out = {}


def kl_from(clean_lp, lp, starts):
    kl = (clean_lp.exp() * (clean_lp - lp)).sum(-1)  # [B,T]
    T = kl.shape[1]
    mask = torch.arange(T, device=kl.device)[None, :] >= torch.tensor(starts, device=kl.device)[:, None]
    return float((kl * mask).sum() / mask.sum())


for n in neurons:
    k = f"{n['layer']}:{n['neuron']}"
    s = sims.get(k)
    if not s or "preds" not in s:
        continue
    li, j = n["layer"], n["neuron"]
    ws = n["sim_windows"]
    scale = n["max_post"] / 10.0
    import numpy as np
    xr = np.concatenate([np.array(w["acts"]) for w in ws]); xs = np.concatenate([np.array(p, float) for p in s["preds"]])
    A = np.stack([xs, np.ones_like(xs)], 1); cal_a, cal_b = np.linalg.lstsq(A, xr, rcond=None)[0]
    tot = {m: 0.0 for m in ("identity", "sim", "simcal", "zero", "mean")}; cnt = 0
    for c0 in range(0, len(ws), 2):
        cw = ws[c0:c0 + 2]; cp = s["preds"][c0:c0 + 2]
        idx = sim_seqs[[w["seq"] for w in cw]].cuda()
        starts = [w["start"] for w in cw]
        tap.want = {li}; tap.override = None
        clean_lp = F.log_softmax(logits_of(model, idx), -1)
        post_clean = F.relu(tap.capture[li][..., j].float()).square()
        npos = sum(256 - a for a in starts)
        for mode in tot:
            p = post_clean.clone()
            for b, (w, pr) in enumerate(zip(cw, cp)):
                a = w["start"]
                if mode == "sim":
                    p[b, a:a + W] = torch.tensor(pr, device=p.device) * scale
                elif mode == "simcal":
                    p[b, a:a + W] = (torch.tensor(pr, device=p.device) * float(cal_a) + float(cal_b)).clamp(min=0)
                elif mode == "zero":
                    p[b, a:a + W] = 0
                elif mode == "mean":
                    p[b, a:a + W] = n["mean_post"]
            tap.override = (li, j, p)
            lp = F.log_softmax(logits_of(model, idx), -1)
            tot[mode] += kl_from(clean_lp, lp, starts) * npos
            del lp
        tap.override = None; cnt += npos
        del clean_lp
    res = {m: v / cnt for m, v in tot.items()}
    f = res["identity"]
    res["damage_avoided_vs_zero"] = 1 - (res["sim"] - f) / (res["zero"] - f) if res["zero"] - f > 0 else None
    res["cal_damage_avoided_vs_zero"] = 1 - (res["simcal"] - f) / (res["zero"] - f) if res["zero"] - f > 0 else None
    res["cal_a"], res["cal_b"] = float(cal_a), float(cal_b)
    res["damage_avoided_vs_mean"] = 1 - (res["sim"] - f) / (res["mean"] - f) if res["mean"] - f > 0 else None
    out[k] = res
json.dump(out, open(os.path.join(args.dir, "subst.json"), "w"))
print("done", len(out))
