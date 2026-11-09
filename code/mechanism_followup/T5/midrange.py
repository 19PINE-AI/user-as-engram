"""Label coverage on weaker activations: for each audited neuron, sample 10 contexts where its post-activation
is between 10% and 50% of its max (from the same held-out collection split), plus 5 zero-activation controls.
GPU part writes <dir>/midrange.json; --judge asks MiMo whether each context fits the existing label.
"""
import argparse, asyncio, json, os, random
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt"); ap.add_argument("--dir", required=True)
ap.add_argument("--n-tokens", type=int, default=500_000)
ap.add_argument("--gb", type=float, default=2.0)
ap.add_argument("--judge", action="store_true")
args = ap.parse_args()
MP = os.path.join(args.dir, "midrange.json")

if not args.judge:
    from audit_lib import cap_gpu, load_model, corpus_tokens, MLPTap, logits_of
    cap_gpu(args.gb)
    model, cfg, tok = load_model(args.ckpt)
    tap = MLPTap(model); tap.want = set(range(cfg.n_layer))
    N = json.load(open(os.path.join(args.dir, "neurons.json")))
    seqs, _ = corpus_tokens(tok, args.n_tokens, 256, skip_docs=0)  # subset of the collection split
    T = 256
    by_layer = {}
    for n in N:
        by_layer.setdefault(n["layer"], []).append(n)
    rng = random.Random(0)
    cand = {f"{n['layer']}:{n['neuron']}": {"mid": [], "zero": []} for n in N}
    for i in range(0, len(seqs), 8):
        logits_of(model, seqs[i:i + 8].cuda())
        for li, ns in by_layer.items():
            idx = torch.tensor([n["neuron"] for n in ns], device="cuda")
            post = F.relu(tap.capture[li].float()).square()[..., idx].reshape(-1, len(ns))  # [N, n]
            mx = torch.tensor([n["max_post"] for n in ns], device="cuda")
            frac = post / mx
            mid_w = ((frac >= 0.1) & (frac <= 0.5)).float().T  # [n, N]
            zero_w = (frac == 0).float().T
            has_m = mid_w.sum(1) > 0; has_z = zero_w.sum(1) > 0
            pm = torch.multinomial(mid_w + (~has_m)[:, None].float(), 3, replacement=True)  # [n,3]
            pz = torch.multinomial(zero_w + (~has_z)[:, None].float(), 1)
            fm = torch.gather(frac.T, 1, pm)
            pm, fm, pz = pm.tolist(), fm.tolist(), pz.flatten().tolist()
            hm, hz = has_m.tolist(), has_z.tolist()
            for jj, n in enumerate(ns):
                k = f"{li}:{n['neuron']}"
                if hm[jj]:
                    cand[k]["mid"].extend((i * T + p, f) for p, f in zip(pm[jj], fm[jj]))
                if hz[jj] and len(cand[k]["zero"]) < 20:
                    cand[k]["zero"].append((i * T + pz[jj], 0.0))
    out = {}
    for k, c in cand.items():
        mid = rng.sample(c["mid"], min(10, len(c["mid"])))
        zero = rng.sample(c["zero"], min(5, len(c["zero"])))
        ctxs = []
        for flat, fr in mid + zero:
            s, p = divmod(flat, T)
            a = max(0, p - 16)
            toks = [tok.decode([t]) for t in seqs[s, a:p + 5].tolist()]
            ctxs.append({"toks": toks, "mark": p - a, "frac": fr, "kind": "mid" if fr > 0 else "zero"})
        out[k] = ctxs
    json.dump(out, open(MP, "w"))
    print("wrote", len(out))
else:
    from mimo import ask_json
    M = json.load(open(MP)); L = json.load(open(os.path.join(args.dir, "labels.json")))
    JP = os.path.join(args.dir, "midrange_judge.json")
    res = json.load(open(JP)) if os.path.exists(JP) else {}

    def prompt(lab, ctxs):
        lines = []
        for i, c in enumerate(ctxs):
            t = [x.replace("\n", "\\n") for x in c["toks"]]; t[c["mark"]] = f"«{t[c['mark']]}»"
            lines.append(f"{i}: {''.join(t)}")
        return f"""A language-model neuron was described as: "{lab}".
For each context below, judge whether the marked token «like this» fits that description (i.e. whether the description would predict the neuron to be at least weakly active there).

{chr(10).join(lines)}

Reply with only JSON: {{"fits": [true/false for each context, in order]}}"""

    async def one(k):
        lab = L.get(k, {}).get("label"); ctxs = M.get(k, [])
        if not lab or not ctxs or k in res:
            return
        order = list(range(len(ctxs))); random.Random(hash(k) & 0xffff).shuffle(order)
        shuffled = [ctxs[i] for i in order]
        try:
            j, _ = await ask_json(prompt(lab, shuffled), max_tokens=800)
            fits = [bool(x) for x in j["fits"]][:len(shuffled)]
            back = [None] * len(ctxs)
            for pos, i in enumerate(order):
                back[i] = fits[pos] if pos < len(fits) else None
            mid = [f for f, c in zip(back, ctxs) if c["kind"] == "mid" and f is not None]
            zero = [f for f, c in zip(back, ctxs) if c["kind"] == "zero" and f is not None]
            res[k] = {"mid_fit": sum(mid) / len(mid) if mid else None,
                      "zero_fit": sum(zero) / len(zero) if zero else None}
        except Exception as e:
            res[k] = {"error": repr(e)}

    async def main():
        ks = list(M)
        for i in range(0, len(ks), 40):
            await asyncio.gather(*(one(k) for k in ks[i:i + 40]))
            json.dump(res, open(JP, "w")); print(f"judge {min(i+40,len(ks))}/{len(ks)}", flush=True)
    asyncio.run(main())
