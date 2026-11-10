"""T6 full design on Qwen3-8B with batched edits.

Phases: setup -> lr -> sweep (n=1, all 200 facts, every method x layer) -> multi (n=10, n=50; shared vs
per-fact units; every method x layer). One JSON line per edit in runs.jsonl; resumable by key.
"""
import argparse, json, os, sys, time, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t6lib as L
import t6blib as B

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--budget", type=float, default=24)
ap.add_argument("--phase", required=True, choices=["setup", "lr", "sweep", "multi", "supp"])
ap.add_argument("--out", default=os.path.expanduser("~/clean-tests/results/T6"))
ap.add_argument("--methods", default="lora_down,lora_up,neuron,neuron_bias,rome,qkv,"
                                     "lora_down+reg,lora_up+reg,neuron+reg,neuron_bias+reg,qkv+reg")
ap.add_argument("--layers", default="")
ap.add_argument("--ns", default="10,50")
ap.add_argument("--E", type=int, default=16)
ap.add_argument("--sweep_text_chunks", type=int, default=4)
ap.add_argument("--multi_text_chunks", type=int, default=20)
ap.add_argument("--n_supp", type=int, default=40)
ap.add_argument("--runs_file", default="runs.jsonl", help="shard file this process appends to")
ap.add_argument("--anchor_group", type=int, default=4)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
OUT = os.path.join(args.out, args.runs_file)
done = set()
import glob as _glob
for _fp in sorted(_glob.glob(os.path.join(args.out, "runs*.jsonl"))):  # skip keys done in ANY shard
    for line in open(_fp):
        try:
            done.add(json.loads(line)["key"])
        except Exception:
            pass

model, tok = L.load(args.model, args.budget)
NL = model.config.num_hidden_layers
LAYERS = [int(x) for x in args.layers.split(",")] if args.layers else list(range(3, NL, 4))
MID = LAYERS[len(LAYERS) // 2]

from datasets import load_dataset
wt_test = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"] if len(t) > 200]
wt_val = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"] if len(t) > 200]
ids = tok("\n\n".join(wt_test), add_special_tokens=False).input_ids
CHUNKS = [torch.tensor(ids[i:i + 1024]) for i in range(0, 1024 * 100, 1024)]

SETUP = os.path.join(args.out, "setup.json")
if not os.path.exists(SETUP):
    ok_c = []
    for c, k in L.CITIES:
        a, _ = L.acc_on(model, tok, [(f"{c} is a city in the country of", k)])
        if a == 1.0:
            ok_c.append((c, k))
    facts = L.make_facts(ok_c, 300, seed=0)
    ev = facts[5:205]
    base = {}
    base["direct"], _ = L.acc_on(model, tok, [(L.DIRECT.format(name=f["name"]), f["city"]) for f in ev])
    base["para"], _ = L.acc_on(model, tok, [(t.format(name=f["name"]), f["city"]) for f in ev for t in L.PARA])
    base["twohop"], _ = L.acc_on(model, tok, [(t.format(name=f["name"]), f["country"]) for f in ev for t in L.TWOHOP])
    base["icl_para"], _ = L.acc_on(model, tok, [(f"{f['name']} lives in {f['city']}. " + t.format(name=f["name"]), f["city"]) for f in ev for t in L.PARA])
    base["icl_twohop"], _ = L.acc_on(model, tok, [(f"{f['name']} lives in {f['city']}. " + t.format(name=f["name"]), f["country"]) for f in ev for t in L.TWOHOP])
    json.dump({"model": args.model, "cities_kept": ok_c, "n_cities_total": len(L.CITIES), "facts": facts,
               "splits": {"dev": [0, 5], "main": [5, 205], "anchor_names": [205, 260], "others": [260, 300]},
               "base": base}, open(SETUP, "w"), indent=1)
setup = json.load(open(SETUP))
FACTS = setup["facts"]
DEVF, MAIN, ANCH, OTHERS = FACTS[0:5], FACTS[5:205], FACTS[205:260], FACTS[260:300]
print("setup", json.dumps(setup["base"]), "cities", len(setup["cities_kept"]), flush=True)
ANCHORS = L.make_anchors(tok, [f["name"] for f in ANCH], wt_val[-200:])

COV = {}
def cov(layer):
    if layer not in COV:
        p = os.path.join(args.out, f"cov_L{layer}.pt")
        if not os.path.exists(p):
            c, _ = L.collect_cov(model, tok, wt_val[:-200], [layer], n_tokens=100_000)
            torch.save(c[layer], p)
        COV[layer] = torch.load(p)
    return COV[layer]

LRP = os.path.join(args.out, "lr.json")
LR = json.load(open(LRP)) if os.path.exists(LRP) else {}


def run_batch(method, layer, groups, keys, r, chunks, extra=None, lr=None, lam=None):
    todo = [i for i, k in enumerate(keys) if k not in done]
    if not todo:
        return
    groups = [groups[i] for i in todo]
    keys = [keys[i] for i in todo]
    reg = method.endswith("+reg")
    base_m = method.replace("+reg", "")
    kind = "neuron" if base_m.startswith("neuron") else base_m
    if kind != "rome" and lr is None:
        lr = LR[base_m]
    t0 = time.time()
    seeds = [zlib.crc32(k.encode()) % 100_000 for k in keys]
    ed = B.BEdit(model, kind, len(groups), layer=None if kind == "qkv" else layer, r=r,
                 bias=(base_m == "neuron_bias"), seeds=seeds)
    try:
        if kind == "rome":
            lam_ = lam if lam is not None else LR.get("rome_lam", 1000.0)
            fits = B.rome_batched(model, tok, ed, groups, cov(layer), lam=lam_)
        else:
            heavy = kind == "qkv"  # LoRA at every layer -> full-depth backprop; keep activations small
            fits = B.train_batched(model, tok, ed, groups, lr=lr, anchors=ANCHORS if reg else None,
                                   mb=10 if heavy else 40, anchor_group=1 if heavy else args.anchor_group)
        res = B.evaluate_batched(model, tok, ed, groups, OTHERS, chunks)
        npe = ed.n_params_per_edit()
    finally:
        ed.remove()
        del ed
        torch.cuda.empty_cache()
    sec = time.time() - t0
    with open(OUT, "a") as fh:
        for k, g, fit, rs in zip(keys, groups, fits, res):
            rec = {"key": k, "method": method, "layer": layer, "n_facts": len(g), "r": r, "lr": lr,
                   "fact_ids": [f["id"] for f in g], "n_params": npe, "fit": fit, **rs,
                   "batch_sec": sec, "batch_E": len(groups), **(extra or {})}
            fh.write(json.dumps(rec) + "\n")
            done.add(k)
    d = sum(x["direct"] for x in res) / len(res)
    p = sum(x["para"] for x in res) / len(res)
    h = sum(x["twohop"] for x in res) / len(res)
    b = sum(x["other_subject_target_p_edit"] for x in res) / len(res)
    tk = sum(x["text"]["kl_per_tok"] for x in res) / len(res)
    print(f"{keys[0].rsplit('|', 1)[0]} x{len(keys)}: direct={d:.2f} para={p:.2f} 2hop={h:.2f} bleed={b:.3f} "
          f"textKL={tk:.2e} {sec:.0f}s", flush=True)


methods = args.methods.split(",")

if args.phase == "setup":
    pass

elif args.phase == "lr":
    grid = {"lora_down": [1e-3, 3e-3, 1e-2, 3e-2], "lora_up": [1e-3, 3e-3, 1e-2, 3e-2],
            "neuron": [1e-3, 3e-3, 1e-2, 3e-2], "neuron_bias": [1e-3, 3e-3, 1e-2, 3e-2],
            "qkv": [1e-4, 3e-4, 1e-3, 3e-3]}
    for m in [m for m in methods if not m.endswith("+reg") and m != "rome"]:
        for lr in grid[m]:
            keys = [f"lr|{m}|L{MID}|lr{lr}|f{f['id']}" for f in DEVF]
            run_batch(m, MID, [[f] for f in DEVF], keys, 16 if m == "qkv" else 1, CHUNKS[:2], extra={"lr_probe": lr}, lr=lr)
    for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        run_batch("rome", MID, [DEVF], [f"lr|rome5|L{MID}|lam{lam}"], len(DEVF), CHUNKS[:2], extra={"lam": lam}, lam=lam)
    rows = [json.loads(l) for fp in sorted(_glob.glob(os.path.join(args.out, "runs*.jsonl"))) for l in open(fp) if l.startswith('{"key": "lr|')]
    for m in [m for m in methods if not m.endswith("+reg") and m != "rome"]:
        by = {}
        for r_ in rows:
            if r_["method"] == m:
                by.setdefault(r_["lr"], []).append(r_["direct"])
        scored = sorted((lr, sum(v) / len(v)) for lr, v in by.items())
        good = [lr for lr, a in scored if a >= 0.9]
        LR[m] = good[0] if good else max(scored, key=lambda x: x[1])[0]
    lams = [(r_["lam"], r_["direct"]) for r_ in rows if r_["method"] == "rome" and "lam" in r_]
    okl = [lam for lam, a in lams if a >= 0.9]
    LR["rome_lam"] = max(okl) if okl else max(lams, key=lambda x: x[1])[0]
    json.dump(LR, open(LRP, "w"), indent=1)
    print("LR", LR, flush=True)

elif args.phase == "sweep":
    E = args.E
    for layer in LAYERS:
        for m in methods:
            if m.startswith("qkv"):
                continue
            Em = 4 if m.endswith("+reg") else E
            for s in range(0, len(MAIN), Em):
                fs = MAIN[s:s + Em]
                run_batch(m, layer, [[f] for f in fs], [f"sweep|{m}|L{layer}|f{f['id']}" for f in fs], 1,
                          CHUNKS[:args.sweep_text_chunks])
    for m in [m for m in methods if m.startswith("qkv")]:
        Em = 2 if m.endswith("+reg") else 4
        for s in range(0, len(MAIN), Em):
            fs = MAIN[s:s + Em]
            run_batch(m, None, [[f] for f in fs], [f"sweep|{m}|all|f{f['id']}" for f in fs], 16,
                      CHUNKS[:args.sweep_text_chunks])

elif args.phase == "multi":
    for n in [int(x) for x in args.ns.split(",")]:
        groups = [MAIN[i:i + n] for i in range(0, len(MAIN), n)]
        Eg = max(1, 40 // n)
        for layer in LAYERS:
            for m in methods:
                if m.startswith("qkv"):
                    continue
                shares = ["rank-n"] if m == "rome" else ["shared", "perfact"]
                for share in shares:
                    r = 1 if share == "shared" else n
                    for s in range(0, len(groups), Eg):
                        gs = groups[s:s + Eg]
                        keys = [f"multi|{m}|L{layer}|n{n}|{share}|g{s + j}" for j in range(len(gs))]
                        run_batch(m, layer, gs, keys, r, CHUNKS[:args.multi_text_chunks], extra={"share": share})
        for m in [m for m in methods if m.startswith("qkv")]:
            for s in range(0, len(groups), 1):
                gs = groups[s:s + 1]
                keys = [f"multi|{m}|all|n{n}|r16|g{s + j}" for j in range(len(gs))]
                run_batch(m, None, gs, keys, 16, CHUNKS[:args.multi_text_chunks], extra={"share": "r16"})
elif args.phase == "supp":
    # supplementary single-fact edits with the other-subject firing metric (keys distinct from sweep)
    for layer in LAYERS:
        for m in methods:
            Em = 4 if m.endswith("+reg") else args.E
            fs_all = MAIN[:args.n_supp]
            for s in range(0, len(fs_all), Em):
                fs = fs_all[s:s + Em]
                run_batch(m, layer, [[f] for f in fs], [f"supp|{m}|L{layer}|f{f['id']}" for f in fs], 1,
                          CHUNKS[:args.sweep_text_chunks])
print("PHASE DONE", args.phase, flush=True)
