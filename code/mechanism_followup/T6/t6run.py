"""T6 runner. Phases: setup, lr, sweep, single, multi. Results appended to results/T6/runs.jsonl."""
import argparse, json, os, sys, time, random, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t6lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--budget", type=float, default=24)
ap.add_argument("--phase", required=True)
ap.add_argument("--out", default=os.path.expanduser("~/clean-tests/results/T6"))
ap.add_argument("--tag", default="")
ap.add_argument("--layers", default="")
ap.add_argument("--methods", default="lora_down,lora_up,neuron,neuron_bias,rome,qkv,lora_down+reg,lora_up+reg,neuron+reg,neuron_bias+reg,qkv+reg")
ap.add_argument("--n_text_chunks", type=int, default=20)
ap.add_argument("--n_facts", type=int, default=10)
ap.add_argument("--lam", type=float, default=15000.0)
ap.add_argument("--best", default="", help="json {method: layer}; single/multi use these per-method layers")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
OUT = os.path.join(args.out, f"runs{args.tag}.jsonl")
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        try:
            done.add(json.loads(line)["key"])
        except Exception:
            pass

model, tok = L.load(args.model, args.budget)
NL = model.config.num_hidden_layers
LAYERS = [int(x) for x in args.layers.split(",")] if args.layers else list(range(3, NL, 4))
print("layers", LAYERS, flush=True)

from datasets import load_dataset
wt_test = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"] if len(t) > 200]
wt_val = [t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")["text"] if len(t) > 200]
ids = tok("\n\n".join(wt_test), add_special_tokens=False).input_ids
CHUNKS_ALL = [torch.tensor(ids[i:i + 1024]) for i in range(0, 1024 * 100, 1024)]
CHUNKS = CHUNKS_ALL[:args.n_text_chunks]

setup_path = os.path.join(args.out, f"setup{args.tag}.json")
if os.path.exists(setup_path):
    setup = json.load(open(setup_path))
else:
    # 1-hop filter: base model must know city -> country
    ok_c = []
    for c, k in L.CITIES:
        a, _ = L.acc_on(model, tok, [(f"{c} is a city in the country of", k)])
        if a == 1.0:
            ok_c.append((c, k))
    facts = L.make_facts(ok_c, 260, seed=0)
    # base knowledge of the fictional facts (should be ~0), and ICL two-hop upper bound
    ev = facts[:100]
    base_direct, _ = L.acc_on(model, tok, [(L.DIRECT.format(name=f["name"]), f["city"]) for f in ev])
    base_para, _ = L.acc_on(model, tok, [(t.format(name=f["name"]), f["city"]) for f in ev for t in L.PARA])
    base_2hop, _ = L.acc_on(model, tok, [(t.format(name=f["name"]), f["country"]) for f in ev for t in L.TWOHOP])
    icl_2hop, _ = L.acc_on(model, tok, [(f"{f['name']} lives in {f['city']}. " + t.format(name=f["name"]), f["country"]) for f in ev for t in L.TWOHOP])
    icl_para, _ = L.acc_on(model, tok, [(f"{f['name']} lives in {f['city']}. " + t.format(name=f["name"]), f["city"]) for f in ev for t in L.PARA])
    setup = {"model": args.model, "cities_kept": ok_c, "n_cities_total": len(L.CITIES), "facts": facts,
             "base": {"direct": base_direct, "para": base_para, "twohop": base_2hop,
                      "icl_twohop": icl_2hop, "icl_para": icl_para}}
    json.dump(setup, open(setup_path, "w"), indent=1)
print("setup", json.dumps(setup["base"]), "cities", len(setup["cities_kept"]), flush=True)
FACTS = setup["facts"]
DEVF, SWEEPF, SINGLEF, MULTIF, OTHERS = FACTS[0:5], FACTS[5:15], FACTS[15:65], FACTS[65:115], FACTS[220:260]

COV = {}
def cov(layer):
    if layer not in COV:
        p = os.path.join(args.out, f"cov{args.tag}_L{layer}.pt")
        if os.path.exists(p):
            COV[layer] = torch.load(p)
        else:
            c, n = L.collect_cov(model, tok, wt_val[:-200], [layer], n_tokens=100_000)
            torch.save(c[layer], p)
            COV[layer] = c[layer]
    return COV[layer]

LR_PATH = os.path.join(args.out, f"lr{args.tag}.json")
LR = json.load(open(LR_PATH)) if os.path.exists(LR_PATH) else {}
LAM_OVERRIDE = None
ANCHORS = L.make_anchors(tok, [f["name"] for f in FACTS[115:220]], wt_val[-200:])


def run_one(method, layer, facts, r, lr, key, extra=None, chunks=None):
    if key in done:
        return
    reg = method.endswith("+reg")
    base_m = method.replace("+reg", "")
    kind = "neuron" if base_m.startswith("neuron") else base_m
    if kind != "rome" and lr is None:
        lr = LR.get(base_m)
    t0 = time.time()
    torch.manual_seed(0)
    ed = L.Edit(model, kind, layer=None if kind == "qkv" else layer, r=r,
                bias=(base_m == "neuron_bias"), seed=zlib.crc32(key.encode()) % 10_000)
    try:
        if kind == "rome":
            lam = LAM_OVERRIDE if LAM_OVERRIDE is not None else LR.get("rome_lam", args.lam)
            fit = L.rome_fit(model, tok, ed, facts, cov(layer), lam=lam)
        else:
            fit = L.train_edit(model, tok, ed, facts, lr=lr, anchors=ANCHORS if reg else None)
        res = L.evaluate(model, tok, ed, facts, OTHERS, chunks or CHUNKS, None)
        rec = {"key": key, "method": method, "layer": layer, "n_facts": len(facts), "r": r, "lr": lr,
               "fact_ids": [f["id"] for f in facts], "n_params": ed.n_params(), "fit": fit, **res,
               "sec": time.time() - t0, **(extra or {})}
    finally:
        ed.remove()
        del ed
        torch.cuda.empty_cache()
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    done.add(key)
    print(f"{key}: direct={rec['direct']:.2f} para={rec['para']:.2f} 2hop={rec['twohop']:.2f} "
          f"textKL={rec['text']['kl_per_tok']:.2e} nbKL={rec['neighbor_kl']:.2e} "
          f"steps={rec['fit'].get('steps')} {rec['sec']:.0f}s", flush=True)


methods = args.methods.split(",")
mid = LAYERS[len(LAYERS) // 2]
BEST = json.load(open(args.best)) if args.best else None
def layers_for(m):
    if BEST is not None:
        return [BEST[m]] if m in BEST else []
    return LAYERS

if args.phase == "lr":
    grid = {"lora_down": [1e-3, 3e-3, 1e-2, 3e-2], "lora_up": [1e-3, 3e-3, 1e-2, 3e-2],
            "neuron": [1e-3, 3e-3, 1e-2, 3e-2], "neuron_bias": [1e-3, 3e-3, 1e-2, 3e-2],
            "qkv": [1e-4, 3e-4, 1e-3, 3e-3]}
    for lam in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        LAM_OVERRIDE = lam
        run_one("rome", mid, DEVF, 5, None, f"lr|rome5|L{mid}|lam{lam}", chunks=CHUNKS[:5], extra={"lam": lam})
    LAM_OVERRIDE = None
    for m in methods:
        if m == "rome" or m.endswith("+reg"):
            continue
        for lr in grid[m]:
            for f in DEVF:
                run_one(m, mid, [f], 1 if m != "qkv" else 16, lr, f"lr|{m}|L{mid}|lr{lr}|f{f['id']}",
                        chunks=CHUNKS[:5])
    # pick: smallest lr with mean direct >= 0.9 on dev; else best direct
    rows = [json.loads(l) for l in open(OUT) if l.startswith('{"key": "lr|')]
    lams = [(r_["lam"], r_["direct"]) for r_ in rows if r_["method"] == "rome" and "lam" in r_]
    okl = [lam for lam, a in lams if a >= 0.9]
    LR["rome_lam"] = max(okl) if okl else max(lams, key=lambda x: x[1])[0]
    for m in methods:
        if m == "rome" or m.endswith("+reg"):
            continue
        by = {}
        for r_ in rows:
            if r_["method"] == m:
                by.setdefault(r_["lr"], []).append(r_["direct"])
        scored = sorted((lr, sum(v) / len(v)) for lr, v in by.items())
        good = [lr for lr, a in scored if a >= 0.9]
        LR[m] = good[0] if good else max(scored, key=lambda x: x[1])[0]
    json.dump(LR, open(LR_PATH, "w"), indent=1)
    print("LR", LR, flush=True)

elif args.phase == "sweep":
    for layer in LAYERS:
        for m in methods:
            if m.startswith("qkv"):
                continue
            for f in SWEEPF[:args.n_facts]:
                run_one(m, layer, [f], 1, None, f"sweep|{m}|L{layer}|f{f['id']}")
    for qm in [m for m in methods if m.startswith("qkv")]:
        for f in SWEEPF[:args.n_facts]:
            run_one(qm, None, [f], 16, None, f"sweep|{qm}|all|f{f['id']}")

elif args.phase == "single":
    # 50 single-fact edits per method at its best layer, 100k-token text locality on first 10
    for m in methods:
        if m.startswith("qkv"):
            continue
        for layer in layers_for(m):
            for i, f in enumerate(SINGLEF[:args.n_facts]):
                run_one(m, layer, [f], 1, None, f"single|{m}|L{layer}|f{f['id']}",
                        chunks=CHUNKS_ALL if i < 10 else CHUNKS)
    for qm in [m for m in methods if m.startswith("qkv")]:
        for i, f in enumerate(SINGLEF[:args.n_facts]):
            run_one(qm, None, [f], 16, None, f"single|{qm}|all|f{f['id']}",
                    chunks=CHUNKS_ALL if i < 10 else CHUNKS)

elif args.phase == "multi":
    for n in (10, 50):
        fs = MULTIF[:n]
        for m in methods:
            if m.startswith("qkv"):
                continue
            for layer in layers_for(m):
                if m == "rome":
                    run_one(m, layer, fs, n, None, f"multi|rome|L{layer}|n{n}", chunks=CHUNKS_ALL)
                    continue
                for share in ("shared", "perfact"):
                    r = 1 if share == "shared" else n
                    run_one(m, layer, fs, r, None, f"multi|{m}|L{layer}|n{n}|{share}",
                            extra={"share": share}, chunks=CHUNKS_ALL)
        for qm in [m for m in methods if m.startswith("qkv")]:
            run_one(qm, None, fs, 16, None, f"multi|{qm}|all|n{n}", chunks=CHUNKS_ALL)
else:
    raise SystemExit("unknown phase")
print("PHASE DONE", args.phase, flush=True)
