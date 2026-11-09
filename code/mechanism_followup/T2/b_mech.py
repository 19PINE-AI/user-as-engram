"""T2 Part B: Engram (Joint OPT) vs per-user LoRA on Mini-Engram-d20, same facts.
Where in depth does each change the residual, and do both route recall through the same
downstream MLP neurons?

Per user: base captures -> Engram J-OPT write (repo recipe) -> captures -> undo ->
LoRA (repo condition B: rank 64, 1500 steps, lr 5e-4, c_q/c_k/c_v) -> captures + patches ->
undo -> re-apply Engram rows -> cross patches.
metric m = logit[gold] - logit[base_top] at the trigger (last) position.
"""
import os, sys, json, argparse, random
import torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 8 / 96)))
CODE = os.path.expanduser("~/clean-tests/user-as-engram/code")
sys.path[:0] = [CODE, CODE + "/nanochat_harness"]
from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import load_model
from scripts.head_to_head_locality import user_to_facts, train_lora_on_user, undo_lora, train_engram_joint_opt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt-dir", default=os.path.expanduser("~/clean-tests/nanochat_base/engram_runs/engram_d20_w1536_optimal"))
ap.add_argument("--uids", default=",".join(f"u{i:03d}" for i in range(8)))
ap.add_argument("--out", default=os.path.expanduser("~/clean-tests/results/T2/partB_mech.json"))
ap.add_argument("--steps", type=int, default=1500)
args = ap.parse_args()
dev = "cuda"
tok = get_tokenizer()
model, config = load_model(args.ckpt_dir, tok, dev)
NL = config.n_layer
blocks = model.transformer.h
bos = tok.get_bos_token_id()
UNREL = ["The capital of France is", "In 1905 Einstein published a paper on",
         "The recipe calls for two cups of flour and"]
tbl = model.engram.tables[str(max(config.engram_layer_ids))]


class Tap:
    def __init__(self):
        self.hs = []; self.reset()

    def reset(self):
        self.patch = {}; self.grad = False; self.resid = []; self.a = {}

    def __enter__(self):
        for l, b in enumerate(blocks):
            self.hs.append(b.register_forward_pre_hook(self._res()))
            self.hs.append(b.mlp.c_proj.register_forward_pre_hook(self._mlp(l)))
        return self

    def __exit__(self, *a):
        for h in self.hs: h.remove()

    def _res(self):
        def f(m, inp):
            self.resid.append(inp[0][0].detach().float())
        return f

    def _mlp(self, l):
        def f(m, inp):
            x = inp[0]
            if l in self.patch:
                idx, val = self.patch[l]
                x = x.clone(); x[0, -1, idx] = val.to(x.dtype)
            if self.grad:
                x = x.detach().requires_grad_(True)
                self.a[l] = x
            else:
                self.a[l] = x[0, -1].detach().float()
            return (x,) + tuple(inp[1:])
        return f


tap = Tap().__enter__()


def run(ids, g, b, grad=False, patch=None):
    tap.reset(); tap.grad = grad; tap.patch = patch or {}
    x = torch.tensor([ids], device=dev)
    with torch.set_grad_enabled(grad):
        lo = model(x)[0, -1].float()
        m = lo[g] - lo[b]
        out = dict(m=m.item(), top1=(lo.argmax().item() == g), resid=[r.clone() for r in tap.resid])
        if grad:
            gr = torch.autograd.grad(m, [tap.a[l] for l in range(NL)])
            out["a"] = torch.stack([tap.a[l][0, -1].detach().float() for l in range(NL)])
            out["g"] = torch.stack([gg[0, -1].float() for gg in gr])
        else:
            out["a"] = torch.stack([tap.a[l] for l in range(NL)])
    tap.reset()
    return out


def unrel_resid():
    return [run(tok.encode(s, prepend=bos), 0, 1)["resid"] for s in UNREL]


def resid_change(r1, r0):
    """per layer: (trigger-pos rel change, mean rel change at other positions)"""
    trig, other = [], []
    for a, b in zip(r1, r0):
        d = (a - b).norm(dim=-1) / (b.norm(dim=-1) + 1e-6)
        trig.append(d[-1].item()); other.append(d[:-1].mean().item() if d.numel() > 1 else 0.0)
    return trig, other


def topk_flat(c, k):
    return torch.topk(c.flatten(), k).indices


def patch_from(flat, a_base):
    NI = a_base.shape[1]; p = {}
    for l in range(NL):
        sel = flat[(flat // NI) == l] % NI
        if len(sel): p[l] = (sel, a_base[l, sel])
    return p


results = {"ckpt": args.ckpt_dir, "users": []}
if os.path.exists(args.out):  # resume: keep finished users
    results = json.load(open(args.out))
done = {u["uid"] for u in results["users"]}
gen = torch.Generator().manual_seed(0)
for uid in args.uids.split(","):
    if uid in done:
        continue
    user = json.load(open(os.path.expanduser(f"~/clean-tests/user-as-engram/data/users/{uid}.json")))
    facts = user_to_facts(user, tok)
    F = []
    for f in facts:
        ids = tok.encode(f["prompt"], prepend=bos)
        lo = model(torch.tensor([ids], device=dev))[0, -1].float()
        bt = lo.argmax().item()
        if bt == f["gold_id"]:
            continue
        F.append(dict(key=f["key"], ids=ids, g=f["gold_id"], b=bt))
    base = {i: run(fd["ids"], fd["g"], fd["b"]) for i, fd in enumerate(F)}
    ur0 = unrel_resid()
    W0 = tbl.embedding.weight.data.clone()
    # --- Engram J-OPT
    torch.manual_seed(0)
    restore, te, nrows = train_engram_joint_opt(model, config, tok, facts, dev, steps=args.steps, lr=0.5)
    W_E = tbl.embedding.weight.data.clone()
    rows = (W_E != W0).any(1).nonzero().flatten()
    E = {i: run(fd["ids"], fd["g"], fd["b"], grad=True) for i, fd in enumerate(F)}
    urE = unrel_resid()
    restore()
    # --- LoRA
    torch.manual_seed(0)
    handles, gs, tl = train_lora_on_user(model, tok, facts, dev, rank=64, alpha=128, steps=args.steps, lr=5e-4)
    L = {i: run(fd["ids"], fd["g"], fd["b"], grad=True) for i, fd in enumerate(F)}
    urL = unrel_resid()
    rec = {}
    for i, fd in enumerate(F):
        cE = (E[i]["a"] - base[i]["a"]) * E[i]["g"]
        cL = (L[i]["a"] - base[i]["a"]) * L[i]["g"]
        E[i]["c"], L[i]["c"] = cE, cL
        r = {}
        for k in (10, 100, 1000):
            own = run(fd["ids"], fd["g"], fd["b"], patch=patch_from(topk_flat(cL, k), base[i]["a"]))["m"]
            cross = run(fd["ids"], fd["g"], fd["b"], patch=patch_from(topk_flat(cE, k), base[i]["a"]))["m"]
            rnd = torch.randperm(cL.numel(), generator=gen)[:k].to(cL.device)
            rm = run(fd["ids"], fd["g"], fd["b"], patch=patch_from(rnd, base[i]["a"]))["m"]
            r[k] = dict(own=own, cross=cross, random=rm)
        rec[i] = {"lora": r}
    undo_lora(handles, gs, model)
    # --- re-apply Engram rows (table values only; equivalent to the trained state)
    with torch.no_grad():
        tbl.embedding.weight.data[rows] = W_E[rows]
    chk = [run(fd["ids"], fd["g"], fd["b"])["m"] for fd in F]
    for i, fd in enumerate(F):
        r = {}
        for k in (10, 100, 1000):
            own = run(fd["ids"], fd["g"], fd["b"], patch=patch_from(topk_flat(E[i]["c"], k), base[i]["a"]))["m"]
            cross = run(fd["ids"], fd["g"], fd["b"], patch=patch_from(topk_flat(L[i]["c"], k), base[i]["a"]))["m"]
            rnd = torch.randperm(E[i]["c"].numel(), generator=gen)[:k].to(E[i]["c"].device)
            rm = run(fd["ids"], fd["g"], fd["b"], patch=patch_from(rnd, base[i]["a"]))["m"]
            r[k] = dict(own=own, cross=cross, random=rm)
        rec[i]["engram"] = r
    with torch.no_grad():
        tbl.embedding.weight.data.copy_(W0)
    # --- summarise per fact
    fr = []
    for i, fd in enumerate(F):
        mB = base[i]["m"]
        def rc(m, X): return (m - mB) / (X[i]["m"] - mB) if abs(X[i]["m"] - mB) > 1e-6 else float("nan")
        cE, cL = E[i]["c"], L[i]["c"]
        tE, tL = set(topk_flat(cE, 100).tolist()), set(topk_flat(cL, 100).tolist())
        dE = (E[i]["a"] - base[i]["a"]); dL = (L[i]["a"] - base[i]["a"])
        cos_layer = [torch.nn.functional.cosine_similarity(dE[l], dL[l], dim=0).item() for l in range(NL)]
        # shuffled baseline: Engram delta of fact i vs LoRA delta of another fact j
        j = (i + 1) % len(F)
        dLj = (L[j]["a"] - base[j]["a"])
        cos_shuf = [torch.nn.functional.cosine_similarity(dE[l], dLj[l], dim=0).item() for l in range(NL)]
        trigE, othE = resid_change(E[i]["resid"], base[i]["resid"])
        trigL, othL = resid_change(L[i]["resid"], base[i]["resid"])
        fr.append(dict(
            key=fd["key"], m_base=mB, m_engram=E[i]["m"], m_lora=L[i]["m"],
            top1_engram=E[i]["top1"], top1_lora=L[i]["top1"], m_engram_reapplied=chk[i],
            resid_trig_engram=trigE, resid_other_engram=othE, resid_trig_lora=trigL, resid_other_lora=othL,
            mlp_attr_layer_engram=cE.sum(1).tolist(), mlp_attr_layer_lora=cL.sum(1).tolist(),
            mlp_attr_total_engram=cE.sum().item(), mlp_attr_total_lora=cL.sum().item(),
            dm_engram=E[i]["m"] - mB, dm_lora=L[i]["m"] - mB,
            top100_jaccard=len(tE & tL) / len(tE | tL),
            dact_cos_layer=cos_layer, dact_cos_layer_shuffled=cos_shuf,
            patch={X: {k: {t: rc(v, E if X == "engram" else L) for t, v in d.items()}
                       for k, d in rec[i][X].items()} for X in ("engram", "lora")},
        ))
    ul = lambda ur: [[sum(resid_change(a, b)[0][l] for _ in [0]) for l in range(NL)] for a, b in zip(ur, ur0)]
    unrel = {"engram": [[(x - y).norm(dim=-1).div(y.norm(dim=-1) + 1e-6).mean().item() for x, y in zip(a, b)]
                        for a, b in zip(urE, ur0)],
             "lora": [[(x - y).norm(dim=-1).div(y.norm(dim=-1) + 1e-6).mean().item() for x, y in zip(a, b)]
                      for a, b in zip(urL, ur0)]}
    results["users"].append(dict(uid=uid, n_facts=len(F), engram_rows=int(rows.numel()),
                                 engram_train_s=te, lora_train_s=tl, facts=fr, unrelated_rel_change=unrel))
    print(uid, len(F), "facts; engram top1", sum(f["top1_engram"] for f in fr), "lora top1",
          sum(f["top1_lora"] for f in fr), flush=True)
    json.dump(results, open(args.out, "w"))
