"""T2 Part A steps 2-4: where do per-user LoRA facts live in Qwen3, and do they ride
pre-existing answer-writing neurons ("off-ramp")?

For every direct-recall fact the LoRA gets right and the base gets wrong:
  target position = first answer token where base's teacher-forced argmax != gold;
  metric m = logit[gold] - logit[base_pred] at that position.
  Recovery of a variant X = (m_X - m_base) / (m_lora - m_base).

Measures per fact:
  layer_dla        per-layer direct-logit attribution of attn/MLP outputs (base, lora)
  layer_axg        per-layer activation x gradient: MLP neurons (sum a*dm/da), attn output
  window           LoRA enabled only in layers >= l (suffix) / < l (prefix): recovery
  offramp          static neuron DLA |S| ranking (base W_down, u=W_U[gold]-W_U[pred]);
                   share of LoRA-induced neuron direct contribution c_i in top-K static
                   vs random-K; ablations (zero / base-patch) at target position
Per adapter (MLP variant): delta-W SVD alignment with existing neuron directions.
"""
import os, re, sys, json, math, random, argparse, glob
import torch
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEMFRAC", 28 / 96)))
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from peft.tuners.lora import LoraLayer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--root", default=os.path.expanduser("~/clean-tests/results/T2"))
ap.add_argument("--variants", default="qkv,mlp")
ap.add_argument("--uids", default="")
ap.add_argument("--out", default="analysis")
ap.add_argument("--max-facts", type=int, default=100)
args = ap.parse_args()
ROOT = Path(args.root)
UD = Path(os.path.expanduser("~/clean-tests/user-as-engram/data/users"))
OUT = ROOT / args.out
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda"
tok = AutoTokenizer.from_pretrained(args.model)
base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=dev)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)
NL = base.config.num_hidden_layers
NI = base.config.intermediate_size
gamma = base.model.norm.weight.float()
W_U = base.lm_head.weight  # (V, d)
Wdown_base = [base.model.layers[l].mlp.down_proj.weight.detach() for l in range(NL)]  # (d, NI)


def layers_of(model):
    out = {}
    for n, m in model.named_modules():
        mm = re.fullmatch(r"(.*\.)?layers\.(\d+)", n)
        if mm:
            out[int(mm.group(2))] = m
    return [out[i] for i in range(NL)]


class Tap:
    """Hooks: capture MLP inner activation a (input of down_proj), attn output, final
    residual. Optional per-layer neuron edits at one position; optional leaf-ification
    for activation x gradient."""

    def __init__(self, model):
        self.model = model
        self.L = layers_of(model)
        self.hs = []
        self.reset()

    def reset(self):
        self.pos = None
        self.edit = {}      # l -> (idx LongTensor, values Tensor or None=zero)
        self.grad = False
        self.a, self.o, self.final = {}, {}, None

    def __enter__(self):
        for l, layer in enumerate(self.L):
            self.hs.append(layer.mlp.down_proj.register_forward_pre_hook(self._mk_mlp(l)))
            self.hs.append(layer.self_attn.register_forward_hook(self._mk_attn(l)))
        norm = [m for n, m in self.model.named_modules() if n.endswith("model.norm")][0]
        self.hs.append(norm.register_forward_pre_hook(self._final))
        return self

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()
        self.hs = []

    def _mk_mlp(self, l):
        def f(mod, inp):
            x = inp[0]
            if self.pos is None:
                return None
            if l in self.edit:
                idx, val = self.edit[l]
                x = x.clone()
                x[0, self.pos, idx] = 0 if val is None else val.to(x.dtype)
            if self.grad:
                x = x.detach().requires_grad_(True)
                self.a[l] = x
            else:
                self.a[l] = x[0, self.pos].detach()
            return (x,) + tuple(inp[1:])
        return f

    def _mk_attn(self, l):
        def f(mod, inp, out):
            o = out[0]
            if self.pos is None:
                return None
            if self.grad:
                o = o.detach().requires_grad_(True)
                self.o[l] = o
                return (o,) + tuple(out[1:])
            self.o[l] = o[0, self.pos].detach()
            return None
        return f

    def _final(self, mod, inp):
        if self.pos is not None:
            self.final = inp[0][0, self.pos]
        return None


def lora_modules(model):
    res = []
    for n, m in model.named_modules():
        if isinstance(m, LoraLayer):
            mm = re.search(r"layers\.(\d+)\.", n)
            res.append((int(mm.group(1)), n, m))
    return res


def set_lora_layers(model, enabled_layers, orig):
    for l, n, m in lora_modules(model):
        for k in m.scaling:
            m.scaling[k] = orig[n][k] if l in enabled_layers else 0.0


def build(q, gold):
    prompt = f"Question: {q}\nAnswer:"
    p_ids = tok(prompt)["input_ids"]
    a_ids = tok(" " + str(gold), add_special_tokens=False)["input_ids"]
    return p_ids, a_ids


@torch.no_grad()
def logits_at(model, ids):
    return model(torch.tensor([ids], device=dev)).logits[0].float()


def metric(model, ids, pos, g, b, tap=None):
    lo = logits_at(model, ids)
    return (lo[pos, g] - lo[pos, b]).item(), lo[pos].argmax().item()


def static_S(u, Wd_list):
    gu = (gamma * u.float())
    return torch.stack([(Wd.float().t() @ gu) for Wd in Wd_list])  # (NL, NI)


def eff_wdown(model):
    """Effective down_proj weights (base + LoRA) per layer, if MLP-LoRA; else base."""
    mods = {l: m for l, n, m in lora_modules(model) if n.endswith("down_proj")}
    out = []
    for l in range(NL):
        W = Wdown_base[l]
        if l in mods:
            m = mods[l]
            k = list(m.scaling.keys())[0]
            A = m.lora_A[k].weight.float(); B = m.lora_B[k].weight.float()
            W = (W.float() + m.scaling[k] * (B @ A)).to(W.dtype)
        out.append(W)
    return out


def analyze_fact(lm, tap, ids, pos, g, b, Wd_eff, orig_scaling):
    R = {}
    full_layers = set(range(NL))
    # base / lora metrics + captures
    with lm.disable_adapter():
        tap.reset(); tap.pos = pos
        m_base, _ = metric(lm, ids, pos, g, b)
        a_B = torch.stack([tap.a[l] for l in range(NL)]).float()
        o_B = torch.stack([tap.o[l] for l in range(NL)]).float()
        fin_B = tap.final.float()
    tap.reset(); tap.pos = pos
    m_lora, am = metric(lm, ids, pos, g, b)
    a_L = torch.stack([tap.a[l] for l in range(NL)]).float()
    o_L = torch.stack([tap.o[l] for l in range(NL)]).float()
    fin_L = tap.final.float()
    R.update(m_base=m_base, m_lora=m_lora, lora_argmax_is_gold=(am == g))
    dm = m_lora - m_base
    rec = lambda m: (m - m_base) / dm if abs(dm) > 1e-6 else float("nan")
    u = (W_U[g].float() - W_U[b].float())
    gu = gamma * u
    rmsB = fin_B.pow(2).mean().sqrt() + 1e-6
    rmsL = fin_L.pow(2).mean().sqrt() + 1e-6
    # 2a. per-layer DLA of components
    S_B = static_S(u, Wdown_base)            # (NL, NI) per-unit-activation DLA (base weights)
    S_L = static_S(u, Wd_eff)
    mlp_dla_B = (a_B * S_B).sum(1) / rmsB
    mlp_dla_L = (a_L * S_L).sum(1) / rmsL
    attn_dla_B = (o_B @ gu) / rmsB
    attn_dla_L = (o_L @ gu) / rmsL
    R["layer_dla"] = dict(mlp_base=mlp_dla_B.tolist(), mlp_lora=mlp_dla_L.tolist(),
                          attn_base=attn_dla_B.tolist(), attn_lora=attn_dla_L.tolist())
    # 2b. activation x gradient (total effect, linearised)
    def axg(model_ctx):
        tap.reset(); tap.pos = pos; tap.grad = True
        with torch.enable_grad():
            lo = lm(torch.tensor([ids], device=dev)).logits[0, pos].float()
            mm = lo[g] - lo[b]
            ts = [tap.a[l] for l in range(NL)] + [tap.o[l] for l in range(NL)]
            grads = torch.autograd.grad(mm, ts, allow_unused=True)
        mlp = [(tap.a[l][0, pos].float() * grads[l][0, pos].float()).sum().item() for l in range(NL)]
        att = [(tap.o[l][0, pos].float() * grads[NL + l][0, pos].float()).sum().item() for l in range(NL)]
        # also whole-sequence attention attribution (attn writes at earlier positions matter)
        att_all = [(tap.o[l][0].float() * grads[NL + l][0].float()).sum().item() for l in range(NL)]
        mlp_all = [(tap.a[l][0].float() * grads[l][0].float()).sum().item() for l in range(NL)]
        tap.reset()
        return dict(mlp_pos=mlp, attn_pos=att, mlp_allpos=mlp_all, attn_allpos=att_all)
    with lm.disable_adapter():
        axB = axg(None)
    axL = axg(None)
    R["layer_axg"] = dict(base=axB, lora=axL)
    # 2c. windows: LoRA only in suffix [l, NL) / prefix [0, l)
    suf, pre = [], []
    for l in range(NL + 1):
        set_lora_layers(lm, set(range(l, NL)), orig_scaling)
        tap.reset(); suf.append(rec(metric(lm, ids, pos, g, b)[0]))
        set_lora_layers(lm, set(range(0, l)), orig_scaling)
        tap.reset(); pre.append(rec(metric(lm, ids, pos, g, b)[0]))
    set_lora_layers(lm, full_layers, orig_scaling)
    R["window"] = dict(suffix=suf, prefix=pre)
    # single-layer-band (4 layers) enabling
    band = []
    for l in range(0, NL, 4):
        set_lora_layers(lm, set(range(l, min(NL, l + 4))), orig_scaling)
        tap.reset(); band.append(rec(metric(lm, ids, pos, g, b)[0]))
    set_lora_layers(lm, full_layers, orig_scaling)
    R["band4"] = band
    # 3. off-ramp
    absS = S_B.abs().flatten()
    order = torch.argsort(absS, descending=True)
    c = ((a_L * S_L) / rmsL - (a_B * S_B) / rmsB).flatten()   # LoRA-induced direct contribution
    tot = c.sum().item()
    R["mlp_direct_delta"] = tot
    R["attn_direct_delta"] = ((attn_dla_L - attn_dla_B).sum()).item()
    R["dm"] = dm
    N = absS.numel()
    shares, rshares = {}, {}
    gen = torch.Generator(device="cpu").manual_seed(0)
    for K in (10, 100, 1000, 10000):
        shares[K] = c[order[:K]].sum().item() / tot if tot else float("nan")
        rs = [c[torch.randperm(N, generator=gen)[:K].to(c.device)].sum().item() / tot for _ in range(20)]
        rshares[K] = sum(rs) / len(rs)
    R["offramp_share_topK_static"] = shares
    R["offramp_share_randomK"] = rshares
    # concentration of c: how many neurons (by |c|) for 50% / 80% of positive total
    cs, ci = torch.sort(c, descending=True)
    cum = torch.cumsum(cs, 0)
    R["n_neurons_for_50pct"] = int((cum < 0.5 * tot).sum().item()) + 1 if tot > 0 else None
    R["n_neurons_for_80pct"] = int((cum < 0.8 * tot).sum().item()) + 1 if tot > 0 else None
    # static-rank percentile of the top-100 contributing neurons
    rank_of = torch.empty_like(order); rank_of[order] = torch.arange(N, device=order.device)
    top100 = ci[:100]
    pct = (rank_of[top100].float() / N)
    R["top100_contrib_static_pct_median"] = pct.median().item()
    R["top100_contrib_in_static_top1000"] = (rank_of[top100] < 1000).float().mean().item()
    R["top100_contrib_layers"] = (top100 // NI).tolist()
    R["static_top1000_layers_hist"] = torch.bincount(order[:1000] // NI, minlength=NL).tolist()
    # ablations at target position
    def run_edit(flat_idx, mode):
        ed = {}
        for l in range(NL):
            sel = flat_idx[(flat_idx // NI) == l] % NI
            if len(sel):
                ed[l] = (sel, None if mode == "zero" else a_B[l, sel])
        tap.reset(); tap.pos = pos; tap.edit = ed
        m, amx = metric(lm, ids, pos, g, b)
        tap.reset()
        return rec(m), amx == g
    abl = {}
    for K in (10, 100, 1000, 5000):
        st = order[:K]
        # layer-matched random
        rnd = []
        for l in range(NL):
            n_l = int(((st // NI) == l).sum().item())
            if n_l:
                rnd.append(l * NI + torch.randperm(NI, generator=gen)[:n_l].to(st.device))
        rnd = torch.cat(rnd)
        abl[K] = dict(zero_static=run_edit(st, "zero"), zero_random_layermatched=run_edit(rnd, "zero"),
                      basepatch_static=run_edit(st, "patch"), basepatch_topcontrib=run_edit(ci[:K], "patch"),
                      basepatch_random_layermatched=run_edit(rnd, "patch"))
    R["ablation"] = abl
    return R


def dW_alignment(lm, variant):
    """SVD of per-layer LoRA deltas; alignment with existing neuron directions."""
    out = {"fro": {}, "down": [], "gate_up": []}
    gen = torch.Generator(device="cpu").manual_seed(1)
    for l, n, m in lora_modules(lm):
        k = list(m.scaling.keys())[0]
        A = m.lora_A[k].weight.float(); B = m.lora_B[k].weight.float()
        dW = m.scaling[k] * (B @ A)
        name = n.split(".")[-1]
        out["fro"].setdefault(name, [0.0] * NL)[l] = dW.norm().item()
        if variant != "mlp":
            continue
        U, Sv, Vh = torch.linalg.svd(dW, full_matrices=False)
        r = 8
        if name == "down_proj":     # dW: (d, NI). Right vecs live in neuron space.
            Wn = Wdown_base[l].float(); Wn = Wn / Wn.norm(dim=0, keepdim=True)   # (d, NI) cols
            cos_out = (U[:, :r].t() @ Wn).abs().max(1).values                      # out-dir vs neuron out-dirs
            rv = torch.randn(Wn.shape[0], 64, generator=gen).to(Wn.device); rv /= rv.norm(dim=0)
            cos_rand = (rv.t() @ Wn).abs().max(1).values.mean().item()
            V = Vh[:r]                                                              # (r, NI)
            k1 = max(1, NI // 100)
            mass_top1 = (V.pow(2).sort(1, descending=True).values[:, :k1].sum(1))
            rvn = torch.randn(64, NI, generator=gen).to(V.device); rvn /= rvn.norm(dim=1, keepdim=True)
            mass_rand = rvn.pow(2).sort(1, descending=True).values[:, :k1].sum(1).mean().item()
            out["down"].append(dict(layer=l, sv=Sv[:r].tolist(), maxcos_out_vs_neurons=cos_out.tolist(),
                                    maxcos_random_vs_neurons=cos_rand,
                                    v_mass_top1pct=mass_top1.tolist(), v_mass_top1pct_random=mass_rand,
                                    top_neurons_v1=torch.topk(Vh[0].abs(), 20).indices.tolist()))
        else:                       # gate/up: dW (NI, d). Right vecs = input dirs in d.
            W = base.model.layers[l].mlp.__getattr__(name).weight.float()
            Wn = W / W.norm(dim=1, keepdim=True)                                    # rows = neuron read dirs
            cos_in = (Vh[:r] @ Wn.t()).abs().max(1).values
            rv = torch.randn(64, Wn.shape[1], generator=gen).to(Wn.device); rv /= rv.norm(dim=1, keepdim=True)
            cos_rand = (rv @ Wn.t()).abs().max(1).values.mean().item()
            k1 = max(1, NI // 100)
            Ul = U[:, :r].t()                                                       # (r, NI) which neurons get written
            mass_top1 = Ul.pow(2).sort(1, descending=True).values[:, :k1].sum(1)
            out["gate_up"].append(dict(layer=l, proj=name, sv=Sv[:r].tolist(),
                                       maxcos_in_vs_neuron_reads=cos_in.tolist(),
                                       maxcos_random=cos_rand, u_mass_top1pct=mass_top1.tolist()))
    return out


uids = args.uids.split(",") if args.uids else sorted(
    {p.split("/")[-1] for p in glob.glob(str(ROOT / "adapters/*/u*")) if os.path.exists(p + "/metrics.json")})
for variant in args.variants.split(","):
    for uid in uids:
        ad = ROOT / "adapters" / variant / uid
        of = OUT / f"{variant}_{uid}.json"
        if of.exists() or not (ad / "metrics.json").exists():
            continue
        met = json.load(open(ad / "metrics.json"))
        bev = json.load(open(ROOT / "base_eval" / f"{uid}.json"))
        lm = PeftModel.from_pretrained(base, str(ad)).eval()
        orig = {n: dict(m.scaling) for _, n, m in lora_modules(lm)}
        Wd_eff = eff_wdown(lm)
        res = {"uid": uid, "variant": variant, "direct_acc": met["direct_acc"],
               "indirect_acc": met["indirect_acc"], "base_direct": bev["direct_acc"],
               "base_indirect": bev["indirect_acc"], "facts": [],
               "indirect_details": met["indirect_details"],
               "base_indirect_details": bev["indirect_details"]}
        user = json.load(open(UD / f"{uid}.json"))
        dq = {q["key"]: q for q in user["direct_qa"]}
        with Tap(lm) as tap:
            for d, bd in zip(met["direct_details"], bev["direct_details"]):
                if not d["ok"] or bd["ok"]:
                    continue
                p_ids, a_ids = build(d["q"], d["gold"])
                ids = p_ids + a_ids
                # first answer token where base teacher-forced argmax != gold
                with lm.disable_adapter():
                    tap.reset(); loB = logits_at(lm, ids)
                tap.reset(); loL = logits_at(lm, ids)
                tgt = None
                for j, t in enumerate(a_ids):
                    pp = len(p_ids) + j - 1
                    bp = loB[pp].argmax().item()
                    if tok.decode([t]).strip() == "":
                        continue  # formatting token (e.g. leading space before digits)
                    if bp != t:
                        tgt = (pp, t, bp, j); break
                if tgt is None:
                    continue
                pp, g, b, j = tgt
                if loL[pp].argmax().item() != g:
                    continue
                r = analyze_fact(lm, tap, ids, pp, g, b, Wd_eff, orig)
                r.update(key=d["key"], gold=d["gold"], tok_gold=tok.decode([g]), tok_base=tok.decode([b]),
                         ans_tok_index=j, seq_len=len(ids))
                res["facts"].append(r)
                if len(res["facts"]) >= args.max_facts:
                    break
        res["dW"] = dW_alignment(lm, variant)
        json.dump(res, open(of, "w"))
        print(f"{variant} {uid}: {len(res['facts'])} facts analysed", flush=True)
        base = lm.unload()
        torch.cuda.empty_cache()
