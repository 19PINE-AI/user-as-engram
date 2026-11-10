"""T6: gated neuron vs rank-1 LoRA vs ROME vs Q/K/V LoRA for fact storage.

All edits are implemented as forward hooks with an `enabled` flag, so the base
model and the edited model share weights and can be compared token-by-token.
"""
import json, math, os, random, time
from contextlib import contextmanager

import torch
import torch.nn.functional as F

DEV = "cuda"

# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
CITIES = [
    ("Lyon", "France"), ("Marseille", "France"), ("Munich", "Germany"), ("Hamburg", "Germany"),
    ("Osaka", "Japan"), ("Kyoto", "Japan"), ("Milan", "Italy"), ("Naples", "Italy"),
    ("Barcelona", "Spain"), ("Seville", "Spain"), ("Porto", "Portugal"), ("Krakow", "Poland"),
    ("Gothenburg", "Sweden"), ("Bergen", "Norway"), ("Rotterdam", "Netherlands"),
    ("Antwerp", "Belgium"), ("Zurich", "Switzerland"), ("Salzburg", "Austria"),
    ("Edinburgh", "Scotland"), ("Manchester", "England"), ("Toronto", "Canada"),
    ("Vancouver", "Canada"), ("Chicago", "America"), ("Sydney", "Australia"),
    ("Melbourne", "Australia"), ("Auckland", "New Zealand"), ("Mumbai", "India"),
    ("Bangalore", "India"), ("Shanghai", "China"), ("Shenzhen", "China"), ("Busan", "Korea"),
    ("Hanoi", "Vietnam"), ("Bangkok", "Thailand"), ("Jakarta", "Indonesia"),
    ("Manila", "Philippines"), ("Istanbul", "Turkey"), ("Cairo", "Egypt"),
    ("Casablanca", "Morocco"), ("Lagos", "Nigeria"), ("Nairobi", "Kenya"),
    ("Johannesburg", "South Africa"), ("Rio de Janeiro", "Brazil"), ("Sao Paulo", "Brazil"),
    ("Buenos Aires", "Argentina"), ("Santiago", "Chile"), ("Lima", "Peru"),
    ("Bogota", "Colombia"), ("Guadalajara", "Mexico"), ("Havana", "Cuba"),
    ("Saint Petersburg", "Russia"), ("Kyiv", "Ukraine"), ("Athens", "Greece"),
    ("Budapest", "Hungary"), ("Prague", "Czech Republic"), ("Dublin", "Ireland"),
    ("Helsinki", "Finland"), ("Copenhagen", "Denmark"), ("Tehran", "Iran"),
    ("Karachi", "Pakistan"), ("Dhaka", "Bangladesh"),
]

SYL_A = ["Ta", "Ve", "Qu", "Or", "Ma", "Ze", "Lo", "Fi", "Ka", "Ny", "Bre", "Sol", "Dra", "Hu", "Ira",
         "Cy", "Pe", "Vo", "Ul", "Xa", "Mi", "Ro", "Ga", "Ele", "Thi"]
SYL_B = ["vrin", "landra", "rell", "thos", "mira", "quen", "dris", "noll", "vexa", "lune", "riss",
         "baro", "thim", "selle", "kory", "dane", "vio", "mar", "nesh", "pell"]
SUR_A = ["Ol", "Quis", "Va", "Bran", "Thor", "Kel", "Mor", "Dar", "Fen", "Hal", "Zan", "Pry", "Wex",
         "Lund", "Cor", "Ast", "Rhe", "Tam", "Yor", "Gul"]
SUR_B = ["quist", "dell", "rova", "worth", "venko", "stad", "mire", "holt", "brecht", "lowe", "ard",
         "ness", "ovic", "berg", "tash", "inger", "aux", "ley", "sor", "drin"]

DIRECT = "{name} lives in the city of"
PARA = ["{name}'s home city is", "The city where {name} lives is", "{name} resides in the city of"]
TWOHOP = ["{name} lives in the country of", "The country where {name} lives is"]
NEIGHBOR = ["{name} works as a", "{name} was born in the year", "{name}'s favorite food is",
            "{name} studied at the University of"]
PREFIXES = ["", "The weather was mild that week.", "According to the local newspaper,",
            "Here is a short note.", "In a recent interview,"]


def make_names(n, seed):
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        nm = rng.choice(SYL_A) + rng.choice(SYL_B) + " " + rng.choice(SUR_A) + rng.choice(SUR_B)
        if nm not in seen:
            seen.add(nm); out.append(nm)
    return out


def make_facts(cities, n_people=260, seed=0):
    names = make_names(n_people, seed)
    rng = random.Random(seed + 1)
    facts = []
    for i, nm in enumerate(names):
        c, k = rng.choice(cities)
        facts.append({"id": i, "name": nm, "city": c, "country": k})
    return facts


# ----------------------------------------------------------------------------
# model helpers
# ----------------------------------------------------------------------------
def load(model_name, budget_gb):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.set_per_process_memory_fraction(budget_gb / 96)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def layers(model):
    return model.model.layers


def enc_target(tok, prompt, target):
    """Token ids of prompt and of ' target' (continuation)."""
    p = tok(prompt, add_special_tokens=False).input_ids
    full = tok(prompt + " " + target, add_special_tokens=False).input_ids
    assert full[:len(p)] == p, (prompt, target)
    return p, full[len(p):]


def batch_pt(tok, pairs):
    """pairs: list of (prompt_ids, target_ids) -> padded input, labels mask (right pad)."""
    L = max(len(p) + len(t) for p, t in pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.full((len(pairs), L), pad, dtype=torch.long)
    att = torch.zeros((len(pairs), L), dtype=torch.long)
    lab = torch.full((len(pairs), L), -100, dtype=torch.long)
    for i, (p, t) in enumerate(pairs):
        s = p + t
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
        lab[i, len(p) - 1:len(s) - 1] = torch.tensor(t)  # position predicting each target token
    return ids.to(DEV), att.to(DEV), lab.to(DEV)


def target_nll(model, ids, att, lab):
    logits = model(input_ids=ids, attention_mask=att).logits  # bf16 [B,T,V]
    m = lab != -100
    b, t = m.nonzero(as_tuple=True)
    sel = logits[b, t].float()                      # only labelled positions
    y = lab[b, t]
    lp = torch.log_softmax(sel, -1).gather(-1, y[:, None]).squeeze(-1)
    hit = sel.argmax(-1) == y
    B = ids.shape[0]
    per_seq = -torch.zeros(B, device=ids.device).index_add(0, b, lp) / m.sum(1)
    miss = torch.zeros(B, device=ids.device).index_add(0, b, (~hit).float())
    return per_seq, miss == 0, None


# ----------------------------------------------------------------------------
# edits
# ----------------------------------------------------------------------------
class Edit:
    """Hook-based additive edit. kinds:
    lora_down: down_proj out += (inp @ A) @ B          A:[d_ff,r] B:[r,d]
    lora_up:   up_proj  out += (x @ A) @ B             A:[d,r]    B:[r,d_ff]
    neuron:    mlp out += (silu(x@Wg [+bg]) * (x@Wu)) @ Wd
    rome:      down_proj out += (inp @ P) @ R          (closed form, not trained)
    qkv:       q/k/v_proj out += (x @ A) @ B * scale at every layer (rank 16)
    """

    def __init__(self, model, kind, layer=None, r=1, bias=False, seed=0):
        self.model, self.kind, self.layer, self.r = model, kind, layer, r
        self.enabled = True
        self.hooks, self.params = [], []
        self.record = None  # for firing stats: list of [tokens, r] tensors
        g = torch.Generator(device="cpu").manual_seed(seed)
        cfg = model.config
        d, dff = cfg.hidden_size, cfg.intermediate_size

        def P(*shape, std=0.0):
            t = (torch.randn(*shape, generator=g) * std) if std > 0 else torch.zeros(*shape)
            t = t.to(DEV).float().requires_grad_(True)
            self.params.append(t)
            return t

        if kind == "lora_down":
            self.A, self.B = P(dff, r, std=1 / math.sqrt(dff)), P(r, d)
            self._hook(layers(model)[layer].mlp.down_proj, self._f_lowrank)
        elif kind == "lora_up":
            self.A, self.B = P(d, r, std=1 / math.sqrt(d)), P(r, dff)
            self._hook(layers(model)[layer].mlp.up_proj, self._f_lowrank)
        elif kind == "neuron":
            self.Wg, self.Wu = P(d, r, std=1 / math.sqrt(d)), P(d, r, std=1 / math.sqrt(d))
            self.Wd = P(r, d)
            self.bg = P(r) if bias else None
            self._hook(layers(model)[layer].mlp, self._f_neuron)
        elif kind == "qkv":
            self.lora = []
            for L in layers(model):
                for name in ("q_proj", "k_proj", "v_proj"):
                    mod = getattr(L.self_attn, name)
                    A = P(mod.in_features, r, std=1 / math.sqrt(mod.in_features))
                    B = P(r, mod.out_features)
                    self.lora.append((A, B))
                    self._hook(mod, self._mk_qkv(A, B))
        elif kind == "rome":
            self.Pm = self.R = None  # set by rome_fit
            self._hook(layers(model)[layer].mlp.down_proj, self._f_rome)
        else:
            raise ValueError(kind)

    def _hook(self, mod, fn):
        self.hooks.append(mod.register_forward_hook(fn))

    def _rec(self, z):
        if self.record is not None:
            self.record.append(z.detach().reshape(-1, z.shape[-1]).float().cpu())

    def _f_lowrank(self, mod, inp, out):
        if not self.enabled:
            return out
        x = inp[0].float()
        z = x @ self.A
        self._rec(z)
        return out + (z @ self.B).to(out.dtype)

    def _f_neuron(self, mod, inp, out):
        if not self.enabled:
            return out
        x = inp[0].float()
        gpre = x @ self.Wg
        if self.bg is not None:
            gpre = gpre + self.bg
        z = F.silu(gpre) * (x @ self.Wu)
        self._rec(z)
        return out + (z @ self.Wd).to(out.dtype)

    def _f_rome(self, mod, inp, out):
        if not self.enabled or self.Pm is None:
            return out
        x = inp[0].float()
        z = x @ self.Pm
        self._rec(z)
        return out + (z @ self.R).to(out.dtype)

    def _mk_qkv(self, A, B):
        def f(mod, inp, out):
            if not self.enabled:
                return out
            return out + ((inp[0].float() @ A) @ B * 2.0).to(out.dtype)
        return f

    def remove(self):
        for h in self.hooks:
            h.remove()

    @contextmanager
    def off(self):
        old = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = old

    def n_params(self):
        if self.kind == "rome":
            return int(self.Pm.numel() + self.R.numel()) if self.Pm is not None else 0
        return int(sum(p.numel() for p in self.params))


# ----------------------------------------------------------------------------
# training (shared by all trainable kinds)
# ----------------------------------------------------------------------------
def train_pairs(tok, facts):
    pairs = []
    for f in facts:
        for pre in PREFIXES:
            pr = (pre + " " if pre else "") + DIRECT.format(name=f["name"])
            pairs.append(enc_target(tok, pr, f["city"]))
    return pairs


def make_anchors(tok, neg_names, texts, n_names=16, n_text=2, text_len=128, seed=0):
    """Anchor inputs for the locality regulariser: other-subject direct prompts + generic text.
    Disjoint from the evaluation sets."""
    rng = random.Random(seed)
    names = rng.sample(neg_names, min(n_names, len(neg_names)))
    prompts = [DIRECT.format(name=n) for n in names] + [p.format(name=n) for n in names[:4] for p in NEIGHBOR[:2]]
    ids = tok("\n\n".join(texts), add_special_tokens=False).input_ids
    starts = rng.sample(range(0, max(1, len(ids) - text_len)), n_text)
    chunks = [torch.tensor(ids[s:s + text_len]) for s in starts]
    return {"prompts": prompts, "chunks": chunks}


def _anchor_logp(model, tok, anchors):
    lp_p = next_dist(model, tok, anchors["prompts"]) if anchors["prompts"] else None
    x = torch.stack(anchors["chunks"]).to(DEV)
    lp_t = torch.log_softmax(model(input_ids=x).logits.float(), -1)
    return lp_p, lp_t


def next_dist_grad(model, tok, prompts):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    L_ = max(map(len, enc))
    ids = torch.full((len(enc), L_), pad, dtype=torch.long)
    att = torch.zeros((len(enc), L_), dtype=torch.long)
    for j, e in enumerate(enc):
        ids[j, L_ - len(e):] = torch.tensor(e)
        att[j, L_ - len(e):] = 1
    lg = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits[:, -1].float()
    return torch.log_softmax(lg, -1)


def train_edit(model, tok, edit, facts, lr, max_steps=100, stop_nll=0.05, mb=40, anchors=None, kl_w=1.0):
    """Plain SFT on the fact prompts (anchors=None, early stop at stop_nll), or SFT + KL(base||edit)
    on anchors (fixed max_steps, so the regulariser gets the full budget)."""
    pairs = train_pairs(tok, facts)
    opt = torch.optim.Adam(edit.params, lr=lr)
    hist, klh = [], []
    t0 = time.time()
    if anchors is not None:
        with edit.off(), torch.no_grad():
            ref_p, ref_t = _anchor_logp(model, tok, anchors)
    for step in range(max_steps):
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for i in range(0, len(pairs), mb):
            ids, att, lab = batch_pt(tok, pairs[i:i + mb])
            nll, _, _ = target_nll(model, ids, att, lab)
            loss = nll.sum() / len(pairs)
            loss.backward()
            tot += loss.item()
        hist.append(tot)
        if anchors is not None:
            lp_p = next_dist_grad(model, tok, anchors["prompts"])
            x = torch.stack(anchors["chunks"]).to(DEV)
            lp_t = torch.log_softmax(model(input_ids=x).logits.float(), -1)
            k = kl(ref_p, lp_p).mean() + kl(ref_t, lp_t).mean()
            (kl_w * k).backward()
            klh.append(k.item())
        elif tot < stop_nll:
            break
        opt.step()
    out = {"steps": len(hist), "final_nll": hist[-1], "train_sec": time.time() - t0}
    if klh:
        out["final_anchor_kl"] = klh[-1]
    return out


# ----------------------------------------------------------------------------
# ROME / single-layer MEMIT
# ----------------------------------------------------------------------------
def collect_cov(model, tok, texts, layer_ids, n_tokens=100_000, seqlen=512):
    """Uncentered second moment E[k k^T] of down_proj inputs at the given layers."""
    dff = model.config.intermediate_size
    acc = {l: torch.zeros(dff, dff, device=DEV) for l in layer_ids}
    cnt = [0]
    hs = []

    def mk(l):
        def f(mod, inp, out):
            k = inp[0].reshape(-1, dff).float()
            acc[l].addmm_(k.T, k)
        return f

    for l in layer_ids:
        hs.append(layers(model)[l].mlp.down_proj.register_forward_hook(mk(l)))
    ids = tok("\n\n".join(texts), add_special_tokens=False).input_ids
    with torch.no_grad():
        for i in range(0, min(len(ids), n_tokens), seqlen):
            x = torch.tensor(ids[i:i + seqlen], device=DEV)[None]
            model(input_ids=x)
            cnt[0] += x.shape[1]
    for h in hs:
        h.remove()
    return {l: (acc[l] / cnt[0]).cpu() for l in layer_ids}, cnt[0]


def subject_last_pos(tok, prompt, name):
    """Index of the last token of `name` inside the tokenized prompt."""
    i = prompt.index(name) + len(name)
    pre = tok(prompt[:i], add_special_tokens=False).input_ids
    return len(pre) - 1


def rome_fit(model, tok, edit, facts, C, lam=15000.0, v_steps=25, v_lr=0.5, clamp=4.0, kl_w=0.0625):
    """Fit ROME (n=1, exact rank-1) or single-layer MEMIT (n>1) at edit.layer."""
    L = edit.layer
    mlp = layers(model)[L].mlp
    dff, d = model.config.intermediate_size, model.config.hidden_size
    Ks, Vres = [], []
    for f in facts:
        prompts = [(pre + " " if pre else "") + DIRECT.format(name=f["name"]) for pre in PREFIXES]
        poss = [subject_last_pos(tok, p, f["name"]) for p in prompts]
        pairs = [enc_target(tok, p, f["city"]) for p in prompts]
        ids, att, lab = batch_pt(tok, pairs)
        # key k*: mean down_proj input at subject last token
        cap = {}
        h = mlp.down_proj.register_forward_hook(lambda m, i, o: cap.__setitem__("k", i[0].detach().float()))
        h2 = mlp.register_forward_hook(lambda m, i, o: cap.__setitem__("v", o.detach().float()))
        with edit.off(), torch.no_grad():
            model(input_ids=ids, attention_mask=att)
        h.remove(); h2.remove()
        rows = torch.arange(len(prompts), device=DEV)
        pos = torch.tensor(poss, device=DEV)
        k = cap["k"][rows, pos].mean(0)
        v0 = cap["v"][rows, pos].mean(0)
        # essence prompt for KL regularisation
        ess = tok(f"{f['name']} is a", add_special_tokens=False).input_ids
        ess_ids = torch.tensor(ess, device=DEV)[None]
        ess_pos = subject_last_pos(tok, f"{f['name']} is a", f["name"])
        with edit.off(), torch.no_grad():
            ess_ref = torch.log_softmax(model(input_ids=ess_ids).logits[0, -1].float(), -1)
        delta = torch.zeros(d, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=v_lr)
        cur = {"pos": pos, "rows": rows}

        def add_delta(m, i, o):
            o = o.clone()
            if o.shape[0] == len(prompts):
                o[cur["rows"], cur["pos"]] += delta.to(o.dtype)
            else:
                o[0, ess_pos] += delta.to(o.dtype)
            return o

        hh = mlp.register_forward_hook(add_delta)
        with edit.off():
            for _ in range(v_steps):
                opt.zero_grad()
                nll, ok, _ = target_nll(model, ids, att, lab)
                lp = torch.log_softmax(model(input_ids=ess_ids).logits[0, -1].float(), -1)
                kl = F.kl_div(lp, ess_ref, log_target=True, reduction="sum")
                loss = nll.mean() + kl_w * kl + 1e-3 * delta.norm() ** 2 / (v0.norm() ** 2)
                if nll.mean().item() < 0.05:
                    break
                loss.backward()
                opt.step()
                with torch.no_grad():
                    mx = clamp * v0.norm()
                    if delta.norm() > mx:
                        delta.mul_(mx / delta.norm())
        hh.remove()
        Ks.append(k)
        Vres.append(delta.detach())  # residual needed at this layer: v* - W k* = delta
    K = torch.stack(Ks, 1)          # [dff, n]
    R = torch.stack(Vres, 0)        # [n, d]
    Cg = C.to(DEV)
    if len(facts) == 1:
        # ROME: dW = r (C^-1 k)^T / (k^T C^-1 k)  ->  out += (x . c) r
        Ck = torch.linalg.solve(Cg + 1e-6 * torch.eye(dff, device=DEV) * Cg.diagonal().mean(), K)
        Pm = Ck / (K.T @ Ck)
    else:
        # single-layer MEMIT: dW = R^T K^T (lam C + K K^T)^-1  ->  out += (x @ P) R, P = (lam C + K K^T)^-1 K
        Pm = torch.linalg.solve(lam * Cg + K @ K.T, K)
    edit.Pm, edit.R = Pm, R
    del Cg
    torch.cuda.empty_cache()
    return {"delta_norm": float(R.norm(dim=1).mean()), "n": len(facts)}


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def acc_on(model, tok, prompts_targets, mb=64):
    if not prompts_targets:
        return 0.0, []
    pairs = [enc_target(tok, p, t) for p, t in prompts_targets]
    oks = []
    for i in range(0, len(pairs), mb):
        ids, att, lab = batch_pt(tok, pairs[i:i + mb])
        _, ok, _ = target_nll(model, ids, att, lab)
        oks += ok.tolist()
    return sum(oks) / len(oks), oks


@torch.no_grad()
def next_dist(model, tok, prompts, mb=64):
    out = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(prompts), mb):
        enc = [tok(p, add_special_tokens=False).input_ids for p in prompts[i:i + mb]]
        L = max(map(len, enc))
        ids = torch.full((len(enc), L), pad, dtype=torch.long)
        att = torch.zeros((len(enc), L), dtype=torch.long)
        for j, e in enumerate(enc):  # left pad
            ids[j, L - len(e):] = torch.tensor(e)
            att[j, L - len(e):] = 1
        lg = model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits[:, -1].float()
        out.append(torch.log_softmax(lg, -1))
    return torch.cat(out)


def kl(lp_ref, lp_new):
    return (lp_ref.exp() * (lp_ref - lp_new)).sum(-1)


@torch.no_grad()
def text_locality(model, edit, chunks, layer_for_resid=None):
    """Mean per-token KL(base||edit), delta loss, and residual change at edit layer."""
    kls, dl, n = 0.0, 0.0, 0
    rs = {"sum": 0.0, "n": 0, "frac_gt_1pct": 0.0}
    cap = {}
    h = None
    if layer_for_resid is not None:
        h = layers(model)[layer_for_resid].register_forward_hook(
            lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().float()))
    for x in chunks:
        x = x[None].to(DEV)
        with edit.off():
            lb = model(input_ids=x).logits[0, :-1]
            hb = cap.get("h")
        le = model(input_ids=x).logits[0, :-1]
        he = cap.get("h")
        y = x[0, 1:]
        for s in range(0, y.numel(), 128):  # blockwise to bound fp32 memory
            lpb = torch.log_softmax(lb[s:s + 128].float(), -1)
            lpe = torch.log_softmax(le[s:s + 128].float(), -1)
            yy = y[s:s + 128, None]
            kls += kl(lpb, lpe).sum().item()
            dl += (-(lpe.gather(-1, yy)) + lpb.gather(-1, yy)).sum().item()
        del lb, le
        n += y.numel()
        if hb is not None:
            dn = (he[0] - hb[0]).norm(dim=-1)
            rel = dn / hb[0].norm(dim=-1).clamp(min=1e-6)
            rs["sum"] += dn.sum().item(); rs["n"] += dn.numel()
            rs["frac_gt_1pct"] += (rel > 0.01).sum().item()
    if h is not None:
        h.remove()
    out = {"kl_per_tok": kls / n, "dloss_per_tok": dl / n, "tokens": n}
    if rs["n"]:
        out["resid_dnorm_mean"] = rs["sum"] / rs["n"]
        out["resid_frac_rel_gt_1pct"] = rs["frac_gt_1pct"] / rs["n"]
    return out


@torch.no_grad()
def firing(model, tok, edit, facts, chunks):
    """Unit activation |z| on unrelated text vs at the trigger (direct prompts)."""
    if edit.kind not in ("lora_down", "lora_up", "neuron", "rome"):
        return None
    edit.record = []
    prompts = [DIRECT.format(name=f["name"]) for f in facts]
    for p in prompts:
        model(input_ids=torch.tensor(tok(p, add_special_tokens=False).input_ids, device=DEV)[None])
    trig = torch.stack([z.abs().max(0).values for z in edit.record])  # [n_prompts, r]
    trig_max = trig.max(0).values  # per unit
    edit.record = []
    for x in chunks:
        model(input_ids=x[None].to(DEV))
    Z = torch.cat(edit.record).abs()  # [tokens, r]
    edit.record = None
    ratio = Z / trig_max.clamp(min=1e-9)
    return {"frac_tokens_gt10pct_trigger": float((ratio > 0.1).float().mean()),
            "frac_tokens_gt50pct_trigger": float((ratio > 0.5).float().mean()),
            "mean_ratio": float(ratio.mean()), "p99_ratio": float(ratio.flatten().quantile(0.99)) if ratio.numel() < 16_000_000 else None}


def evaluate(model, tok, edit, facts, others, chunks, base_cache):
    res = {}
    d = [(DIRECT.format(name=f["name"]), f["city"]) for f in facts]
    p = [(t.format(name=f["name"]), f["city"]) for f in facts for t in PARA]
    h = [(t.format(name=f["name"]), f["country"]) for f in facts for t in TWOHOP]
    res["direct"], _ = acc_on(model, tok, d)
    res["para"], _ = acc_on(model, tok, p)
    res["twohop"], _ = acc_on(model, tok, h)
    # neighbor prompts (same subject, other relations): KL vs base
    nb = [t.format(name=f["name"]) for f in facts for t in NEIGHBOR]
    with edit.off():
        lb = next_dist(model, tok, nb)
    le = next_dist(model, tok, nb)
    res["neighbor_kl"] = float(kl(lb, le).mean())
    # other subjects with the same relation: bleed of the edited city
    op = [DIRECT.format(name=o["name"]) for o in others]
    with edit.off():
        lb = next_dist(model, tok, op)
    le = next_dist(model, tok, op)
    res["other_subject_kl"] = float(kl(lb, le).mean())
    first_ids = list({tok(" " + f["city"], add_special_tokens=False).input_ids[0] for f in facts})
    ft = torch.tensor(first_ids, device=DEV)
    res["other_subject_target_p_base"] = float(lb.exp()[:, ft].sum(-1).mean())
    res["other_subject_target_p_edit"] = float(le.exp()[:, ft].sum(-1).mean())
    res["text"] = text_locality(model, edit, chunks, layer_for_resid=edit.layer)
    res["firing"] = firing(model, tok, edit, facts, chunks[:4])
    return res
