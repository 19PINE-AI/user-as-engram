"""Batched T6 edits: E independent edits share one forward pass; each row is routed to its own edit.

Adam is element-wise, so training E stacked edits on a summed loss is equivalent to E independent runs
(each slice only receives gradient from its own rows). Finished edits (early stop) are frozen by
restoring their slices after each optimizer step.
"""
import math, random, time, zlib
from contextlib import contextmanager

import torch
import torch.nn.functional as F

import t6lib as L
from t6lib import DEV, DIRECT, PARA, TWOHOP, NEIGHBOR, PREFIXES, enc_target, layers, kl


def _rand(shape, std, seeds):
    out = []
    for s in seeds:
        g = torch.Generator(device="cpu").manual_seed(s)
        out.append(torch.randn(*shape, generator=g) * std)
    return torch.stack(out)


class BEdit:
    """E stacked edits of one kind. Route rows with .route(eids) (−1 = base / no edit)."""

    def __init__(self, model, kind, E, layer=None, r=1, bias=False, seeds=None):
        self.model, self.kind, self.E, self.layer, self.r = model, kind, E, layer, r
        self.eid = None
        self.hooks, self.params = [], []
        self.record = None
        seeds = seeds or list(range(E))
        cfg = model.config
        d, dff = cfg.hidden_size, cfg.intermediate_size

        def P(t):
            t = t.to(DEV).float().requires_grad_(True)
            self.params.append(t)
            return t

        if kind == "lora_down":
            self.A, self.B = P(_rand((dff, r), 1 / math.sqrt(dff), seeds)), P(torch.zeros(E, r, d))
            self._hook(layers(model)[layer].mlp.down_proj, self._f_lowrank_AB)
        elif kind == "lora_up":
            self.A, self.B = P(_rand((d, r), 1 / math.sqrt(d), seeds)), P(torch.zeros(E, r, dff))
            self._hook(layers(model)[layer].mlp.up_proj, self._f_lowrank_AB)
        elif kind == "neuron":
            self.Wg = P(_rand((d, r), 1 / math.sqrt(d), seeds))
            self.Wu = P(_rand((d, r), 1 / math.sqrt(d), [s + 7919 for s in seeds]))
            self.Wd = P(torch.zeros(E, r, d))
            self.bg = P(torch.zeros(E, r)) if bias else None
            self._hook(layers(model)[layer].mlp, self._f_neuron)
        elif kind == "rome":
            self.Pm = torch.zeros(E, dff, r, device=DEV)
            self.R = torch.zeros(E, r, d, device=DEV)
            self._hook(layers(model)[layer].mlp.down_proj, self._f_rome)
        elif kind == "qkv":
            self.lora = []
            for li, Ly in enumerate(layers(model)):
                for name in ("q_proj", "k_proj", "v_proj"):
                    mod = getattr(Ly.self_attn, name)
                    A = P(_rand((mod.in_features, r), 1 / math.sqrt(mod.in_features), [s * 131 + li for s in seeds]))
                    B = P(torch.zeros(E, r, mod.out_features))
                    self._hook(mod, self._mk_qkv(A, B))
        else:
            raise ValueError(kind)

    # --- routing -----------------------------------------------------------------------------
    @contextmanager
    def route(self, eids):
        old = self.eid
        self.eid = eids if isinstance(eids, torch.Tensor) else torch.tensor(eids, device=DEV)
        try:
            yield
        finally:
            self.eid = old

    def _sel(self, x):
        idx = self.eid
        if idx is None or x.shape[0] != idx.shape[0]:
            return None, None
        valid = idx >= 0
        if not bool(valid.any()):
            return None, None
        return idx.clamp(min=0), valid

    def _hook(self, mod, fn):
        self.hooks.append(mod.register_forward_hook(fn))

    def _rec(self, z, idx, valid):
        if self.record is not None:
            self.record.append((z.detach().float(), idx.detach(), valid.detach()))

    def _apply(self, x, A, B, out, idx, valid, act=None):
        z = torch.einsum("bti,bir->btr", x.float(), A[idx])
        if act is not None:
            z = act(z)
        self._rec(z, idx, valid)
        delta = torch.einsum("btr,bro->bto", z, B[idx]) * valid[:, None, None]
        return out + delta.to(out.dtype)

    def _f_lowrank_AB(self, mod, inp, out):
        idx, valid = self._sel(inp[0])
        if idx is None:
            return out
        return self._apply(inp[0], self.A, self.B, out, idx, valid)

    def _f_neuron(self, mod, inp, out):
        idx, valid = self._sel(inp[0])
        if idx is None:
            return out
        x = inp[0].float()
        gpre = torch.einsum("bti,bir->btr", x, self.Wg[idx])
        if self.bg is not None:
            gpre = gpre + self.bg[idx][:, None, :]
        z = F.silu(gpre) * torch.einsum("bti,bir->btr", x, self.Wu[idx])
        self._rec(z, idx, valid)
        delta = torch.einsum("btr,bro->bto", z, self.Wd[idx]) * valid[:, None, None]
        return out + delta.to(out.dtype)

    def _f_rome(self, mod, inp, out):
        idx, valid = self._sel(inp[0])
        if idx is None:
            return out
        return self._apply(inp[0], self.Pm, self.R, out, idx, valid)

    def _mk_qkv(self, A, B):
        def f(mod, inp, out):
            idx, valid = self._sel(inp[0])
            if idx is None:
                return out
            z = torch.einsum("bti,bir->btr", inp[0].float(), A[idx])
            delta = torch.einsum("btr,bro->bto", z, B[idx]) * valid[:, None, None] * 2.0
            return out + delta.to(out.dtype)
        return f

    def remove(self):
        for h in self.hooks:
            h.remove()

    def n_params_per_edit(self):
        if self.kind == "rome":
            return int((self.Pm.numel() + self.R.numel()) // self.E)
        return int(sum(p.numel() for p in self.params) // self.E)


# ---------------------------------------------------------------------------------------------
# batched forward helpers
# ---------------------------------------------------------------------------------------------
def _pad_right(tok, seqs):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    Lm = max(map(len, seqs))
    ids = torch.full((len(seqs), Lm), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), Lm), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids.to(DEV), att.to(DEV)


def _pad_left(tok, seqs):
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    Lm = max(map(len, seqs))
    ids = torch.full((len(seqs), Lm), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), Lm), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, Lm - len(s):] = torch.tensor(s)
        att[i, Lm - len(s):] = 1
    return ids.to(DEV), att.to(DEV)


def target_rows(model, tok, edit, rows, grad=False):
    """rows: list of (prompt_ids, target_ids, eid). Returns per-row nll [B] and correct [B]."""
    seqs = [p + t for p, t, _ in rows]
    ids, att = _pad_right(tok, seqs)
    lab = torch.full_like(ids, -100)
    for i, (p, t, _) in enumerate(rows):
        lab[i, len(p) - 1:len(p) + len(t) - 1] = torch.tensor(t, device=DEV)
    eids = torch.tensor([e for _, _, e in rows], device=DEV)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, edit.route(eids):
        nll, ok, _ = L.target_nll(model, ids, att, lab)
    return nll, ok


@torch.no_grad()
def next_lp_rows(model, tok, edit, prompts, eids, mb=96):
    out = []
    for i in range(0, len(prompts), mb):
        ids, att = _pad_left(tok, [tok(p, add_special_tokens=False).input_ids for p in prompts[i:i + mb]])
        with edit.route(torch.tensor(eids[i:i + mb], device=DEV)):
            lg = model(input_ids=ids, attention_mask=att).logits[:, -1].float()
        out.append(torch.log_softmax(lg, -1))
    return torch.cat(out)


# ---------------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------------
def train_batched(model, tok, edit, groups, lr, max_steps=100, stop_nll=0.05, mb=40, anchors=None, kl_w=1.0,
                  anchor_group=4):
    """groups: list (len E) of fact lists. Plain: per-edit early stop. +reg: fixed max_steps with
    KL(base||edit) on anchors for every edit."""
    E = len(groups)
    rows = []
    for e, facts in enumerate(groups):
        for f in facts:
            for pre in PREFIXES:
                pr = (pre + " " if pre else "") + DIRECT.format(name=f["name"])
                p, t = enc_target(tok, pr, f["city"])
                rows.append((p, t, e))
    cnt = torch.zeros(E, device=DEV)
    for _, _, e in rows:
        cnt[e] += 1
    opt = torch.optim.Adam(edit.params, lr=lr)
    active = torch.ones(E, dtype=torch.bool, device=DEV)
    steps = torch.zeros(E, dtype=torch.long)
    final_nll = torch.zeros(E)
    final_kl = torch.zeros(E)
    t0 = time.time()
    refs = {}
    if anchors is not None:
        x = torch.stack(anchors["chunks"]).to(DEV)
        if "_ids" not in anchors:
            anchors["_ids"] = _pad_left(tok, [tok(p, add_special_tokens=False).input_ids for p in anchors["prompts"]])
        aids, aatt = anchors["_ids"]
        na, nc = aids.shape[0], x.shape[0]

        def ref_for(gs):
            # base reference computed with the SAME batch shape as the edited pass, so a zero edit
            # gives exactly zero KL (bf16 GEMM results depend on batch shape)
            if gs not in refs:
                with torch.no_grad(), edit.route(torch.full((gs * na,), -1, device=DEV)):
                    rp = torch.log_softmax(model(input_ids=aids.repeat(gs, 1), attention_mask=aatt.repeat(gs, 1)).logits[:, -1].float(), -1)[:na]
                with torch.no_grad(), edit.route(torch.full((gs * nc,), -1, device=DEV)):
                    rt = torch.log_softmax(model(input_ids=x.repeat(gs, 1)).logits.float(), -1)[:nc]
                refs[gs] = (rp, rt)
            return refs[gs]
    for step in range(max_steps):
        act_ids = active.nonzero().flatten().tolist()
        if not act_ids:
            break
        aset = set(act_ids)
        opt.zero_grad(set_to_none=True)
        loss_e = torch.zeros(E, device=DEV)
        rr = [r for r in rows if r[2] in aset]
        for i in range(0, len(rr), mb):
            chunk = rr[i:i + mb]
            nll, _ = target_rows(model, tok, edit, chunk, grad=True)
            eids = torch.tensor([e for _, _, e in chunk], device=DEV)
            w = 1.0 / cnt[eids]
            (nll * w).sum().backward()
            loss_e.index_add_(0, eids, (nll * w).detach())
        if anchors is not None:
            for g0 in range(0, len(act_ids), anchor_group):  # several edits per forward
                es = act_ids[g0:g0 + anchor_group]
                ref_p, ref_t = ref_for(len(es))
                eid_p = torch.tensor(es, device=DEV).repeat_interleave(na)
                with edit.route(eid_p):
                    lp = torch.log_softmax(model(input_ids=aids.repeat(len(es), 1),
                                                 attention_mask=aatt.repeat(len(es), 1)).logits[:, -1].float(), -1)
                eid_t = torch.tensor(es, device=DEV).repeat_interleave(nc)
                with edit.route(eid_t):
                    lpt = torch.log_softmax(model(input_ids=x.repeat(len(es), 1)).logits.float(), -1)
                kp = kl(ref_p[None], lp.view(len(es), na, -1)).mean(1)
                kt = kl(ref_t[None], lpt.view(len(es), nc, *lpt.shape[1:])).view(len(es), -1).mean(1)
                k = kp + kt
                (kl_w * k.sum()).backward()
                for j, e in enumerate(es):
                    final_kl[e] = k[j].item()
        for e in act_ids:
            steps[e] += 1
            final_nll[e] = loss_e[e].item()
        if anchors is None:
            done = [e for e in act_ids if loss_e[e].item() < stop_nll]
            for e in done:
                active[e] = False
        frozen = (~active).nonzero().flatten()
        saved = [p.detach()[frozen].clone() for p in edit.params] if len(frozen) else None
        opt.step()
        if saved is not None:
            with torch.no_grad():
                for p, s in zip(edit.params, saved):
                    p[frozen] = s
    out = [{"steps": int(steps[e]), "final_nll": float(final_nll[e])} for e in range(E)]
    if anchors is not None:
        for e in range(E):
            out[e]["final_anchor_kl"] = float(final_kl[e])
    sec = time.time() - t0
    for o in out:
        o["train_sec_batch"] = sec
    return out


# ---------------------------------------------------------------------------------------------
# ROME / single-layer MEMIT (fit each edit sequentially with the reference implementation)
# ---------------------------------------------------------------------------------------------
def rome_batched(model, tok, edit, groups, C, lam):
    infos = []
    for e, facts in enumerate(groups):
        single = L.Edit(model, "rome", layer=edit.layer)
        try:
            info = L.rome_fit(model, tok, single, facts, C, lam=lam)
            n = single.Pm.shape[1]
            edit.Pm[e, :, :n] = single.Pm
            edit.R[e, :n] = single.R
        finally:
            single.remove()
        infos.append(info)
    return infos


# ---------------------------------------------------------------------------------------------
# evaluation (batched across edits)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def acc_rows(model, tok, edit, items, mb=96):
    """items: list of (prompt, target, eid) -> list of bool."""
    oks = []
    for i in range(0, len(items), mb):
        rows = []
        for p, t, e in items[i:i + mb]:
            pi, ti = enc_target(tok, p, t)
            rows.append((pi, ti, e))
        _, ok = target_rows(model, tok, edit, rows)
        oks += ok.tolist()
    return oks


@torch.no_grad()
def evaluate_batched(model, tok, edit, groups, others, chunks, text_rows_per_fwd=1):
    E = len(groups)
    res = [dict() for _ in range(E)]

    def per_edit_acc(templates, field):
        items, owner = [], []
        for e, fs in enumerate(groups):
            for f in fs:
                for t in templates:
                    items.append((t.format(name=f["name"]), f[field], e))
                    owner.append(e)
        oks = acc_rows(model, tok, edit, items)
        acc = [[] for _ in range(E)]
        for o, k in zip(owner, oks):
            acc[o].append(k)
        return [sum(a) / len(a) for a in acc]

    for name, tmpl, field in (("direct", [DIRECT], "city"), ("para", PARA, "city"), ("twohop", TWOHOP, "country")):
        for e, v in enumerate(per_edit_acc(tmpl, field)):
            res[e][name] = v

    # neighbor prompts (same subject, other relation)
    for e, fs in enumerate(groups):
        nb = [t.format(name=f["name"]) for f in fs for t in NEIGHBOR]
        lb = next_lp_rows(model, tok, edit, nb, [-1] * len(nb))
        le = next_lp_rows(model, tok, edit, nb, [e] * len(nb))
        res[e]["neighbor_kl"] = float(kl(lb, le).mean())

    # other subjects, same relation
    op = [DIRECT.format(name=o["name"]) for o in others]
    lb = next_lp_rows(model, tok, edit, op, [-1] * len(op))
    pb = lb.exp()
    for e, fs in enumerate(groups):
        le = next_lp_rows(model, tok, edit, op, [e] * len(op))
        ft = torch.tensor(sorted({tok(" " + f["city"], add_special_tokens=False).input_ids[0] for f in fs}), device=DEV)
        res[e]["other_subject_kl"] = float(kl(lb, le).mean())
        res[e]["other_subject_target_p_base"] = float(pb[:, ft].sum(-1).mean())
        res[e]["other_subject_target_p_edit"] = float(le.exp()[:, ft].sum(-1).mean())

    # unrelated text: base once per chunk, then edits in small groups
    tk = torch.zeros(E); dl = torch.zeros(E); ntok = 0
    rd = torch.zeros(E); rf = torch.zeros(E)
    cap = {}
    hk = None
    if edit.layer is not None:
        hk = layers(model)[edit.layer].register_forward_hook(
            lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach().float()))
    fire_text = [[] for _ in range(E)]
    for x in chunks:
        x = x[None].to(DEV)
        with edit.route(torch.tensor([-1], device=DEV)):
            lbl = model(input_ids=x).logits[0, :-1]
        hb = cap.get("h")
        y = x[0, 1:]
        ntok += y.numel()
        for s in range(0, E, text_rows_per_fwd):
            es = list(range(s, min(E, s + text_rows_per_fwd)))
            xx = x.expand(len(es), -1)
            if edit.record is None and edit.kind != "qkv":
                edit.record = []
                recording = True
            else:
                recording = False
            with edit.route(torch.tensor(es, device=DEV)):
                lel = model(input_ids=xx).logits[:, :-1]
            if recording:
                for z, idx, valid in edit.record:
                    for j, e in enumerate(es):
                        fire_text[e].append(z[j].abs().cpu())
                edit.record = None
            he = cap.get("h")
            for j, e in enumerate(es):
                for b in range(0, y.numel(), 128):
                    lpb = torch.log_softmax(lbl[b:b + 128].float(), -1)
                    lpe = torch.log_softmax(lel[j, b:b + 128].float(), -1)
                    yy = y[b:b + 128, None]
                    tk[e] += kl(lpb, lpe).sum().item()
                    dl[e] += (lpb.gather(-1, yy) - lpe.gather(-1, yy)).sum().item()
                if hb is not None:
                    dn = (he[j] - hb[0]).norm(dim=-1)
                    rel = dn / hb[0].norm(dim=-1).clamp(min=1e-6)
                    rd[e] += dn.sum().item(); rf[e] += (rel > 0.01).sum().item()
            del lel
        del lbl
    if hk is not None:
        hk.remove()
    for e in range(E):
        t = {"kl_per_tok": float(tk[e] / ntok), "dloss_per_tok": float(dl[e] / ntok), "tokens": ntok}
        if edit.layer is not None:
            t["resid_dnorm_mean"] = float(rd[e] / ntok)
            t["resid_frac_rel_gt_1pct"] = float(rf[e] / ntok)
        res[e]["text"] = t

    # firing: per-unit trigger activation on the direct prompts vs unrelated text
    if edit.kind != "qkv":
        for e, fs in enumerate(groups):
            edit.record = []
            for f in fs:
                ids = torch.tensor(tok(DIRECT.format(name=f["name"]), add_special_tokens=False).input_ids, device=DEV)[None]
                with edit.route(torch.tensor([e], device=DEV)):
                    model(input_ids=ids)
            trig = torch.stack([z[0].abs().max(0).values for z, _, _ in edit.record]).max(0).values.cpu()
            edit.record = None
            Z = torch.cat(fire_text[e]) if fire_text[e] else None
            if Z is None:
                res[e]["firing"] = None
                continue
            ratio = Z / trig.clamp(min=1e-9)
            res[e]["firing"] = {"frac_tokens_gt10pct_trigger": float((ratio > 0.1).float().mean()),
                                "frac_tokens_gt50pct_trigger": float((ratio > 0.5).float().mean()),
                                "mean_ratio": float(ratio.mean())}
            # unit activation on OTHER people's same-relation prompts (does it fire only for its subject?)
            edit.record = []
            for o in others:
                ids = torch.tensor(tok(DIRECT.format(name=o["name"]), add_special_tokens=False).input_ids, device=DEV)[None]
                with edit.route(torch.tensor([e], device=DEV)):
                    model(input_ids=ids)
            om = torch.stack([z[0].abs().max(0).values for z, _, _ in edit.record]).cpu()  # [n_others, r]
            edit.record = None
            orat = om / trig.clamp(min=1e-9)
            res[e]["firing"].update({"other_subj_frac_gt10pct": float((orat > 0.1).float().mean()),
                                     "other_subj_frac_gt50pct": float((orat > 0.5).float().mean()),
                                     "other_subj_mean_ratio": float(orat.mean())})
    else:
        for e in range(E):
            res[e]["firing"] = None
    return res
