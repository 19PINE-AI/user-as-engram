"""
Multi-user Engram serving system.

Architecture:
  - One server holds the frozen base + global Engram tables.
  - Per user, an OverrideMap: List[(global_row_idx, row_vector)].
  - Per organisation/domain, the same OverrideMap structure.
  - On request: resolve user_id → optional org_id → apply both override maps
    in order (org first, then user) → generate → restore.

The override apply/restore is a sequence of `embedding.weight.data[rows] = ...`
operations. No graph rewriting, no router. Disjoint addresses commute, so the
order of stacking different domains doesn't matter for non-overlapping rows.

Public API:
  server = EngramServer(ckpt_dir)
  server.register_user(user_id, facts, training='joint_opt' | 'opt' | 'unembed_p', steps=...)
  server.register_org(org_id, facts, ...)
  out_text = server.serve(user_id, prompt, max_tokens=20, org_id=None)

Internal:
  server._train_overrides(facts, training)  → returns OverrideMap
  server._apply(maps)                        → save originals, write
  server._restore(originals_list)            → restore in reverse order
"""
from __future__ import annotations
import os, json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from nanochat.tokenizer import get_tokenizer
from scripts.insertion_strategies_v2 import (
    load_model, trigger_global_rows, restore_rows,
    make_marker_OPT, make_marker_UNEMBED_P,
)


@dataclass
class OverrideMap:
    """A trained set of (global_row_idx, row_vector) pairs for one user/org."""
    rows_global: torch.Tensor              # [num_rows] long, on device
    values: torch.Tensor                   # [num_rows, embed_dim] on device, dtype=embedding dtype
    fact_meta: List[dict] = field(default_factory=list)  # [{prompt, gold, ...}, ...]
    train_time_s: float = 0.0
    n_facts: int = 0


def _train_joint_opt(model, eng, last_layer, Wv_pinv, total_heads, embed_dim,
                      tokenizer, facts, device, scale=20.0, steps=2000, lr=0.5):
    """Joint OPT: trains all rows together. Returns OverrideMap."""
    bos = tokenizer.get_bos_token_id()
    fact_data = []
    all_global_rows_set = set()
    for f in facts:
        ids = tokenizer.encode(f["prompt"], prepend=bos)
        gold_id = tokenizer.encode(f["gold"])[0]
        idx_t = torch.tensor([ids], dtype=torch.long, device=device)
        trig_pos = len(ids) - 1
        gr = trigger_global_rows(eng, idx_t, last_layer, trig_pos, user_salt=0)
        init = make_marker_UNEMBED_P(model, eng, last_layer, gold_id, idx_t, trig_pos,
                                       scale, total_heads, embed_dim, Wv_pinv=Wv_pinv)
        fact_data.append({"prompt_ids": ids, "gold_id": gold_id, "global_rows": gr, "init": init})
        all_global_rows_set.update(gr.tolist())

    all_rows_list = sorted(all_global_rows_set)
    all_rows_t = torch.tensor(all_rows_list, dtype=torch.long, device=device)
    addr_to_leaf = {a: i for i, a in enumerate(all_rows_list)}
    addr_to_leaf_t = torch.full((eng.tables[str(last_layer)].embedding.weight.size(0),),
                                  -1, dtype=torch.long, device=device)
    for a, li in addr_to_leaf.items():
        addr_to_leaf_t[a] = li

    # Initialise: average UNEMBED_P inits where rows collide
    init_stack = torch.zeros(len(all_rows_list), embed_dim, device=device)
    counts = torch.zeros(len(all_rows_list), device=device)
    for fd in fact_data:
        for i, a in enumerate(fd["global_rows"].tolist()):
            li = addr_to_leaf[a]
            init_stack[li] += fd["init"][i]
            counts[li] += 1
    counts = counts.clamp_min(1)
    init_stack = init_stack / counts.unsqueeze(-1)

    tbl = eng.tables[str(last_layer)]
    saved_orig = tbl.embedding.weight.data[all_rows_t].clone()

    # Snapshot grads
    grad_snap = []
    for p in model.parameters():
        grad_snap.append(p.requires_grad)
        p.requires_grad_(False)

    # Zero out and install hook
    with torch.no_grad():
        tbl.embedding.weight.data[all_rows_t] = 0
    row_leaves = init_stack.detach().clone().to(tbl.embedding.weight.dtype).requires_grad_(True)

    def hook(module, inputs, output):
        in_idx = inputs[0]
        leaf_idx = addr_to_leaf_t[in_idx]
        mask = (leaf_idx >= 0)
        if not mask.any(): return output
        flat = leaf_idx.clamp_min(0)
        added = row_leaves[flat].to(output.dtype)
        m = mask.unsqueeze(-1).to(output.dtype)
        return output * (1 - m) + added * m

    handle = tbl.embedding.register_forward_hook(hook)
    optim = torch.optim.Adam([row_leaves], lr=lr)
    t0 = time.time()
    try:
        for step in range(steps):
            i = torch.randint(0, len(fact_data), (1,)).item()
            fd = fact_data[i]
            x = torch.tensor([fd["prompt_ids"]], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :]
            loss = F.cross_entropy(logits.unsqueeze(0).float(),
                                     torch.tensor([fd["gold_id"]], device=device))
            grads = torch.autograd.grad(loss, [row_leaves])
            if row_leaves.grad is None:
                row_leaves.grad = grads[0].clone()
            else:
                row_leaves.grad.copy_(grads[0])
            optim.step()
            optim.zero_grad()
        train_time = time.time() - t0
    finally:
        handle.remove()
        # Restore original rows
        with torch.no_grad():
            tbl.embedding.weight.data[all_rows_t] = saved_orig
        for p, rg in zip(model.parameters(), grad_snap):
            p.requires_grad_(rg)

    return OverrideMap(
        rows_global=all_rows_t,
        values=row_leaves.detach().to(tbl.embedding.weight.dtype).clone(),
        fact_meta=[{"prompt": f["prompt"], "gold": f["gold"]} for f in facts],
        train_time_s=train_time,
        n_facts=len(facts),
    )


class EngramServer:
    """Multi-user Engram serving with per-user override maps."""
    def __init__(self, ckpt_dir: str, device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = get_tokenizer()
        self.model, self.config = load_model(ckpt_dir, self.tokenizer, device)
        self.eng = self.model.engram
        self.last_layer = max(self.config.engram_layer_ids)
        self.Wv_pinv = torch.linalg.pinv(
            self.eng.layers_module[str(self.last_layer)].value_proj.weight.data.float())
        self.embed_dim = self.eng.embed_per_head
        self.total_heads = self.config.engram_n_head_per_ngram * (self.config.engram_max_ngram_size - 1)
        self.users: Dict[str, OverrideMap] = {}
        self.orgs: Dict[str, OverrideMap] = {}

    def _train(self, facts, training='joint_opt', steps=2000, lr=0.5, scale=20.0):
        if training == 'joint_opt':
            return _train_joint_opt(self.model, self.eng, self.last_layer, self.Wv_pinv,
                                     self.total_heads, self.embed_dim, self.tokenizer,
                                     facts, self.device, scale, steps, lr)
        else:
            raise ValueError(f"Training mode {training} not implemented in server (use joint_opt)")

    def register_user(self, user_id: str, facts: list, **train_kw):
        ov = self._train(facts, **train_kw)
        self.users[user_id] = ov
        return {"user_id": user_id, "n_facts": ov.n_facts, "train_time_s": ov.train_time_s,
                "n_rows": int(ov.rows_global.numel()),
                "size_kb": ov.rows_global.numel() * self.embed_dim * 4 / 1024}

    def register_org(self, org_id: str, facts: list, **train_kw):
        ov = self._train(facts, **train_kw)
        self.orgs[org_id] = ov
        return {"org_id": org_id, "n_facts": ov.n_facts, "train_time_s": ov.train_time_s,
                "n_rows": int(ov.rows_global.numel()),
                "size_kb": ov.rows_global.numel() * self.embed_dim * 4 / 1024}

    def _apply(self, maps: List[OverrideMap]) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        tbl = self.eng.tables[str(self.last_layer)]
        saved = []
        for m in maps:
            originals = tbl.embedding.weight.data[m.rows_global].clone()
            saved.append((m.rows_global, originals))
            tbl.embedding.weight.data[m.rows_global] = m.values
        return saved

    def _restore(self, saved):
        tbl = self.eng.tables[str(self.last_layer)]
        for rows, originals in reversed(saved):
            tbl.embedding.weight.data[rows] = originals

    @torch.no_grad()
    def serve(self, user_id: Optional[str], prompt: str,
              max_tokens: int = 1, org_id: Optional[str] = None) -> dict:
        """Generate `max_tokens` tokens after `prompt`, with the given user's
        and (optionally) org's overrides applied. Returns a dict with timing,
        the generated text, and the top-1/top-5 next-token IDs.

        max_tokens=1 is the recall-test mode (for evaluating fact retrieval).
        """
        bos = self.tokenizer.get_bos_token_id()
        ids = self.tokenizer.encode(prompt, prepend=bos)
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)

        active_maps = []
        if org_id is not None and org_id in self.orgs:
            active_maps.append(self.orgs[org_id])
        if user_id is not None and user_id in self.users:
            active_maps.append(self.users[user_id])

        timings = {}
        torch.cuda.synchronize() if self.device == "cuda" else None

        t0 = time.time()
        saved = self._apply(active_maps)
        torch.cuda.synchronize() if self.device == "cuda" else None
        timings["apply_ms"] = (time.time() - t0) * 1000

        try:
            t1 = time.time()
            cur = idx
            generated = []
            top1_id = top5_ids = None
            for step in range(max_tokens):
                logits = self.model(cur)[0, -1, :]
                if top1_id is None:
                    top1_id = int(logits.argmax().item())
                    _, topi = torch.topk(logits, 5)
                    top5_ids = [int(t.item()) for t in topi]
                next_id = int(logits.argmax().item())
                generated.append(next_id)
                cur = torch.cat([cur, torch.tensor([[next_id]], dtype=torch.long, device=self.device)], dim=1)
            torch.cuda.synchronize() if self.device == "cuda" else None
            timings["forward_ms"] = (time.time() - t1) * 1000
        finally:
            t2 = time.time()
            self._restore(saved)
            torch.cuda.synchronize() if self.device == "cuda" else None
            timings["restore_ms"] = (time.time() - t2) * 1000

        return {
            "user_id": user_id, "org_id": org_id,
            "prompt": prompt,
            "generated_text": self.tokenizer.decode(generated),
            "top1_id": top1_id,
            "top1_text": self.tokenizer.decode([top1_id]),
            "top5_ids": top5_ids,
            "top5_text": [self.tokenizer.decode([t]) for t in top5_ids],
            "timings": timings,
            "n_active_maps": len(active_maps),
        }
