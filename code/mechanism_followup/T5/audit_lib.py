"""Shared neuron-audit utilities for T5/T8 (nanochat d8 models, relu^2 MLPs).

Neuron = one of the 4*n_embd hidden units of a block's MLP.
  pre  = c_fc(norm(x))[j]           (signed; the essay's "negative pole" lives here)
  post = relu(pre)^2                (what c_proj reads)
"""
import os, sys, json, math, random
import numpy as np
import torch
import torch.nn.functional as F

HARNESS = os.path.expanduser("~/clean-tests/user-as-engram/code/nanochat_harness")
sys.path.insert(0, HARNESS)
os.environ.setdefault("NANOCHAT_BASE_DIR", os.path.expanduser("~/clean-tests/nanochat_ckpts"))

from nanochat.gpt import GPT, GPTConfig  # noqa: E402
from nanochat.tokenizer import get_tokenizer  # noqa: E402
from nanochat.common import get_base_dir  # noqa: E402

BASE = os.environ["NANOCHAT_BASE_DIR"]
VAL_SHARD = os.path.join(BASE, "base_data_climbmix", "shard_06542.parquet")


def cap_gpu(gb):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gb / 96.0)


def load_model(ckpt_dir, device="cuda", state_path=None):
    tok = get_tokenizer()
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        d = json.load(f)
    keys = {"sequence_len", "vocab_size", "n_layer", "n_head", "n_kv_head", "n_embd",
            "window_pattern", "engram_layer_ids", "engram_max_ngram_size",
            "engram_vocab_per_ngram", "engram_n_head_per_ngram", "engram_n_embed_per_ngram",
            "engram_kernel_size"}
    kw = {k: d[k] for k in keys if k in d}
    if "engram_layer_ids" in kw:
        kw["engram_layer_ids"] = tuple(kw["engram_layer_ids"])
    cfg = GPTConfig(**kw)
    with torch.device("meta"):
        model = GPT(cfg)
    model.to_empty(device=device)
    model.init_weights()
    if cfg.engram_layer_ids:
        model.attach_engram(tok, base_dir=get_base_dir())
    sd = torch.load(state_path or os.path.join(ckpt_dir, "model.pt"), map_location=device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg, tok


def corpus_tokens(tok, n_tokens, seq_len=256, skip_docs=0, seed=0):
    """Held-out text from the val shard, packed into [N, seq_len] with BOS at each doc start."""
    import pyarrow.parquet as pq
    texts = pq.read_table(VAL_SHARD, columns=["text"]).column("text").to_pylist()
    texts = texts[skip_docs:]
    bos = tok.get_bos_token_id()
    buf, seqs, used = [], [], 0
    for i, t in enumerate(texts):
        ids = [bos] + tok.encode(t)
        buf.extend(ids)
        while len(buf) >= seq_len:
            seqs.append(buf[:seq_len]); buf = buf[seq_len:]
            if len(seqs) * seq_len >= n_tokens:
                return torch.tensor(seqs), skip_docs + i + 1
    return torch.tensor(seqs), skip_docs + len(texts)


class MLPTap:
    """Captures pre-activations of chosen layers; optionally overrides one neuron's post-activation."""

    def __init__(self, model):
        self.model = model
        self.capture = {}          # layer -> tensor [B,T,H] (pre)
        self.want = set()
        self.override = None       # (layer, neuron, tensor[B,T] post-values) or (layer, neuron, 'zero'/'mean', val)
        self._orig = {}
        for li, blk in enumerate(model.transformer.h):
            mlp = blk.mlp
            self._orig[li] = mlp.forward
            mlp.forward = self._make_fwd(li, mlp)

    def _make_fwd(self, li, mlp):
        def fwd(x):
            pre = mlp.c_fc(x)
            if li in self.want:
                self.capture[li] = pre.detach()
            post = F.relu(pre).square()
            ov = self.override
            if ov is not None and ov[0] == li:
                post = post.clone()
                post[..., ov[1]] = ov[2].to(post.dtype)
            return mlp.c_proj(post)
        return fwd


def logits_of(model, idx):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(idx)
    return out.float()


def direct_logit_tokens(model, tok, layer, neuron, k=10):
    """Tokens most promoted / suppressed by the neuron's output direction through the unembedding."""
    w = model.transformer.h[layer].mlp.c_proj.weight[:, neuron].float()  # [n_embd]
    lm = model.lm_head.weight.float()  # [V, n_embd]
    s = lm @ (w / w.norm())
    V = tok.get_vocab_size()
    s = s[:V]
    top = torch.topk(s, k).indices.tolist()
    bot = torch.topk(-s, k).indices.tolist()
    dec = lambda ids: [tok.decode([i]) for i in ids]
    return dec(top), dec(bot)
