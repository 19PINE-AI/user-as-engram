"""
Engram module ported to nanochat.

Adapted from deepseek-ai/Engram's engram_demo_v1.py to:
  - drop the multi-branch (hc_mult=4) integration; nanochat is single-branch
  - use nanochat's RustBPETokenizer (tiktoken) instead of HuggingFace
  - take torch tensors as input_ids (instead of numpy)
  - keep weights on the same device as the host module

The architecture follows Cheng et al. 2026 (arXiv:2601.07372) §2:
  - tokenizer compression (canonical-collapse via NFKC + lowercase + space norm)
  - multi-head hashing with K hash heads per N-gram order
  - context-aware sigmoid gate
  - depthwise causal short conv
  - residual into the backbone

Key differences from the demo for clarity:
  - explicit constructor args (no global engram_cfg)
  - returns the residual contribution (caller decides to add it)
  - `forward(hidden_states, input_ids)` is robust to either np or torch inputs
"""
import math
import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sympy import isprime
from tokenizers import normalizers, Regex


@dataclass
class EngramSubConfig:
    layer_ids: List[int]                  # which transformer layers get an Engram module
    max_ngram_size: int = 3               # 2..max_ngram_size suffix N-grams
    engram_vocab_per_ngram: int = 100_000 # base table size per N-gram order before next-prime search
    n_head_per_ngram: int = 8
    n_embed_per_ngram: int = 256          # total per-N-gram embedding width (split across heads)
    kernel_size: int = 4
    pad_id: int = 0
    seed: int = 0


class _AutoCastLinear(nn.Linear):
    """Mirrors nanochat.gpt.Linear: cast weights to input dtype in forward.

    The Engram embedding tables are cast to compute dtype (bf16) to save memory,
    so the projection weights need to follow the activation dtype rather than
    the master weight dtype.
    """
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype) if self.bias is not None else None)


class _AutoCastConv1d(nn.Conv1d):
    """Conv1d that casts weights to input dtype in forward (depthwise variant in our use)."""
    def forward(self, x):
        return self._conv_forward(x, self.weight.to(dtype=x.dtype), self.bias.to(dtype=x.dtype) if self.bias is not None else None)


def _find_next_prime(start: int, seen_primes: set) -> int:
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


def build_canonical_collapse_map(tokenizer, save_path: Optional[str] = None) -> np.ndarray:
    """
    Build a surjection P: V -> V' that collapses semantically-equivalent tokens
    (e.g. "Apple" and " apple") into a single canonical id.

    Mirrors the demo's CompressedTokenizer normalisation pipeline. Slow (~minutes
    on the first run), so result is cached to disk if save_path is given.
    """
    if save_path and os.path.exists(save_path):
        with open(save_path, "rb") as f:
            arr, num_new = pickle.load(f)
        return arr, num_new

    SENTINEL = ""
    norm = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
        normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
        normalizers.Replace(Regex(r"^ $"), SENTINEL),
        normalizers.Strip(),
        normalizers.Replace(SENTINEL, " "),
    ])

    vocab_size = tokenizer.get_vocab_size()
    old2new = {}
    key2new = {}
    n_new = 0
    for tid in range(vocab_size):
        text = tokenizer.id_to_token(tid)
        # tiktoken/rustbpe id_to_token returns a string (decoded bytes)
        if "�" in text:
            key = f"__b__{tid}"
        else:
            normed = norm.normalize_str(text)
            key = normed if normed else text
        nid = key2new.get(key)
        if nid is None:
            nid = n_new
            key2new[key] = nid
            n_new += 1
        old2new[tid] = nid

    arr = np.empty(vocab_size, dtype=np.int64)
    for tid in range(vocab_size):
        arr[tid] = old2new[tid]

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump((arr, n_new), f)

    return arr, n_new


class NgramHashMapping:
    """Computes hash slot indices for suffix N-grams.

    For each layer in `layer_ids`, for each N in [2..max_ngram_size], for each
    of `n_head_per_ngram` hash heads, returns an int64 index into a per-head
    embedding table. The hash is multiplicative-XOR with per-(layer, position)
    odd multipliers, and per-head prime moduli.
    """
    def __init__(
        self,
        layer_ids: List[int],
        max_ngram_size: int,
        engram_vocab_per_ngram: int,
        n_head_per_ngram: int,
        compressed_vocab_size: int,
        compressed_pad_id: int,
        seed: int = 0,
    ):
        self.layer_ids = list(layer_ids)
        self.max_ngram_size = max_ngram_size
        self.engram_vocab_per_ngram = engram_vocab_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.compressed_vocab_size = compressed_vocab_size
        self.pad_id = compressed_pad_id
        self.seed = seed

        # Per-layer odd multipliers (deterministic, seeded).
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // max(self.compressed_vocab_size, 1))
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}
        for layer_id in self.layer_ids:
            base_seed = int(self.seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(low=0, high=half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = r * 2 + 1   # force odd

        # Find disjoint primes near `engram_vocab_per_ngram` for each (layer, ngram, head).
        seen_primes = set()
        self.head_primes = {}
        for layer_id in self.layer_ids:
            self.head_primes[layer_id] = []
            start = self.engram_vocab_per_ngram - 1
            for n in range(2, self.max_ngram_size + 1):
                primes_for_this_ngram = []
                cur = start
                for _ in range(self.n_head_per_ngram):
                    p = _find_next_prime(cur, seen_primes)
                    seen_primes.add(p)
                    primes_for_this_ngram.append(p)
                    cur = p
                self.head_primes[layer_id].append(primes_for_this_ngram)

    def total_slots(self) -> int:
        return sum(sum(heads) for layer in self.head_primes.values() for heads in layer)

    def slots_per_layer(self, layer_id: int) -> List[int]:
        # flattened list, length = (max_ngram_size - 1) * n_head_per_ngram
        return [p for heads in self.head_primes[layer_id] for p in heads]

    def _hash_layer(self, x_compressed: np.ndarray, layer_id: int, user_salt: int = 0) -> np.ndarray:
        """x_compressed: [B, T] int64 of canonical ids. Returns [B, T, total_heads].

        `user_salt` is XORed into the multiplicative-XOR mix before the modulo.
        Salt 0 reproduces the original hash (used during pretraining and global
        retrieval). Per-user salts (large odd ints) shift the address space so
        identical surface triggers map to disjoint rows for different users.
        """
        B, T = x_compressed.shape
        multipliers = self.layer_multipliers[layer_id]
        salt = np.int64(user_salt)

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x_compressed
            shifted = np.pad(x_compressed, ((0, 0), (k, 0)),
                             mode="constant", constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_idx = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            if salt != 0:
                mix = np.bitwise_xor(mix, salt)
            head_primes = self.head_primes[layer_id][n_idx]
            for j in range(self.n_head_per_ngram):
                p = int(head_primes[j])
                head_hash = mix % p
                all_hashes.append(head_hash.astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash_all_layers(self, input_ids_compressed: np.ndarray, user_salt: int = 0) -> dict:
        return {lid: self._hash_layer(input_ids_compressed, lid, user_salt=user_salt) for lid in self.layer_ids}


class MultiHeadEmbedding(nn.Module):
    """One big embedding table indexed by (head_offset + per-head local index)."""
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.D = D
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long), persistent=False)
        self.total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=self.total_N, embedding_dim=D)

    def forward(self, head_indices: torch.Tensor) -> torch.Tensor:
        """head_indices: [..., num_heads] -> [..., num_heads, D]"""
        shifted = head_indices + self.offsets.to(head_indices.device)
        return self.embedding(shifted)


class _ShortConv(nn.Module):
    """Depthwise causal conv with SiLU activation (single-branch variant)."""
    def __init__(self, hidden_size: int, kernel_size: int = 4, dilation: int = 1):
        super().__init__()
        self.conv = _AutoCastConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]  ->  conv on C-dim, causally padded
        B, T, C = x.shape
        y = x.transpose(1, 2)
        y = self.conv(y)[..., :T]
        y = self.act(y)
        y = y.transpose(1, 2).contiguous()
        return y


class EngramLayer(nn.Module):
    """
    A single Engram module attached at a given layer.
    Mirrors the paper's §2.3 single-branch fusion:
        e_t = concat over (n, h) of E_{n,h}[hash(...)]
        k_t = W_K e_t,  v_t = W_V e_t
        alpha_t = sigmoid(RMS(h_t) . RMS(k_t) / sqrt(d))
        v~_t = alpha_t * v_t
        Y = SiLU(Conv1D(RMS(v~)) ) + v~
        H <- H + Y
    """
    def __init__(
        self,
        hidden_size: int,
        embed_per_head: int,
        max_ngram_size: int,
        n_head_per_ngram: int,
        kernel_size: int = 4,
    ):
        super().__init__()
        engram_hidden = (max_ngram_size - 1) * n_head_per_ngram * embed_per_head
        self.engram_hidden = engram_hidden
        self.value_proj = _AutoCastLinear(engram_hidden, hidden_size, bias=False)
        self.key_proj = _AutoCastLinear(engram_hidden, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        self.k_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        # dilation = max_ngram so the conv kernel doesn't smear adjacent N-grams together too hard
        self.short_conv = _ShortConv(hidden_size, kernel_size=kernel_size, dilation=max_ngram_size)
        self.short_conv_norm = nn.RMSNorm(hidden_size, elementwise_affine=False)
        # Conv params init zero so Engram starts as identity (paper's recipe).
        nn.init.zeros_(self.short_conv.conv.weight)

    def forward(self, hidden_states: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [B, T, hidden_size]
        embeddings:    [B, T, n_head_per_ngram * (max_ngram_size - 1), embed_per_head]
        returns:       [B, T, hidden_size] residual to add to hidden_states
        """
        # Flatten heads/embed to a single feature dim
        e_t = embeddings.flatten(start_dim=-2)  # [B, T, engram_hidden]
        k_t = self.key_proj(e_t)
        v_t = self.value_proj(e_t)
        # Cast hidden_states to match k/v dtype (RMSNorm weight may be in different dtype than activations)
        h = hidden_states.to(k_t.dtype)
        gate = (self.q_norm(h) * self.k_norm(k_t)).sum(dim=-1) / math.sqrt(h.shape[-1])
        # Mirrors the demo's gate magnitude smoothing: sqrt-ish on absolute, sign-preserving
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gate = gate.sigmoid().unsqueeze(-1)  # [B, T, 1]
        v_gated = gate * v_t
        y = self.short_conv(self.short_conv_norm(v_gated)) + v_gated
        return y.to(hidden_states.dtype)


class EngramSet(nn.Module):
    """
    All Engram modules + the shared hash mapping + the embedding tables.
    `forward_layer(hidden_states, input_ids, layer_idx)` is called from each
    transformer block at the configured `layer_ids`.
    """
    def __init__(
        self,
        cfg: EngramSubConfig,
        hidden_size: int,
        compressed_vocab_size: int,
        compressed_pad_id: int,
        token_canonical_map: torch.Tensor,
    ):
        super().__init__()
        self.cfg = cfg
        embed_per_head = cfg.n_embed_per_ngram // cfg.n_head_per_ngram
        self.embed_per_head = embed_per_head
        # Hash mapping is stateless (numpy-based), no parameters
        self.hash_mapping = NgramHashMapping(
            layer_ids=cfg.layer_ids,
            max_ngram_size=cfg.max_ngram_size,
            engram_vocab_per_ngram=cfg.engram_vocab_per_ngram,
            n_head_per_ngram=cfg.n_head_per_ngram,
            compressed_vocab_size=compressed_vocab_size,
            compressed_pad_id=compressed_pad_id,
            seed=cfg.seed,
        )
        # Per-layer embedding tables (each layer has its own MultiHeadEmbedding)
        self.tables = nn.ModuleDict()
        self.layers_module = nn.ModuleDict()
        for lid in cfg.layer_ids:
            list_of_N = self.hash_mapping.slots_per_layer(lid)
            tbl = MultiHeadEmbedding(list_of_N, embed_per_head)
            self.tables[str(lid)] = tbl
            self.layers_module[str(lid)] = EngramLayer(
                hidden_size=hidden_size,
                embed_per_head=embed_per_head,
                max_ngram_size=cfg.max_ngram_size,
                n_head_per_ngram=cfg.n_head_per_ngram,
                kernel_size=cfg.kernel_size,
            )
        # Token canonical-collapse map (frozen lookup, registered as buffer)
        # token_canonical_map: int64 tensor of shape [raw_vocab_size]
        self.register_buffer("token_canonical_map", token_canonical_map.long(), persistent=False)
        self._cached_compressed_ids = None
        self._cached_hashes = None
        self._cached_input_id_ptr = None
        # Per-request user salt for the hash. Default 0 (= legacy/global).
        # Set this attribute before model.forward() to retrieve from a per-user
        # subspace of the hash table.
        self.user_salt = 0

    def total_engram_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def total_table_params(self) -> int:
        return sum(t.embedding.weight.numel() for t in self.tables.values())

    def _compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T] long, values in raw vocab
        return self.token_canonical_map[input_ids]

    def _compute_hashes_if_needed(self, input_ids: torch.Tensor):
        """Compute hash indices for all layers once per forward pass and cache.

        Includes user_salt in the cache key so per-user salts produce per-user
        hash addresses without invalidating the per-step cache reuse pattern.
        """
        cache_key = (input_ids.data_ptr(), tuple(input_ids.shape), int(self.user_salt))
        if self._cached_input_id_ptr == cache_key and self._cached_hashes is not None:
            return self._cached_hashes
        comp = self._compress(input_ids).cpu().numpy()
        hashes_per_layer = self.hash_mapping.hash_all_layers(comp, user_salt=int(self.user_salt))
        device = input_ids.device
        self._cached_hashes = {lid: torch.from_numpy(h).to(device) for lid, h in hashes_per_layer.items()}
        self._cached_input_id_ptr = cache_key
        return self._cached_hashes

    def reset_cache(self):
        self._cached_hashes = None
        self._cached_input_id_ptr = None

    def forward_layer(self, hidden_states: torch.Tensor, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """If layer_idx is configured, returns the Engram residual to add. Else returns 0."""
        if layer_idx not in self.cfg.layer_ids:
            return None  # caller treats None as no-op
        hashes = self._compute_hashes_if_needed(input_ids)
        h_idx = hashes[layer_idx]  # [B, T, total_heads]
        tbl = self.tables[str(layer_idx)]
        embeddings = tbl(h_idx)    # [B, T, total_heads, embed_per_head]
        layer_mod = self.layers_module[str(layer_idx)]
        return layer_mod(hidden_states, embeddings)
