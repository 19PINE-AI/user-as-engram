"""
T1.2 — Probe the Engram hash mechanism on synthetic user facts.

Questions:
  Q-A: Given a user fact rendered in natural language, which (n, head, slot)
       triples does retrieval touch? How many distinct slots per fact?
  Q-B: How sparse is the slot space (total slots vs slots used by a 30-fact user)?
  Q-C: Can we identify "free" slots — addresses no fact in the user's set
       activates? (Required for surgical user-row insertion.)

Approach:
  - Import the Engram demo's NgramHashMapping (no training needed).
  - Build a small synthetic user (mirrors user-as-lora schema).
  - Render each fact as text, tokenize, hash, log all (layer, n, head, slot).
  - Aggregate.
"""
import sys, json
from collections import Counter, defaultdict
from pathlib import Path

# Allow importing from the cloned demo
sys.path.insert(0, str(Path(__file__).parent.parent / "refs"))
from engram_demo_v1 import NgramHashMapping, engram_cfg  # noqa

import numpy as np
from transformers import AutoTokenizer

# --- synthetic user (matches user-as-lora style) ---
USER_FACTS = [
    "My name is Alex Chen.",
    "I was born in 1991.",
    "I live in Portland, Oregon.",
    "My spouse is Jamie Chen.",
    "Jamie was born in 1993.",
    "We have one child named Jordan.",
    "Jordan was born in 2018.",
    "My doctor is Dr. Patel.",
    "My dentist is Dr. Lee.",
    "I am allergic to peanuts.",
    "I am allergic to shellfish.",
    "My emergency contact is my sister Robin.",
    "I work as a software engineer.",
    "I work at Globex Corp.",
    "My commute is 32 minutes.",
    "I take the bus to work.",
    "My gym day is Monday.",
    "My yoga day is Wednesday.",
    "I run on Saturdays.",
    "My favorite color is charcoal.",
    "My favorite food is ramen.",
    "I do not eat meat on Sundays.",
    "My pet is a cat named Whiskers.",
    "Whiskers was born in 2020.",
    "My birthday is March 14.",
    "Jamie's birthday is August 22.",
    "Jordan's birthday is June 5.",
    "I drive a Subaru Outback.",
    "My license plate is ABC-1234.",
    "My favorite movie is Spirited Away.",
]

def main():
    # Build the hash mapping using the demo's config
    hm = NgramHashMapping(
        engram_vocab_size=engram_cfg.engram_vocab_size,
        max_ngram_size=engram_cfg.max_ngram_size,
        n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
        n_head_per_ngram=engram_cfg.n_head_per_ngram,
        layer_ids=engram_cfg.layer_ids,
        tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
        pad_id=engram_cfg.pad_id,
        seed=engram_cfg.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)

    print("\n--- Engram hash configuration ---")
    print(f"layer_ids:           {engram_cfg.layer_ids}")
    print(f"max_ngram_size:      {engram_cfg.max_ngram_size}")
    print(f"n_head_per_ngram:    {engram_cfg.n_head_per_ngram}")
    print(f"engram_vocab_size:   {engram_cfg.engram_vocab_size}")
    print(f"compressed_vocab:    {hm.tokenizer_vocab_size}")
    for layer_id in engram_cfg.layer_ids:
        print(f"layer {layer_id} primes: {hm.vocab_size_across_layers[layer_id]}")
    total_slots = 0
    for layer_id in engram_cfg.layer_ids:
        for ngram_heads in hm.vocab_size_across_layers[layer_id]:
            total_slots += sum(ngram_heads)
    print(f"total addressable slots across layers/ngrams/heads: {total_slots:,}")

    # --- Hash each fact ---
    per_fact = []
    fact_signatures = []
    all_addresses = defaultdict(list)  # addr -> list of fact_idx
    print("\n--- Per-fact hash signatures (layer 1) ---")
    for f_idx, text in enumerate(USER_FACTS):
        ids = tokenizer(text, return_tensors="np").input_ids  # [1, T]
        T = ids.shape[1]
        hashes_per_layer = hm.hash(ids)  # {layer: [B,T,num_heads_total]}
        sig_addresses = set()
        for layer_id, h_arr in hashes_per_layer.items():
            # h_arr shape [B, T, total_heads_for_layer]
            B, Tt, H = h_arr.shape
            assert B == 1
            for t in range(Tt):
                for h_idx in range(H):
                    addr = (layer_id, h_idx, int(h_arr[0, t, h_idx]))
                    sig_addresses.add(addr)
                    all_addresses[addr].append(f_idx)
        fact_signatures.append(sig_addresses)
        per_fact.append({
            "fact_idx": f_idx,
            "text": text,
            "n_tokens": int(T),
            "n_distinct_addresses": len(sig_addresses),
        })
        if f_idx < 5:
            print(f"[fact {f_idx}] tokens={T}  distinct addresses={len(sig_addresses)}: {text!r}")

    # --- Per-user aggregate ---
    union = set()
    for s in fact_signatures:
        union |= s
    print(f"\nUser has {len(USER_FACTS)} facts; union of touched addresses = {len(union):,}")
    print(f"Total slot space:                                   {total_slots:,}")
    print(f"Occupancy fraction:                                 {len(union)/total_slots:.6%}")

    # --- Free-slot reachability check ---
    # Pick the first hash table (layer=1, ngram=2, head=0, prime size = primes[1][0][0])
    layer_id = engram_cfg.layer_ids[0]
    head0_size = hm.vocab_size_across_layers[layer_id][0][0]
    used_in_layer1_ngram2_head0 = set()
    for addr in union:
        l, h, s = addr
        if l == layer_id and h == 0:
            used_in_layer1_ngram2_head0.add(s)
    print(f"\nLayer {layer_id}, ngram=2, head=0:")
    print(f"  prime table size:   {head0_size:,}")
    print(f"  slots used by user: {len(used_in_layer1_ngram2_head0):,}")
    print(f"  free slots:         {head0_size - len(used_in_layer1_ngram2_head0):,}")
    print(f"  free fraction:      {1 - len(used_in_layer1_ngram2_head0)/head0_size:.6%}")

    # --- Address frequency: which addresses are touched by multiple facts? ---
    addr_collision = Counter({addr: len(set(fids)) for addr, fids in all_addresses.items() if len(set(fids)) > 1})
    print(f"\nAddresses touched by ≥2 facts: {len(addr_collision):,} (this is intra-user fact overlap, expected for shared tokens)")
    if addr_collision:
        sample = addr_collision.most_common(5)
        print("Top intra-user collision addresses:")
        for addr, n_facts in sample:
            print(f"  layer={addr[0]} head={addr[1]} slot={addr[2]}: shared by {n_facts} facts")

    # Save raw data
    out = {
        "config": {
            "layer_ids": list(engram_cfg.layer_ids),
            "max_ngram_size": engram_cfg.max_ngram_size,
            "n_head_per_ngram": engram_cfg.n_head_per_ngram,
            "engram_vocab_size": list(engram_cfg.engram_vocab_size),
            "total_slots": total_slots,
        },
        "per_fact": per_fact,
        "user_summary": {
            "n_facts": len(USER_FACTS),
            "union_addresses": len(union),
            "occupancy_fraction": len(union) / total_slots,
        },
    }
    out_path = Path(__file__).parent.parent / "results" / "t1_2_probe_hash.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
