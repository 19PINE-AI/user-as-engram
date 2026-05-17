"""
T1.3 — Slot read/write through the Engram gate.

Existence proof: when we surgically write a known vector into the rows that
a target N-gram hashes to, the Engram forward output at *exactly that
position* changes in a predictable, recoverable way, while output at
*other positions* (whose hash slots we did not touch) remains unchanged.

This does not require a trained model — random W_K, W_V are fine for the
structural question. What we are testing is whether the architecture
*mechanically* admits surgical insertion. If it does, Tier 3's Q1
(subspace alignment) becomes meaningful; if it doesn't, the whole
direction is dead.

Protocol:
  Step A. Default Engram (random embedding rows) → record output O_A
          at every position.
  Step B. Identify trigger position t* and the (head, slot) set the
          forward will retrieve at that position.
          Identify also a control non-trigger position t_c.
  Step C. Overwrite the embedding rows at the trigger slots with a known
          large unit-direction `v_user` (broadcast across heads).
          Forward again → record O_B.
  Step D. Compute:
            - delta_t* = ||O_B[t*] - O_A[t*]||
            - delta_other = mean over t != t* of ||O_B[t] - O_A[t]||
            (Want delta_t*  >>  delta_other)
          - cos(O_B[t*] - O_A[t*],  W_V(v_user_concat))
            (sanity: the change at t* aligns with the value-path projection
             of our written vector)

Output: pass/fail on the existence proof, plus quantitative numbers.
"""
import sys, json, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "refs"))
from engram_demo_v1 import Engram, engram_cfg, backbone_config  # noqa

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    # We test on layer_id = 1 (the first Engram layer in the demo's config).
    layer_id = engram_cfg.layer_ids[0]
    print(f"Testing layer_id={layer_id}")

    eng = Engram(layer_id=layer_id).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)

    # The trigger phrase ends in a unique-ish 3-gram. We want a position
    # whose suffix-3-gram is unlikely to appear elsewhere in normal text.
    text = "Today my doctor is Dr Patel and that is final."
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids.to(DEVICE)  # [1, T]
    B, T = input_ids.shape
    print(f"\nText: {text!r}")
    print(f"Tokens ({T}): {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")

    # Compute hash addresses for every position
    hashes = eng.hash_mapping.hash(input_ids.cpu().numpy())[layer_id]  # [B, T, total_heads]
    hashes = torch.from_numpy(hashes).to(DEVICE)  # int64
    Hh = hashes.shape[-1]

    # The MultiHeadEmbedding offsets each head's local index into a single big embedding.
    # We need *global* row indices (= head_offset + local_slot).
    offsets = eng.multi_head_embedding.offsets.to(DEVICE)  # [Hh]
    print(f"head offsets ({len(offsets)}): {offsets.tolist()[:8]}...")

    # Build hidden_states: deterministic, normalized; the demo's BackBoneConfig
    # has hidden_size=1024, hc_mult=4, so hidden is [B, T, hc_mult, hidden_size].
    hidden_size = backbone_config.hidden_size
    hc_mult = backbone_config.hc_mult
    h = torch.randn(B, T, hc_mult, hidden_size, device=DEVICE) * 0.5

    # ---- Step A: default forward ----
    with torch.no_grad():
        O_A = eng(hidden_states=h, input_ids=input_ids.cpu().numpy())  # [B, T, hc_mult, hidden_size]
    print(f"\nDefault forward output shape: {tuple(O_A.shape)}")
    print(f"Default output norm range: min={O_A.norm(dim=-1).min().item():.4f}  "
          f"max={O_A.norm(dim=-1).max().item():.4f}")

    # ---- Pick trigger position ----
    # Use the position of "Patel" (unique-ish single token), which makes the
    # suffix 2-gram and 3-gram both rare.
    target_token_str = "Patel"
    tok_ids = input_ids[0].tolist()
    decoded = [tokenizer.decode([t]) for t in tok_ids]
    t_star = None
    for i, d in enumerate(decoded):
        if target_token_str in d:
            t_star = i
            break
    assert t_star is not None, f"could not find {target_token_str!r}"
    # Control position: a generic word like "and"
    t_ctrl = None
    for i, d in enumerate(decoded):
        if "and" in d.strip():
            t_ctrl = i
            break
    assert t_ctrl is not None and t_ctrl != t_star
    print(f"\nTrigger position t*={t_star}  (token={decoded[t_star]!r})")
    print(f"Control position t_c={t_ctrl} (token={decoded[t_ctrl]!r})")

    # ---- Step B: identify trigger slots & inspect non-overlap ----
    trigger_local = hashes[0, t_star, :]                # [Hh]
    trigger_global = trigger_local + offsets             # [Hh]
    ctrl_global = hashes[0, t_ctrl, :] + offsets         # [Hh]

    overlap = set(trigger_global.tolist()) & set(ctrl_global.tolist())
    print(f"Trigger global rows: {trigger_global.tolist()[:6]} ... ({Hh} total)")
    print(f"Control global rows: {ctrl_global.tolist()[:6]} ... ({Hh} total)")
    print(f"Overlap between trigger & control rows: {len(overlap)} of {Hh}")

    # Are any trigger rows shared with OTHER positions in this same sentence?
    trigger_set = set(trigger_global.tolist())
    other_positions_overlap = []
    for t in range(T):
        if t == t_star:
            continue
        other_global = (hashes[0, t, :] + offsets).tolist()
        n_shared = len(trigger_set & set(other_global))
        other_positions_overlap.append((t, decoded[t], n_shared))
    n_other_pos_with_overlap = sum(1 for _, _, n in other_positions_overlap if n > 0)
    print(f"Other positions sharing ≥1 trigger row: {n_other_pos_with_overlap} of {T-1}")
    for t, tok_str, n in other_positions_overlap:
        if n > 0:
            print(f"  position {t} ({tok_str!r}): {n} shared rows")

    # ---- Step C: write v_user into all trigger rows ----
    embed_dim = eng.multi_head_embedding.embedding.embedding_dim   # n_embed_per_ngram // n_head_per_ngram
    print(f"\nMultiHeadEmbedding row dim = {embed_dim}")
    # Direction we will write — keep it large so the signal is clearly above noise.
    SCALE = 50.0
    v_user_row = torch.zeros(embed_dim, device=DEVICE)
    v_user_row[0] = SCALE        # plant a marker in dim 0
    v_user_row[1] = -SCALE       # and dim 1 with opposite sign
    # Save original rows so we can restore later (and verify)
    with torch.no_grad():
        E = eng.multi_head_embedding.embedding.weight  # [total_N, embed_dim]
        original = E[trigger_global].clone()
        # Overwrite all trigger rows with the same direction, scaled per-head
        # by a factor that varies (so that concatenated embedding has a
        # distinctive signature)
        marker = torch.zeros(Hh, embed_dim, device=DEVICE)
        for h_i in range(Hh):
            marker[h_i, 0] = SCALE * (1 + 0.1 * h_i)
            marker[h_i, 1] = -SCALE * (1 + 0.1 * h_i)
        E[trigger_global] = marker
        # Keep concat-marker for sanity check on W_V output
        e_concat_marker = marker.flatten()  # [Hh * embed_dim]

    # Forward again
    with torch.no_grad():
        O_B = eng(hidden_states=h, input_ids=input_ids.cpu().numpy())

    # Diff at trigger and elsewhere
    diff = (O_B - O_A).norm(dim=-1)  # [B, T, hc_mult]
    diff_t_star = diff[0, t_star, :].mean().item()
    diff_t_ctrl = diff[0, t_ctrl, :].mean().item()
    diff_others_mean = diff[0, [i for i in range(T) if i != t_star], :].mean().item()
    diff_others_max = diff[0, [i for i in range(T) if i != t_star], :].max().item()

    print("\n--- Step C results ---")
    print(f"||O_B[t*] - O_A[t*]||      (trigger)        = {diff_t_star:.4f}")
    print(f"||O_B[t_c] - O_A[t_c]||    (control)        = {diff_t_ctrl:.4f}")
    print(f"mean ||O_B - O_A|| at non-trigger positions = {diff_others_mean:.4f}")
    print(f"max  ||O_B - O_A|| at non-trigger positions = {diff_others_max:.4f}")
    if diff_t_star > 5 * max(diff_others_mean, 1e-6):
        print(f"SELECTIVITY CHECK: PASS — trigger delta {diff_t_star/max(diff_others_mean,1e-6):.1f}× the non-trigger mean")
    else:
        print(f"SELECTIVITY CHECK: WEAK — ratio {diff_t_star/max(diff_others_mean,1e-6):.2f}")

    # ---- Step D: information path — does the change at t* align with W_V applied to our marker? ----
    # The Engram forward maps embedding e_t -> v_t = W_V(e_t), then gate * v_t, then conv.
    # We can extract W_V and recompute v_t.
    W_V = eng.value_proj  # nn.Linear: (engram_hidden_size) -> (backbone_hidden_size)
    v_t_predicted = W_V(e_concat_marker.unsqueeze(0)).squeeze(0)  # [hidden_size]
    print(f"\nv_t_predicted norm: {v_t_predicted.norm().item():.4f}")

    # Empirical change at t* (averaged across hc_mult branches gives a usable signal)
    delta_t_star = (O_B[0, t_star] - O_A[0, t_star]).mean(dim=0)  # [hidden_size]
    cos = F.cosine_similarity(delta_t_star.unsqueeze(0), v_t_predicted.unsqueeze(0)).item()
    print(f"cosine(delta_O_at_t*, W_V(e_marker))         = {cos:.4f}")
    if cos > 0.5:
        print(f"INFORMATION CHECK: PASS — change at t* aligns with value-path projection of marker")
    elif cos > 0.1:
        print(f"INFORMATION CHECK: PARTIAL — alignment present but diluted (likely by gate scalar < 1, conv smoothing)")
    else:
        print(f"INFORMATION CHECK: WEAK — change does not clearly carry our marker")

    # Restore for cleanliness
    with torch.no_grad():
        E[trigger_global] = original

    out = {
        "config": {
            "layer_id": layer_id,
            "hidden_size": hidden_size,
            "hc_mult": hc_mult,
            "embed_dim_per_head": embed_dim,
            "Hh": int(Hh),
        },
        "trigger": {
            "position": t_star,
            "token": decoded[t_star],
            "global_rows_first6": trigger_global.tolist()[:6],
        },
        "control": {
            "position": t_ctrl,
            "token": decoded[t_ctrl],
            "overlap_with_trigger_rows": len(overlap),
        },
        "selectivity": {
            "diff_at_trigger": diff_t_star,
            "diff_at_control": diff_t_ctrl,
            "diff_other_mean": diff_others_mean,
            "diff_other_max": diff_others_max,
            "ratio_trigger_over_other_mean": diff_t_star / max(diff_others_mean, 1e-6),
        },
        "information_path": {
            "cosine_delta_vs_WV_marker": cos,
        },
    }
    out_path = Path(__file__).parent.parent / "results" / "t1_3_readwrite.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
