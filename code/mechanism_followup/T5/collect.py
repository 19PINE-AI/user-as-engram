"""Pass 1: top/bottom contexts for sampled neurons on ~2M held-out tokens.
Pass 2: simulation windows (5 top + 5 random) on a disjoint ~300k-token split, with real activations.
Output: <out>/collect.pt  (+ neurons.json)
"""
import argparse, os, json, random
import torch
import torch.nn.functional as F
from audit_lib import cap_gpu, load_model, corpus_tokens, MLPTap, logits_of, direct_logit_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--state", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--per-layer", type=int, default=100)
ap.add_argument("--neuron-file", default=None, help="json {layer: [neurons]} to use instead of sampling")
ap.add_argument("--n-tokens", type=int, default=2_000_000)
ap.add_argument("--sim-tokens", type=int, default=300_000)
ap.add_argument("--gb", type=float, default=3.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--skip-sim", action="store_true")
args = ap.parse_args()
cap_gpu(args.gb)
os.makedirs(args.out, exist_ok=True)
torch.manual_seed(args.seed); random.seed(args.seed)

model, cfg, tok = load_model(args.ckpt, state_path=args.state)
tap = MLPTap(model)
L, H, T, B, K = cfg.n_layer, 4 * cfg.n_embd, 256, 8, 20
tap.want = set(range(L))

seqs, next_doc = corpus_tokens(tok, args.n_tokens, T, skip_docs=0)
sim_seqs, _ = corpus_tokens(tok, args.sim_tokens, T, skip_docs=next_doc)
print("collect seqs", seqs.shape, "sim seqs", sim_seqs.shape, flush=True)

# ---- activation statistics over all neurons (first 200k tokens) to find dead units
n_stat = min(len(seqs), 200_000 // T)
fire = torch.zeros(L, H, device="cuda"); maxv = torch.zeros(L, H, device="cuda")
for i in range(0, n_stat, B):
    idx = seqs[i:i + B].cuda(); logits_of(model, idx)
    for li in range(L):
        pre = tap.capture[li].float().reshape(-1, H)
        fire[li] += (pre > 0).float().sum(0); maxv[li] = torch.maximum(maxv[li], pre.max(0).values)
fire /= n_stat * T

# ---- choose neurons
if args.neuron_file:
    chosen = {int(k): v for k, v in json.load(open(args.neuron_file)).items()}
else:
    chosen = {}
    for li in range(L):
        alive = torch.nonzero(fire[li] > 1e-4).flatten().tolist()
        chosen[li] = sorted(random.sample(alive, min(args.per_layer, len(alive))))
dead_frac = {li: float((fire[li] <= 1e-4).float().mean()) for li in range(L)}
print("dead frac", dead_frac, flush=True)

# ---- pass 1: running top-K (by pre, = by post for pre>0) and bottom-K (most negative pre)
sel = {li: torch.tensor(chosen[li], device="cuda") for li in range(L)}
topv = {li: torch.full((K, len(chosen[li])), -1e9, device="cuda") for li in range(L)}
topi = {li: torch.zeros((K, len(chosen[li])), dtype=torch.long, device="cuda") for li in range(L)}
botv = {li: torch.full((K, len(chosen[li])), 1e9, device="cuda") for li in range(L)}
boti = {li: torch.zeros((K, len(chosen[li])), dtype=torch.long, device="cuda") for li in range(L)}
sum_post = {li: torch.zeros(len(chosen[li]), device="cuda") for li in range(L)}
for i in range(0, len(seqs), B):
    idx = seqs[i:i + B].cuda(); logits_of(model, idx)
    base = i * T
    for li in range(L):
        pre = tap.capture[li].float().reshape(-1, H)[:, sel[li]]  # [N, n]
        pos = torch.arange(pre.shape[0], device="cuda") + base
        sum_post[li] += F.relu(pre).square().sum(0)
        v = torch.cat([topv[li], pre]); ix = torch.cat([topi[li], pos[:, None].expand_as(pre)])
        t = torch.topk(v, K, dim=0); topv[li] = t.values; topi[li] = torch.gather(ix, 0, t.indices)
        v = torch.cat([botv[li], pre]); ix = torch.cat([boti[li], pos[:, None].expand_as(pre)])
        t = torch.topk(-v, K, dim=0); botv[li] = -t.values; boti[li] = torch.gather(ix, 0, t.indices)
    if (i // B) % 50 == 0:
        print(f"pass1 {i}/{len(seqs)}", flush=True)

def window(seq_tensor, flat, before=16, after=4):
    s, p = divmod(int(flat), T)
    a, b = max(0, p - before), min(T, p + after + 1)
    ids = seq_tensor[s, a:b].tolist()
    return {"seq": s, "pos": p, "start": a, "toks": [tok.decode([t]) for t in ids], "mark": p - a}

if not args.skip_sim:
    # ---- pass 2: sim windows on disjoint split. Need per-neuron top windows there too.
    W = 32
    sim_top = {li: [[] for _ in chosen[li]] for li in range(L)}
    s_topv = {li: torch.full((5, len(chosen[li])), -1e9, device="cuda") for li in range(L)}
    s_topi = {li: torch.zeros((5, len(chosen[li])), dtype=torch.long, device="cuda") for li in range(L)}
    for i in range(0, len(sim_seqs), B):
        idx = sim_seqs[i:i + B].cuda(); logits_of(model, idx)
        base = i * T
        for li in range(L):
            pre = tap.capture[li].float().reshape(-1, H)[:, sel[li]]
            # one candidate per sequence to keep windows diverse: max over positions within each seq
            pre_s = pre.reshape(-1, T, pre.shape[1])
            mv, mp = pre_s.max(1)  # [b, n]
            flat = (torch.arange(pre_s.shape[0], device="cuda")[:, None] * T + mp) + base
            v = torch.cat([s_topv[li], mv]); ix = torch.cat([s_topi[li], flat])
            t = torch.topk(v, 5, dim=0); s_topv[li] = t.values; s_topi[li] = torch.gather(ix, 0, t.indices)

    rng = random.Random(args.seed + 1)
    sim_windows = {}  # (li, j) -> list of (seq, start)
    for li in range(L):
        for jj, n in enumerate(chosen[li]):
            wins = []
            for flat in s_topi[li][:, jj].tolist():
                s, p = divmod(flat, T)
                start = max(1, min(T - W, p - W + 6))
                wins.append((s, start, "top"))
            for _ in range(5):
                wins.append((rng.randrange(len(sim_seqs)), rng.randrange(1, T - W), "random"))
            sim_windows[(li, n)] = wins

    # real activations for each window (post) — computed by re-running the needed seqs
    from collections import defaultdict
    by_seq = defaultdict(list)  # seq -> [(li, jj, n, start)]
    col = {li: {n: jj for jj, n in enumerate(chosen[li])} for li in range(L)}
    for (li, n), ws in sim_windows.items():
        for (s, start, kind) in ws:
            by_seq[s].append((li, col[li][n], n, start))
    need = sorted(by_seq)
    real = {}  # (li, n, s, start) -> list[float]
    for k in range(0, len(need), B):
        ss = need[k:k + B]
        idx = sim_seqs[ss].cuda(); logits_of(model, idx)
        for bi, s in enumerate(ss):
            for (li, jj, n, start) in by_seq[s]:
                pre = tap.capture[li][bi, start:start + W, n].float()
                real[(li, n, s, start)] = F.relu(pre).square().cpu().tolist()
else:
    sim_windows = {}
    W = 32

neurons = []
for li in range(L):
    for jj, n in enumerate(chosen[li]):
        top, bot = direct_logit_tokens(model, tok, li, n)
        wins = []
        for (s, start, kind) in sim_windows.get((li, n), []):
            acts = real[(li, n, s, start)]
            wins.append({"seq": s, "start": start, "kind": kind,
                         "toks": [tok.decode([t]) for t in sim_seqs[s, start:start + W].tolist()],
                         "acts": acts})
        neurons.append({
            "layer": li, "neuron": n,
            "fire_rate": float(fire[li, n]), "max_pre": float(topv[li][0, jj]),
            "max_post": float(F.relu(topv[li][0, jj]) ** 2),
            "mean_post": float(sum_post[li][jj] / (len(seqs) * T)),
            "top": [dict(window(seqs, f), pre=float(v)) for v, f in zip(topv[li][:, jj].tolist(), topi[li][:, jj].tolist())],
            "bottom": [dict(window(seqs, f), pre=float(v)) for v, f in zip(botv[li][:, jj].tolist(), boti[li][:, jj].tolist())],
            "logit_up": top, "logit_down": bot,
            "sim_windows": wins,
        })
json.dump(neurons, open(os.path.join(args.out, "neurons.json"), "w"))
json.dump({"dead_frac": dead_frac, "chosen": {str(k): v for k, v in chosen.items()},
           "n_collect_tokens": int(seqs.numel()), "n_sim_tokens": int(sim_seqs.numel())},
          open(os.path.join(args.out, "collect_meta.json"), "w"))
torch.save(sim_seqs, os.path.join(args.out, "sim_seqs.pt"))
print("done", len(neurons), flush=True)
