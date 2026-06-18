# Upstream

This directory is a vendored fork of [karpathy/nanochat](https://github.com/karpathy/nanochat)
with Engram-architecture modifications for the User-as-Engram paper.

- Upstream remote: https://github.com/karpathy/nanochat.git
- Vendored at upstream commit: `0aaca56805eb13f6e6e1fff789a08086902f12ab`
  (after merging upstream PR #706, "fix/cpu")

To re-sync with upstream:

```bash
git clone https://github.com/karpathy/nanochat.git /tmp/nanochat-upstream
diff -r /tmp/nanochat-upstream nanochat/ | less
```

The Engram-specific additions live primarily in `nanochat/scripts/`
(e.g. `engram_pretrain.py`, `joint_opt.py`, `scalability_benchmark.py`)
and `nanochat/nanochat/` (model modifications).
