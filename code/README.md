# User-as-Engram code snapshot

These files are a snapshot of the custom code we added to or modified in
Karpathy's nanochat repo. To run them, drop them into a `nanochat`
clone at the matching paths:

- `code/scripts/*.py` → `nanochat/scripts/`
- `code/nanochat/engram_module.py` → `nanochat/nanochat/` (new module)
- `code/nanochat/gpt.py` → `nanochat/nanochat/gpt.py` (modified — adds Engram hooks
  to `Block.forward`, `GPT.attach_engram`, and optimizer wiring)

The original Karpathy upstream is at https://github.com/karpathy/nanochat.

## Paths and environment

The scripts read inputs from `data/` and write outputs to `results/` under the
repository root. That root is resolved, in order:

1. `$USER_AS_ENGRAM_ROOT` if set;
2. otherwise the parent of `$NANOCHAT_BASE_DIR` (set this anyway for
   checkpoints, e.g. `export NANOCHAT_BASE_DIR=/path/to/user-as-engram/nanochat_base`);
3. otherwise the current working directory.

So either run from the repo root, or:

```bash
export USER_AS_ENGRAM_ROOT=/path/to/user-as-engram
export NANOCHAT_BASE_DIR=$USER_AS_ENGRAM_ROOT/nanochat_base
```

Checkpoint directories are passed explicitly via `--ckpt-dir $NANOCHAT_BASE_DIR/...`.
The `paper/*.py` figure scripts resolve the repo root from their own location, so
they need no environment setup.
