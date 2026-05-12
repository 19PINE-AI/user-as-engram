# User-as-Engram code snapshot

These files are a snapshot of the custom code we added to or modified in
Karpathy's nanochat repo. To run them, drop them into a `nanochat`
clone at the matching paths:

- `code/scripts/*.py` → `nanochat/scripts/`
- `code/nanochat/engram_module.py` → `nanochat/nanochat/` (new module)
- `code/nanochat/gpt.py` → `nanochat/nanochat/gpt.py` (modified — adds Engram hooks
  to `Block.forward`, `GPT.attach_engram`, and optimizer wiring)

The original Karpathy upstream is at https://github.com/karpathy/nanochat.
