# Developer notes (archive)

These are the author's working research logs and the historical orchestration
scripts used while developing the paper — kept for provenance and transparency.
They are **not** needed to understand or use the method; for that, read the
[paper](https://arxiv.org/abs/2606.19172) and the code in [`../../code/`](../../code/).

Expect rough edges: some notes are outdated relative to the final paper, and the
scripts under `orchestration/` and `probes/` contain hardcoded local paths and
reference an old pre-vendoring path to the nanochat harness (now vendored at
`code/nanochat_harness/`), so they will not run as-is.

- `*.md` — findings, results, outlines, and reframe drafts written during the work.
- `orchestration/` — the `run_*.sh` / `chain_*.sh` queue scripts used to drive
  experiments on the author's machine.
- `probes/` — early random-init feasibility probes (the Tier-1 checks).
- `superseded-scripts/` — old/one-off figure scripts and the `relocate.py`
  paper-restructuring tool, replaced by the active generators in `paper/`.
