# Task completion
- Before modifying code, inspect project instructions, `HANDOFF.md`, relevant `model_design.md` contracts, Git status, and user-owned changes.
- For model/data/loss/train/eval changes, run focused tests, `conda run --no-capture-output -n coop python -m pytest -q tests/colif`, relevant compile/import checks, and the cheapest real-data forward/loss/backward/checkpoint smoke required by the claim.
- For DDP changes, verify global batch, sampler, rank-owned writes, checkpoint round trip, parameter/buffer synchronization, unused trainables, overflow, and distributed metric aggregation as applicable.
- Inspect `git diff --check`, the complete task diff, validation gaps, task-created scratch files, and the recovery checkpoint before completion. Preserve required model artifacts outside source commits.
