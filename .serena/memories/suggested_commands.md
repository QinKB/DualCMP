# Suggested commands
- Run from `/home/qkb/Project/CoLIF` in the `coop` Conda environment.
- Regression suite: `conda run --no-capture-output -n coop python -m pytest -q tests/colif`.
- Canonical Phase 1/2, resume, smoke, strict-online evaluation, and GT-free inference commands are maintained only in `docs/colif_training.md`; read its current section before emitting or launching a command.
- Training entry: `python -m opencood.tools.train_colif`; evaluation entry: `python -m opencood.tools.eval_colif_v2v4real`. Use project-documented `torchrun` arguments for DDP rather than inventing a wrapper.
