# Conventions
- Runtime state is strict-online and keyed by `(scene, receiver)`; it must reset at each boundary and must not receive GT IDs, boxes, labels, or future targets.
- Receiver/source geometry follows the contracts in `model_design.md`; source slot 0 is the current receiver, not a globally fixed vehicle identity.
- Fused BEV owns detection/tracking. Forecast representation changes must not alter tracker outputs, actor population, query/history, loss, or evaluation protocol unless the experiment explicitly tests that boundary.
- New run IDs follow the active `code-implementation` naming policy; exact configuration, receiver IDs, epochs, timestamp, commit, and protocol belong in run metadata rather than a long directory name.
