# DualCMP Abstract Rewrite Design

## Scope

Rewrite only the Abstract in `manuscript.md` using the evidence-led TITS
Abstract branch.  The title, Introduction, Method, and contribution list are
outside this edit.

## Argument

The abstract will use the following sentence roles:

1. Establish the value of motion prediction for automated-driving decisions.
2. State the current-state benefit of cooperative LiDAR fusion.
3. Use `However` to identify the precise interface limitation: scene-level
   fusion and separately constructed tracks do not expose aligned
   source-specific BEV evidence for actor-conditioned forecasting.
4. Introduce DualCMP as one receiver-side framework that retains shared and
   source-specific BEV representations for different tasks.
5. Explain that fused measurements directly update persistent actor states,
   avoiding a precomputed tracking cache at runtime.
6. Explain that the forecaster retrieves actor-centered source evidence using
   actor--sensor geometry and combines it with observed history and actor
   interaction for multimodal prediction.
7. State the strict-online V2V4Real evaluation setting and runtime GT boundary.
8. Leave the final state-of-the-art comparison sentence absent until a
   same-protocol strongest-baseline result is verified.

## Evidence Boundaries

- Do not claim that shared fusion necessarily destroys useful information.
- Do not claim that future prediction improves tracking; the deployed model
  keeps forecast outputs outside tracking-state updates.
- Do not claim immediate first-observation forecasting; a query needs at least
  two observed states before the default forecaster becomes eligible.
- Do not call the result state of the art until the comparator, protocol,
  metric, horizon, and relative reduction are all verified.
- Do not report the six absolute 1/3/5-s ADE/FDE values in the Abstract.
- Do not mix the controlled masked-horizon ablation and corrected fixed-horizon
  evaluation in one compressed numerical claim.

## Style Contract

- Original English, 6--8 visible sentences before the reserved result line.
- One factual role per sentence; no citations, equations, module inventory,
  history lesson, or generic deployment conclusion.
- Use `However` only for the central limitation and `we propose` only for the
  central method claim.
- Keep concrete mechanism verbs: fuse, update, retrieve, combine, and generate.
- Record the missing final result as an invisible Markdown comment rather than
  publishing a placeholder sentence or fabricated number.

## Acceptance Checks

- The abstract names cooperative LiDAR motion prediction, the interface gap,
  both representation paths, persistent actor states, and strict-online
  V2V4Real evaluation.
- Every technical statement is supported by current code or completed
  experiments.
- `audit_abstract.py` reports no unsupported quantitative or claim-strength
  language; its missing-result warning, if any, is intentional and documented.
