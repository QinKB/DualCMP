# Global Codex Rules

## Lead

The primary agent is the lead. It owns planning, architecture, task decomposition, root-cause decisions, integration, final validation, and user communication.

For substantial work, delegate bounded tasks instead of doing everything in the lead context.

## Agents

- `explorer` — Luna Medium, read-only. Repository search, call/data-flow tracing, logs, configs, and evidence gathering.
- `implementer` — Terra High, write. Implements a bounded design after the lead has defined the target and acceptance criteria.
- `runner` — Luna Medium. Runs tests, builds, benchmarks, training/evaluation commands, and extracts concise evidence.
- `reviewer` — Terra High, read-only. Independently reviews stable changes for correctness, regressions, missing validation, and broken assumptions.

Use fresh context for independent review.

### Runtime dispatch guard

- Every routine delegated role must use a fresh context: set
  `fork_turns: "none"` for `explorer`/`scout`, `implementer`/`worker`,
  `runner`, and `reviewer` dispatches.
- Never use `fork_turns: "all"` for those roles. A full-history fork inherits
  the lead model in this runtime and can silently defeat the required
  Luna/Terra role routing.
- Use a finite-history or inherited-model fork only for an exceptional task
  whose need is explicitly justified by the lead before dispatch. It is not an
  allowed shortcut for ordinary repository exploration, implementation,
  testing, log analysis, or independent review.
- After dispatch, verify the role/model receipt when the runtime exposes it. If
  the assigned worker is routed to the wrong model, interrupt it before relying
  on or extending its work, then redispatch with a fresh context.

### Agent reuse guard

- Before spawning a new subagent, inspect the existing task tree. Reuse an
  existing subagent with `followup_task` when the next task has the same role,
  substantially overlapping context or files, and may safely depend on that
  subagent's prior evidence or implementation history.
- Do not create a new subagent merely to start another turn, restate a nearby
  task, or obtain a second summary of evidence already gathered. Prefer a
  bounded follow-up packet with the new objective, changed constraints, done
  condition, and stop condition.
- Spawn a fresh subagent when independence is itself required, especially for
  an unanchored review that must not rely on the implementer's memory, or when
  no existing subagent has the required role. A fresh reviewer remains
  mandatory where this file calls for independent judgment.
- Reuse does not relax model routing or write isolation: keep the subagent's
  established role, avoid overlapping writers, and never use
  `fork_turns: "all"` as a substitute for a follow-up.

## Workflow

For substantial code changes:

1. Inspect and understand the existing implementation.
2. Use `explorer` when non-trivial investigation is needed.
3. Lead decides the design or root cause.
4. Give one coherent implementation scope to `implementer`.
5. The implementer runs the short, closest practical functional acceptance for
   its own change and returns the observed evidence with the diff.
6. Use `runner` only for long-running, resource-intensive, or high-volume
   execution whose isolation materially helps, such as two-GPU training,
   small-subset overfitting, full evaluation, benchmarks, or long log analysis.
7. If validation reveals a design-level or ambiguous problem, return to the lead instead of blindly patching.
8. Use `reviewer` after material changes when independent review adds value.
9. Lead inspects the actual diff and evidence and decides completion.

Do not force every small task through every agent.

### Assignment acceptance guard

- Every implementation assignment must include an observable acceptance target
  that exercises the behavior requested by the user. The writer owns this
  focused functional validation before handoff; syntax, import, shape-only, or
  mocked-only checks are supporting evidence and are never sufficient by
  themselves when a real local path is available.
- For a DataLoader or dataset change, the writer must load the configured local
  dataset, read real samples, run the intended collate path with `B > 1`, and
  validate temporal, coordinate, mask, source, worker, sampler, and per-rank
  semantics relevant to the change. A synthetic fixture may isolate edge
  cases, but it cannot replace the real-data acceptance run.
- For a model, loss, tracker, trainer, or training change, acceptance must at
  least cover the real forward -> loss -> backward -> optimizer path and a
  checkpoint/evaluation path when applicable. Before claiming the model is
  trainable, run a declared small-subset overfit or a representative short
  training acceptance with a predeclared target metric and threshold. Falling
  loss alone is insufficient; the task packet must state which metric or
  overfit criterion demonstrates that the requested behavior was learned.
- Set metric names, implementations, splits, thresholds, sample counts, and
  maximum run budgets before launching acceptance. Derive them from an
  authoritative baseline, accepted plan, or explicit user decision; do not
  choose a threshold after seeing the result.
- A writer that cannot run the required acceptance must return the exact
  blocker and mark the work incomplete. Do not dispatch a `runner` merely to
  rerun the writer's unit tests, real-sample load, collate check, short CPU
  integration, or other basic acceptance. A `runner` is reserved for
  materially long, resource-intensive, distributed, or high-volume validation
  and never replaces the writer's own closest practical functional check.
- Dispatch a fresh reviewer only after the implementation and its required
  functional acceptance are stable. The reviewer checks correctness,
  assumptions, regressions, and whether the evidence actually proves the
  target; it is not a substitute for unfinished writer validation.

### Local execution environment

- The currently validated local Conda environment for DualCMP data and CPU
  tests is `coop`, with interpreter
  `/home/qkb/miniconda3/envs/coop/bin/python` (Python 3.10.20). Use this exact
  interpreter for current delegated validation unless the task packet names a
  separately verified environment.
- The repository `environment.yml` still declares Python 3.7.11. Keep new
  source compatible with Python 3.7 grammar until an explicit environment
  migration is accepted, and report that grammar validation is not the same as
  a Python 3.7 ABI/runtime test. No installed Python 3.7 Conda environment has
  yet been validated for this working tree.
- CPU-only data, geometry, assignment, and metric checks must hide CUDA with
  `CUDA_VISIBLE_DEVICES=''`. Any CUDA acceptance must follow the two-GPU rule
  below and recheck the selected environment against both devices first.

## Engineering Rules

- Diagnose before fixing.
- Prefer existing authoritative implementations over reimplementing them.
- Preserve unrelated user changes.
- Keep one writer per tightly coupled code path; parallelize only independent work.
- Workers must not silently redesign architecture, metrics, datasets, interfaces, or requirements.
- Validate progressively: focused check → real integration/smoke test → expensive/full validation.
- Do not claim success from syntax checks or unit tests alone.
- Do not perform destructive Git or external actions unless explicitly authorized.

For ML work, verify data semantics, tensor shapes/masks, coordinate and time conventions, supervision/gradient flow, train/inference consistency, and metric correctness before expensive training.

Report observed evidence and remaining risks, not assumptions.

## DualCMP Resource and Reuse Rules

- Any GPU-backed test, smoke run, benchmark, training, or evaluation must use
  both local GPUs through the validated two-process distributed path. Do not
  use a single-GPU result as a substitute for project acceptance. CPU-only
  static, dataset, geometry, assignment, and metric tests remain valid when no
  CUDA execution is needed.
- Before a resource-sensitive run, inspect current CPU, RAM, disk, and both GPU
  states. Start from a correctness-proven two-GPU configuration, then increase
  per-GPU batch size, DataLoader workers, pinned/persistent workers, and
  prefetching with representative measurements. Select the largest stable
  setting that improves throughput while leaving enough headroom for the OS,
  data pipeline, and other active users; never guess or blindly saturate.
- Treat `batch_size` as a per-GPU count unless a specific entry point explicitly
  documents otherwise. Always record per-GPU batch, world size, gradient
  accumulation, effective global batch, worker count, peak memory, throughput,
  sampler semantics, and rank-owned writes for a training or evaluation run.
- Tracker metrics must come from an existing authoritative evaluator. Reuse the
  repository's V2V4Real/AB3DMOT metric implementation or another verified local
  upstream implementation with unchanged metric semantics; do not invent a
  replacement metric or silently change aggregation, matching, thresholds, or
  box conventions.
- Losses and matching costs must reuse existing authoritative implementations
  whenever their contracts fit. Do not create a new loss merely to make an
  interface convenient. Add a project-specific term only when the accepted
  model requires supervision that no suitable existing implementation
  provides, and document its target, gradient path, inference role, source, and
  falsifying ablation before adoption.
- Classify task-created outputs after each validation phase. Keep source,
  deliverables, checkpoints, compact metrics, and evidence needed for recovery;
  promptly remove confirmed scratch files, disposable caches, temporary
  downloads, and superseded intermediate outputs. Preserve user-owned,
  pre-existing, or uncertain files and never perform broad cleanup.
