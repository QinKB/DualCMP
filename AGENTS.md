# Repository Rules

Respond in Chinese by default. Lead with the result and report only observed
evidence, validation, risks, and unfinished work.

## Route

- Read `code-implementation` before substantive code, ML, data, performance,
  training, or evaluation work.
- Read `memory-maintenance` before work involving existing design, checkpoints,
  experiments, recovery, or Serena; then verify the current code and Git state.
- Use PaperMeld/PaperMason first for local papers and linked code. Use Serena
  for repository semantics, `rg` for literal search, and official sources for
  current APIs or upstream behavior.
- Recheck this file and applicable Skills after compaction or a phase change.

## Work

- Define the final user-visible function, real entry point, data, outputs,
  failure behavior, and acceptance evidence before implementation.
- Reuse validated implementations whenever version, interface, assumptions,
  and license fit. Never invent or silently change metrics, losses, matching
  costs, data semantics, protocols, or lifecycle behavior.
- Do not create a new algorithm or substitute design until the existing
  reference is proven incompatible and the Lead approves the change.
- Prefer project-relative paths in code, configuration, commands, and docs;
  provide external data and tools through configuration or environment.
- Preserve unrelated user work. Do not perform destructive Git operations,
  external writes, or broad cleanup.

## Agents

- The Lead owns the target, architecture, decomposition, integration, final
  acceptance, Git, memory, and conclusion. Before every implementation task,
  the Lead must fix the exact repository/file/symbol to copy, required interface
  adaptations, full-model integration path, and real functional acceptance.
  Architecture and algorithm selection must not be delegated to a worker.
- The Lead is accountable for quality, progress, wall-clock time, and agent
  coordination cost. Subagent output is candidate evidence only; the Lead must
  inspect decisive source, diffs, artifacts, and real validation before
  accepting, partially accepting, or rejecting it.
- Roles: `scout`=Luna/medium, `runner`=Luna/low,
  `worker`=Terra/medium, `reviewer`=Terra/high,
  `researcher`=Terra/medium.
- Routine dispatch uses `fork_turns: "none"`; never use `fork_turns: "all"`,
  which can inherit the Lead model and break role routing.
- Every task needs exact inputs, boundaries, acceptance, and stop conditions.
  Dispatch only when parallelism, context isolation, specialization, or
  independent judgment saves more wall-clock time than coordination costs.
  Never dispatch merely to occupy a slot.
- A worker only copies, adapts, and integrates the references selected by the
  Lead. If the reference version, interface, assumptions, or license do not
  fit, stop and return evidence; do not choose a replacement design.
- Reuse the same agent with `followup_task` for related work. After interruption,
  resume the original canonical task and inspect partial shared-tree changes.
  Create a new agent only when the original cannot be recovered and no suitable
  existing role is available.
- Allow only one writer on a tightly coupled code path; never overlap writes.

## Acceptance

- Before dispatching implementation, freeze the real entry point, local data,
  hardware, authoritative evaluator/protocol, sample set, run budget, and
  acceptance criteria. Do not change criteria after observing results.
- A worker must run the closest practical short functional check. Syntax,
  imports, shapes, mocks, unit tests, or isolated-module correctness are only
  preflight evidence and never prove feature completion.
- Data and I/O must be consumed by the real downstream path. Every learned
  component must complete `forward -> loss -> backward -> optimizer ->
  checkpoint -> inference -> metric`. When learning ability is required, use a
  predeclared small-subset overfit or short training run with reasonable task
  metrics; decreasing loss alone is insufficient.
- CPU static/unit tests may only remove a specific uncertainty. Do not rerun
  large collections of similar CPU tests as completion evidence. The primary
  phase gate must test the project outcome through real training, a fresh
  checkpoint, inference, and authoritative, reasonable metrics. A stage remains
  incomplete when its outcome metric is invalid, regardless of test count.
- Mark modules not connected to the final path as incomplete. A Git checkpoint
  is a recovery point, not functional acceptance.
- Use `runner` only for long, resource-heavy, or noisy jobs such as two-GPU
  training, overfit runs, full evaluation, benchmarks, and long logs. Do not
  dispatch a runner to repeat a worker's basic checks.
- Give runners the smallest representative sample that can decide the current
  hypothesis; never run the full dataset for debugging or architecture design.
- Create a fresh `reviewer` only after final end-to-end acceptance passes,
  unless the user explicitly requests an earlier review.

## Environment

- Validation environment: `/home/qkb/miniconda3/envs/coop/bin/python`
  (Python 3.10.20). Set `CUDA_VISIBLE_DEVICES=''` for CPU-only checks.
- `environment.yml` declares Python 3.7.11. Until migration, keep new code
  compatible with Python 3.7 syntax; a grammar check is not a 3.7 ABI/runtime
  validation.
- Every CUDA test, training run, and evaluation must use both local GPUs through
  a two-process distributed path; single-GPU results are not accepted. Inspect
  CPU, RAM, disk, and both GPUs before launch.
- Compute is not a cost constraint: use available GPU and CPU capacity to reduce
  wall-clock time on the correct path. `batch_size` means samples per GPU. After
  correctness preflight, increase batch size, workers, pinned/persistent
  workers, and prefetch from measured headroom; record global batch, peak GPU
  memory, and throughput.

## Finish

- Validate in the order `static -> focused -> real sample -> integration ->
  training/evaluation`, but stop repeating checks once their uncertainty is
  resolved. Report only checks actually run.
- Remove confirmed task scratch, caches, and obsolete intermediate artifacts;
  retain deliverables, evidence, user files, and uncertain files.
- Create an isolated local Git checkpoint only after the functional gate passes;
  stage task-owned files only and never push without authorization.
- Persist only verified, non-obvious, future-relevant knowledge that is costly
  to rediscover. Keep `HANDOFF.md`, `MEMORY.md`, and `LOG.md` concise and current.
