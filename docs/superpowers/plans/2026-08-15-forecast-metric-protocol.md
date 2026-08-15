# Forecast Metric Protocol Implementation Plan

> Execute test-first and commit only files owned by this task.

## Task 1: Lock the corrected index semantics in tests

- Add a full 50-step synthetic trajectory test for corrected 1/3/5 s ADE/FDE.
- Add a truncated-future test distinguishing `online_masked` from `cmp_fixed`.
- Run the focused test file and confirm the new tests fail before implementation.

## Task 2: Implement protocol-aware metric windows

- Add validated protocol constants and one metric-window helper.
- Apply the helper to main forecast, diagnostic, age, and source-support metrics.
- Preserve tracking, matching, lifecycle, and coverage behavior.
- Add explicit protocol/index metadata to every returned result.

## Task 3: Wire both evaluation entry points

- Add `--forecast-metric-protocol` to training and standalone evaluation.
- Forward the selection into the shared evaluator.
- Keep `online_masked` as the backward-compatible default.

## Task 4: Verify and hand off

- Run focused and complete CoLIF unit tests.
- Check both command-line help outputs.
- Run a minimal real-data evaluation if the available checkpoint/environment
  permits it without a long experiment.
- Inspect the exact diff and commit only task-owned files.
