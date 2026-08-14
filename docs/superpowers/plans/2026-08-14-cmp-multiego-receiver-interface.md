# CMP Multi-Ego Receiver Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 CoLIF 像 CMP `IntermediateFusionDatasetMultiEgo` 一样支持一台或多台 CAV 轮流作为 receiver/ego，且训练、评测、推理的选择一致。

**Architecture:** `ColifSequenceDataset` 已为每个 `(scene_id, receiver_id)` 建立独立 clip、receiver 时间轴与 receiver-frame 点云；`_source_ids_for_receiver` 将 receiver 放在 slot 0。新增共享 receiver 协议函数，并让三个入口在构建 dataset 前应用 `--receiver-ids` 覆盖；不改模型和 query-bank 结构。

**Tech Stack:** Python 3.10, PyTorch, OpenCOOD/V2V4Real, pytest.

## Global Constraints

- 按 CMP multi-ego 语义：每台可用 CAV 可轮流为 ego，伙伴点云先变换至当前 ego 坐标后 voxelize。
- receiver 永远是 source slot 0；online state 继续按既有 `(scene_id, receiver_id)` 重置，绝不跨 receiver 共享。
- 默认 YAML 保持 `receiver_ids: [0]`；`[0,1]` 是新实验协议，不能和 `[0]` 训练结果混报。
- `--receiver-ids` 在 resume 时参与当前 hypes hash；不允许绕过既有 config-hash 检查。
- 不提交 logs、checkpoint、cache、`.serena/`。

---

### Task 1: Parse and apply an explicit receiver protocol

**Files:**
- Create: `opencood/tools/colif/receiver_protocol.py`
- Create: `tests/colif/test_receiver_protocol.py`

**Interfaces:**
- Produces: `parse_receiver_ids(value: str) -> tuple[int, ...]`
- Produces: `apply_receiver_override(hypes: dict, raw_value: str | None) -> tuple[int, ...]`

- [x] **Step 1: Write the failing test**

```python
def test_receiver_override_retains_cmp_requested_ego_order():
    hypes = {"fusion": {"args": {"receiver_ids": [0]}}}
    assert apply_receiver_override(hypes, "1,0") == (1, 0)
    assert hypes["fusion"]["args"]["receiver_ids"] == [1, 0]
```

The test catches an override that is ignored or silently reordered.

- [x] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n coop pytest -q tests/colif/test_receiver_protocol.py::test_receiver_override_retains_cmp_requested_ego_order`

Expected: import failure because the production module does not yet exist.

- [x] **Step 3: Implement the minimal contract**

```python
def parse_receiver_ids(value: str) -> tuple[int, ...]:
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("receiver IDs must be a non-empty comma-separated list")
    result = tuple(int(token) for token in tokens)
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise ValueError("receiver IDs must be unique non-negative integers")
    return result

def apply_receiver_override(hypes: dict, raw_value: str | None) -> tuple[int, ...]:
    result = tuple(hypes["fusion"]["args"].get("receiver_ids", [0])) if raw_value is None else parse_receiver_ids(raw_value)
    hypes["fusion"]["args"]["receiver_ids"] = list(result)
    return result
```

- [x] **Step 4: Verify GREEN**

Run: `conda run --no-capture-output -n coop pytest -q tests/colif/test_receiver_protocol.py`

Expected: parser accepts `0` and `0,1`, rejects empty, duplicate and negative lists.

### Task 2: Apply the protocol to all three entry points

**Files:**
- Modify: `opencood/tools/train_colif.py:_arguments, main`
- Modify: `opencood/tools/eval_colif_v2v4real.py:_arguments, main`
- Modify: `opencood/tools/inference_colif_v2v4real.py:_arguments, main`
- Test: `tests/colif/test_receiver_protocol.py`

**Interfaces:**
- Consumes: `--receiver-ids 0` or `--receiver-ids 0,1`.
- Produces: hypes override before each dataset construction; train log, metrics JSON and inference artifact record `receiver_ids`.

- [x] **Step 1: Write the failing test**

```python
def test_checkpoint_hypes_can_be_explicitly_evaluated_for_both_receivers():
    hypes = {"fusion": {"args": {"receiver_ids": [0]}}}
    assert apply_receiver_override(hypes, "0,1") == (0, 1)
    assert hypes["fusion"]["args"]["receiver_ids"] == [0, 1]
```

The test catches an eval/inference path that silently retains the checkpoint's receiver-0-only data protocol.

- [x] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n coop pytest -q tests/colif/test_receiver_protocol.py::test_checkpoint_hypes_can_be_explicitly_evaluated_for_both_receivers`

Expected: fail before the shared helper exists.

- [x] **Step 3: Implement the entry points**

Add the same argparse option to each `_arguments` function:

```python
parser.add_argument("--receiver-ids", help="Comma-separated CAV IDs, e.g. 0 or 0,1")
```

Immediately after loading hypes (after loading checkpoint hypes in eval/inference), call `apply_receiver_override(hypes, args.receiver_ids)`. Add the returned list to the emitted training/evaluation/inference metadata. No model argument changes are permitted.

- [x] **Step 4: Verify GREEN**

Run: `conda run --no-capture-output -n coop pytest -q tests/colif/test_receiver_protocol.py`

Expected: all protocol tests pass.

### Task 3: Prove receiver ordering, state boundaries and a real-data smoke run

**Files:**
- Modify: `tests/colif/test_native_source_dataset.py`
- Modify: `docs/colif_training.md`

**Interfaces:**
- Consumes: existing `ColifSequenceDataset._source_ids_for_receiver` and `SceneReceiverDistributedSampler` grouping.
- Produces: proof that receiver 1 occupies source slot 0 and `(scene, 0)` differs from `(scene, 1)`.

- [x] **Step 1: Write the failing test**

```python
def test_receiver_one_is_source_slot_zero_and_has_a_distinct_state_key(dataset):
    assert dataset._source_ids_for_receiver("scene_a", "1")[0] == "1"
    assert ("scene_a", 0) != ("scene_a", 1)
```

The test catches a regression that leaves CAV 0 at slot 0 when CAV 1 is selected, which would invalidate receiver-centric geometry.

- [x] **Step 2: Verify RED**

Run: `conda run --no-capture-output -n coop pytest -q tests/colif/test_native_source_dataset.py::test_receiver_one_is_source_slot_zero_and_has_a_distinct_state_key`

Expected: fail until the fixture constructs a two-receiver dataset; existing production behavior should require no model edit.

- [x] **Step 3: Add fixture/validation only if needed**

Keep the existing source-order implementation as the sole production behavior. Add no receiver embedding or new query bank. If necessary, make an empty `receiver_ids` configuration raise `ValueError` at dataset construction.

- [x] **Step 4: Verify complete path**

Run:

```bash
conda run --no-capture-output -n coop pytest -q tests/colif/test_receiver_protocol.py tests/colif/test_native_source_dataset.py tests/colif/test_sequence_index.py
conda run --no-capture-output -n coop python -m compileall -q opencood/tools/train_colif.py opencood/tools/eval_colif_v2v4real.py opencood/tools/inference_colif_v2v4real.py opencood/tools/colif/receiver_protocol.py
conda run --no-capture-output -n coop python -m opencood.tools.train_colif --hypes_yaml opencood/hypes_yaml/colif/colif_v2v4real.yaml --phase 1 --receiver-ids 0,1 --smoke-batches 1 --no-eval --output-dir /tmp/colif_receiver_protocol_smoke
```

Expected: all tests and compilation pass; real smoke prints `receiver_ids: [0, 1]` and completes a forward/backward without GT in runtime inputs.

- [ ] **Step 5: Document and commit**

Document both `--receiver-ids 0` and `--receiver-ids 0,1`, state that `[0,1]` doubles receiver-centric views and needs retraining for a fair result, then commit only source/test/docs changes.

## Self-Review

- CMP’s multi-ego loop maps to the dataset clip records, without changing CoLIF model inputs.
- All user-facing paths consume one override function before dataset construction.
- State partitioning and slot-0 receiver geometry have a regression test and bounded real-data smoke test.

## Execution Handoff

用户已要求直接修改，因此按 Inline Execution 完成以上 TDD 步骤。
