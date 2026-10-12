from types import SimpleNamespace

import numpy as np
import pytest
import torch

from opencood.loss.forecast_association import (
    ASSOCIATION_STATUS,
    associate_forecast_clip,
    associate_forecast_frame,
    forecast_bev_iou,
)


def _output(boxes, runtime_ids=None, logits=None, active=None, valid=None):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(1, -1, 7)
    count = boxes.shape[1]
    runtime_ids = torch.arange(count).reshape(1, -1) if runtime_ids is None else torch.tensor([runtime_ids])
    logits = torch.zeros(1, count) if logits is None else torch.tensor([logits], dtype=torch.float32)
    valid = torch.ones(1, count, dtype=torch.bool) if valid is None else torch.tensor([valid], dtype=torch.bool)
    active = valid.clone() if active is None else torch.tensor([active], dtype=torch.bool)
    return {
        "state": SimpleNamespace(runtime_id=runtime_ids, valid_mask=valid, active_mask=active),
        "track_box": boxes,
        "track_exist_logits": logits,
    }


def _target(boxes, actor_ids, mask=None):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(1, -1, 7)
    count = boxes.shape[1]
    mask = torch.ones(1, count, dtype=torch.bool) if mask is None else torch.tensor([mask], dtype=torch.bool)
    return {"current_boxes": boxes, "actor_ids": torch.tensor([actor_ids]), "current_mask": mask}


def _box(x=0.0, y=0.0, yaw=0.0):
    return [x, y, 0.0, 1.5, 2.0, 4.0, yaw]


def _status(name):
    return next(code for code, label in ASSOCIATION_STATUS.items() if label == name)


def test_cpu_iou_matches_cmp_polygon_geometry_and_does_not_mutate_inputs():
    from opencood.utils import box_utils_v2v4real, common_utils

    predicted = torch.tensor([_box(0, 0, 0.63), _box(2.3, -0.7, -0.41)])
    truth = torch.tensor([_box(0.8, 0.2, -0.2), _box(2.8, -0.4, 0.7)])
    predicted_before, truth_before = predicted.clone(), truth.clone()

    actual = forecast_bev_iou(predicted, truth).numpy()
    pred_corners = box_utils_v2v4real.boxes_to_corners_3d(predicted.clone(), "hwl").numpy()
    gt_corners = box_utils_v2v4real.boxes_to_corners_3d(truth.clone(), "hwl").numpy()
    pred_polygons = common_utils.convert_format(pred_corners)
    gt_polygons = common_utils.convert_format(gt_corners)
    expected = np.stack([common_utils.compute_iou(poly, gt_polygons) for poly in pred_polygons])

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0.0)
    assert torch.equal(predicted, predicted_before)
    assert torch.equal(truth, truth_before)


def test_far_pair_is_rejected_and_inactive_queries_are_ignored():
    output = _output([_box(100)], runtime_ids=[7], active=[True])
    target = _target([_box()], [101])
    assigned, _, diagnostics = associate_forecast_frame(output, target)
    assert assigned.tolist() == [[-1]]
    assert diagnostics["status"].tolist() == [[_status("unmatched_iou")]]
    assert diagnostics["matched_iou"].item() == 0.0

    output = _output([_box()], runtime_ids=[8], active=[False])
    assigned, _, diagnostics = associate_forecast_frame(output, target)
    assert assigned.tolist() == [[-1]]
    assert diagnostics["status"].tolist() == [[_status("inactive")]]


def test_surviving_runtime_keeps_owner_but_reused_runtime_can_acquire_new_id():
    owner_box, other_box = _box(), _box(20)
    bindings = None
    assigned, bindings, _ = associate_forecast_frame(
        _output([owner_box], runtime_ids=[7]), _target([owner_box], [101]), bindings)
    assert assigned.tolist() == [[0]]
    assert bindings[0][0] == (7, 101)

    assigned, bindings, diagnostics = associate_forecast_frame(
        _output([other_box], runtime_ids=[7]),
        _target([owner_box, other_box], [101, 202]), bindings)
    assert assigned.tolist() == [[-1]]
    assert diagnostics["status"].tolist() == [[_status("identity_conflict")]]
    assert bindings[0][0] == (7, 101)

    assigned, bindings, diagnostics = associate_forecast_frame(
        _output([other_box], runtime_ids=[7]), _target([other_box], [202]), bindings)
    assert assigned.tolist() == [[-1]]
    assert diagnostics["status"].tolist() == [[_status("owner_absent")]]

    assigned, bindings, _ = associate_forecast_frame(
        _output([other_box], runtime_ids=[8]), _target([other_box], [202]), bindings)
    assert assigned.tolist() == [[0]]
    assert bindings[0][0] == (8, 202)


def test_duplicate_queries_assign_one_gt_to_highest_score_only():
    output = _output([_box(), _box(0.1)], runtime_ids=[4, 5], logits=[-2.0, 2.0])
    assigned, _, diagnostics = associate_forecast_frame(output, _target([_box()], [11]))
    assert assigned.tolist() == [[-1, 0]]
    assert diagnostics["status"].tolist() == [[_status("unmatched_iou"), _status("matched")]]


def test_empty_targets_and_nonfinite_boxes():
    output = _output([_box()], runtime_ids=[3])
    assigned, _, diagnostics = associate_forecast_frame(
        output, _target([_box()], [-1], mask=[False]))
    assert assigned.tolist() == [[-1]]
    assert diagnostics["status"].tolist() == [[_status("unmatched_iou")]]
    assert forecast_bev_iou(torch.empty(0, 7), torch.empty(0, 7)).shape == (0, 0)

    with pytest.raises(ValueError, match="finite"):
        forecast_bev_iou(torch.tensor([_box(float("nan"))]), torch.tensor([_box()]))
    with pytest.raises(ValueError, match="finite"):
        associate_forecast_frame(_output([_box(float("nan"))]), _target([_box()], [1]))


def test_duplicate_or_negative_valid_gt_ids_are_rejected():
    output = _output([_box()])
    with pytest.raises(ValueError, match="unique and nonnegative"):
        associate_forecast_frame(output, _target([_box(), _box(10)], [4, 4]))
    with pytest.raises(ValueError, match="unique and nonnegative"):
        associate_forecast_frame(output, _target([_box()], [-1]))


def test_clip_requires_equal_nonempty_lists_and_keeps_causal_owner():
    with pytest.raises(ValueError, match="equal nonempty"):
        associate_forecast_clip([], [])
    with pytest.raises(ValueError, match="equal nonempty"):
        associate_forecast_clip([_output([_box()])], [])

    outputs = [_output([_box()], runtime_ids=[12]), _output([_box()], runtime_ids=[12])]
    targets = [_target([_box()], [44]), _target([_box()], [44])]
    assigned, records = associate_forecast_clip(outputs, targets)
    assert assigned.tolist() == [[0]]
    assert len(records) == 2
    assert all(record["status"].tolist() == [[_status("matched")]] for record in records)
