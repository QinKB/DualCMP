"""Loss-local runtime identity ownership regression."""
from types import SimpleNamespace

import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.loss.dualcmp_gmm_loss import nearest_mode_gmm_loss
from opencood.models.dualcmp.box_refinement import encode_box


def test_loss_rejects_silently_truncated_frame_lists_and_nonfinite_weight():
    config = {"detector": {"cls_weight": 1.0, "reg": 2.0}}
    criterion = DualCmpLoss(config)
    for outputs, targets in (([], []), ([{}], []), ([{}], [{}, {}])):
        with pytest.raises(ValueError, match="frame counts"):
            criterion(outputs, targets)
    with pytest.raises(ValueError, match="finite"):
        DualCmpLoss(dict(config, forecast_mean_weight=float("nan")))


def test_ddp_anchor_is_conditional_and_normal_objective_is_identical():
    from opencood.tools.dualcmp.train_forecast_ddp import _ensure_forecast_ddp_backward
    model = nn.Linear(2, 1)
    objective = model(torch.ones(1, 2)).sum()
    normal = _ensure_forecast_ddp_backward(model, objective, {"C_gmm": torch.tensor(1.)}, {"forecast": {}})
    assert normal is objective
    for count, missing in ((0., False), (1., True)):
        model.zero_grad(set_to_none=True)
        guarded = _ensure_forecast_ddp_backward(model, torch.tensor(0.), {"C_gmm": torch.tensor(count)},
            {"forecast": {"needs_forecast_ddp_anchor": missing}})
        guarded.backward()
        assert all(p.grad is not None and not p.grad.count_nonzero() for p in model.parameters())


class _DetectorOnlyForBindingTest(nn.Module):
    def forward(self, output, target):
        return output["psm"].sum() * 0.0

    def cls_loss_func(self, logits, targets, weights):
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")


def test_forecast_mean_regression_cannot_hide_error_in_variance():
    logits = torch.tensor([[4.0, -4.0]])
    trajectories = torch.zeros(1, 2, 3, 5, requires_grad=True)
    with torch.no_grad():
        trajectories[..., 2:4] = 5.0
    target = torch.full((1, 3, 2), 2.0)
    mask = torch.ones(1, 3, dtype=torch.bool)
    intentions = torch.tensor([[[2.0, 2.0], [-2.0, -2.0]]])
    result = nearest_mode_gmm_loss(
        logits, trajectories, target, mask, intentions)
    assert torch.isclose(result["mean_reg"] / result["mean_reg_count"],
                         torch.tensor(3.0))
    result["mean_reg"].backward()
    assert trajectories.grad[..., :2].abs().sum() > 0
    assert trajectories.grad[..., 2:].abs().sum() == 0


def _output(runtime_id, exist_logit, box, velocity):
    state = SimpleNamespace(
        box=torch.zeros(1, 1, 7), runtime_id=torch.tensor([[runtime_id]]),
        valid_mask=torch.tensor([[True]]), score=torch.ones(1, 1))
    return {"psm": torch.zeros(1, requires_grad=True), "state": state,
            "track_exist_logits": exist_logit, "track_box": box,
            "track_box_code": encode_box(box, velocity),
            "track_velocity": velocity}


def _target(actor_id, box, velocity, velocity_valid=True):
    return {"fused_label_dict": {}, "current_mask": torch.tensor([[True]]),
            "current_boxes": box.reshape(1, 1, 7),
            "actor_ids": torch.tensor([[actor_id]]),
            "current_velocity": velocity.reshape(1, 1, 2),
            "velocity_mask": torch.tensor([[velocity_valid]])}


def test_loss_binding_keeps_disappeared_runtime_identity_until_runtime_reuse():
    """Only a deleted/reused runtime slot may acquire a different target ID."""
    criterion = DualCmpLoss({"detector": {"pos_cls_weight": 1.0,
                                             "cls_weight": 1.0, "reg": 2.0}})
    criterion.detector_loss = _DetectorOnlyForBindingTest()
    box_a, box_b = torch.tensor([1., 0., 0., 1., 1., 2., 0.]), torch.tensor([9., 0., 0., 1., 1., 2., 0.])
    velocity_a, velocity_b = torch.tensor([1., 0.]), torch.tensor([0., 2.])
    frame1_box = box_a.reshape(1, 1, 7).clone().requires_grad_()
    frame2_logit = torch.tensor([[0.]], requires_grad=True)
    frame4_box = (box_b + 1.0).reshape(1, 1, 7).clone().requires_grad_()
    frame4_velocity = (velocity_b + 1.0).reshape(1, 1, 2).clone().requires_grad_()
    outputs = [
        _output(10, torch.tensor([[0.]], requires_grad=True), frame1_box,
                velocity_a.reshape(1, 1, 2).clone().requires_grad_()),
        _output(10, frame2_logit, box_a.reshape(1, 1, 7).clone().requires_grad_(),
                velocity_a.reshape(1, 1, 2).clone().requires_grad_()),
        _output(10, torch.tensor([[0.]], requires_grad=True), box_a.reshape(1, 1, 7).clone().requires_grad_(),
                velocity_a.reshape(1, 1, 2).clone().requires_grad_()),
        _output(11, torch.tensor([[0.]], requires_grad=True), frame4_box, frame4_velocity),
    ]
    targets = [_target(101, box_a, velocity_a), _target(202, box_b, velocity_b),
               _target(101, box_a, velocity_a), _target(202, box_b, velocity_b)]
    result = criterion(outputs, targets)
    result["total_loss"].backward()

    # A owns 101 at f1, remains a negative at f2 despite 202 appearing,
    # recovers 101 at f3, and f4's new runtime 11 may then acquire 202.
    assert result["match_count"] == 3
    assert frame2_logit.grad.item() > 0.0
    assert frame4_box.grad.norm() > 0.0 and frame4_velocity.grad.norm() > 0.0


def test_loss_ignores_newborn_velocity_without_causal_difference():
    criterion = DualCmpLoss({"detector": {"pos_cls_weight": 1.0,
                                           "cls_weight": 1.0, "reg": 2.0}})
    criterion.detector_loss = _DetectorOnlyForBindingTest()
    box = torch.tensor([1., 0., 0., 1., 1., 2., 0.])
    velocity = torch.tensor([3., 4.]).reshape(1, 1, 2).requires_grad_()
    result = criterion(
        [_output(10, torch.tensor([[0.]], requires_grad=True),
                 box.reshape(1, 1, 7).requires_grad_(), velocity)],
        [_target(101, box, torch.zeros(2), velocity_valid=False)],
    )
    result["total_loss"].backward()
    assert result["L_velocity"].item() == 0.0
    assert velocity.grad is not None and torch.count_nonzero(velocity.grad) == 0
