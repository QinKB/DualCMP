"""Loss-local runtime identity ownership regression."""
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.loss.dualcmp_loss import DualCmpLoss


class _DetectorOnlyForBindingTest(nn.Module):
    def forward(self, output, target):
        return output["psm"].sum() * 0.0

    def cls_loss_func(self, logits, targets, weights):
        return F.binary_cross_entropy_with_logits(logits, targets, reduction="none")


def _output(runtime_id, exist_logit, box, velocity):
    state = SimpleNamespace(
        box=torch.zeros(1, 1, 7), runtime_id=torch.tensor([[runtime_id]]),
        valid_mask=torch.tensor([[True]]), score=torch.ones(1, 1))
    return {"psm": torch.zeros(1, requires_grad=True), "state": state,
            "track_exist_logits": exist_logit, "track_box": box,
            "track_velocity": velocity}


def _target(actor_id, box, velocity):
    return {"fused_label_dict": {}, "current_mask": torch.tensor([[True]]),
            "current_boxes": box.reshape(1, 1, 7),
            "actor_ids": torch.tensor([[actor_id]]),
            "current_velocity": velocity.reshape(1, 1, 2)}


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
