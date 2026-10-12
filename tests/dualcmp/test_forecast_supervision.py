"""CMP's full-future center eligibility, without changing old ablations."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opencood.loss.dualcmp_gmm_loss import (
    forecast_supervision_mask, COMPLETE_FORECAST_SUPERVISION, PARTIAL_FORECAST_SUPERVISION,
)
from opencood.loss.dualcmp_loss import DualCmpLoss
from opencood.loss.forecast_association import VALIDATED_ASSOCIATION
from opencood.models.dualcmp.box_refinement import encode_box


def test_complete_targets_and_dense_centers_but_partial_context_points_remain_valid():
    mask = torch.ones(1, 2, 50, dtype=torch.bool)
    mask[0, 1, 10:] = False
    assert forecast_supervision_mask(mask, COMPLETE_FORECAST_SUPERVISION).tolist() == [[True, False]]
    assert forecast_supervision_mask(mask, PARTIAL_FORECAST_SUPERVISION).tolist() == [[True, True]]
    with pytest.raises(ValueError, match="fifty"):
        forecast_supervision_mask(mask[..., :10], COMPLETE_FORECAST_SUPERVISION)
    boxes = torch.tensor([[[0., 0., 0., 1., 1., 2., 0.], [4., 0., 0., 1., 1., 2., 0.]]])
    main = torch.zeros(1, 2, 1, 2, 50, 5, requires_grad=True)
    dense = torch.zeros(2, 2, 50, 5, requires_grad=True)
    output = dict(psm=torch.zeros(1, requires_grad=True),
        state=SimpleNamespace(box=boxes, runtime_id=torch.tensor([[0, 1]]), valid_mask=torch.ones(1, 2, dtype=torch.bool)),
        track_box=boxes, track_box_code=encode_box(boxes), track_velocity=torch.zeros(1, 2, 2),
        track_exist_logits=torch.zeros(1, 2, requires_grad=True), forecast={
            "forecast_coordinate_frame": "current_receiver", "forecast_valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "loss_pred_scores_layers": torch.zeros(1, 2, 1, 2, requires_grad=True),
            "loss_pred_trajs_local_layers": main,
            "loss_intention_points_layers": torch.zeros(1, 2, 1, 2, 2),
            "packed_dense_trajs_local": dense, "packed_entries": ((0, 0), (0, 1)),
            "packed_context_slots": torch.tensor([[0, 1], [0, 1]]), "packed_center_heading": torch.zeros(2),
        })
    target = dict(current_mask=torch.ones(1, 2, dtype=torch.bool), actor_ids=torch.tensor([[101, 102]]),
        current_boxes=boxes, current_velocity=torch.zeros(1, 2, 2), velocity_mask=torch.zeros(1, 2, dtype=torch.bool),
        future_xy=torch.zeros(1, 2, 50, 2), future_mask=mask, forecast_coordinate_frame="current_receiver",
        fused_label_dict={})
    class DetectorZero(nn.Module):
        def forward(self, output, target):
            return output['psm'].sum() * 0.
        def cls_loss_func(self, logits, target, weights):
            return logits * 0.
    losses = {}
    for contract in (PARTIAL_FORECAST_SUPERVISION, COMPLETE_FORECAST_SUPERVISION):
        loss = DualCmpLoss({'detector': {'cls_weight': 1., 'reg': 2.}, 'forecast_mean_weight': 5.,
                           'forecast_association_policy': VALIDATED_ASSOCIATION, 'forecast_supervision': contract})
        loss.detector_loss = DetectorZero()
        losses[contract] = loss([output], [target], forecast_assignments=[torch.tensor([[0, 1]])])
    old, fixed = losses[PARTIAL_FORECAST_SUPERVISION], losses[COMPLETE_FORECAST_SUPERVISION]
    assert old['C_gmm'] == 2 and old['C_mean_reg'] == 60 and old['C_dense_gmm'] == 2
    assert fixed['C_gmm'] == 1 and fixed['C_mean_reg'] == 50 and fixed['C_dense_gmm'] == 1
    (fixed['S_gmm'] + fixed['S_mode'] + 5 * fixed['S_mean_reg']).backward(retain_graph=True)
    assert main.grad[0, 0].abs().sum() > 0 and main.grad[0, 1].abs().sum() == 0
    fixed['S_dense_gmm'].backward()
    assert dense.grad[0, 1, :10].abs().sum() > 0
    assert dense.grad[0, 1, 10:].abs().sum() == 0
    assert dense.grad[1].abs().sum() == 0
