"""CPU contracts for tracker feature ownership."""

from dataclasses import replace

import torch

from opencood.models.dualcmp.state import ActorState
from opencood.models.dualcmp.tracker import ActorTracker


def _state(feature):
    pose = torch.eye(4).unsqueeze(0)
    state = ActorState.empty(1, 1, feature.shape[-1], 1, pose, torch.zeros(1))
    box = torch.tensor([[[1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0]]])
    mean = torch.tensor([[[1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0]]])
    return replace(
        state, feature=feature, box=box, valid_mask=torch.tensor([[True]]),
        active_mask=torch.tensor([[True]]), runtime_id=torch.tensor([[7]]),
        next_runtime_id=torch.tensor([8]), last_update_time_s=torch.zeros(1, 1),
        filter_mean=mean, filter_cov=torch.eye(10).reshape(1, 1, 10, 10),
    )


def _proposals(boxes):
    return {"boxes": boxes, "scores": torch.ones(boxes.shape[:2]),
            "mask": torch.ones(boxes.shape[:2], dtype=torch.bool)}


def test_tracker_feature_follows_runtime_id_and_preserves_retained_gradient():
    tracker = ActorTracker(max_miss=1)
    pose, time_s = torch.eye(4).unsqueeze(0), torch.tensor([0.1])

    retained_feature = torch.tensor([[[3.0, -4.0]]], requires_grad=True)
    retained = tracker(_state(retained_feature), _proposals(torch.tensor([[[1.0, 0.0, 0.0, 1, 1, 2, 0]]])),
                       pose, time_s, 1.0)
    torch.testing.assert_close(retained.feature, retained_feature)
    retained.feature.sum().backward()
    torch.testing.assert_close(retained_feature.grad, torch.ones_like(retained_feature))

    invalid = tracker(_state(torch.tensor([[[5.0, 6.0]]])), _proposals(torch.zeros(1, 0, 7)),
                      pose, time_s, 1.0)
    assert not invalid.valid_mask.any()
    assert not invalid.feature.any()

    reused = tracker(_state(torch.tensor([[[7.0, 8.0]]])),
                     _proposals(torch.tensor([[[10.0, 0.0, 0.0, 1, 1, 2, 0]]])), pose, time_s, 1.0)
    assert reused.valid_mask.tolist() == [[True]]
    assert reused.runtime_id.tolist() == [[8]]
    assert not reused.feature.any()
