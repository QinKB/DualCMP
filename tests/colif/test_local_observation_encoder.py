import torch

from opencood.models.colif.local_observation_encoder import (
    LocalObservationEncoder,
    delta_to_boxes3d,
)


def _encoder() -> LocalObservationEncoder:
    return LocalObservationEncoder(
        input_dim=8,
        anchor_number=2,
        args={
            "max_per_source": 4,
            "pre_nms_topk": 8,
            "score_threshold": 0.0,
            "nms_threshold": 0.15,
            "query_dim": 16,
            "lidar_range": [-4.0, -2.0, -2.0, 4.0, 2.0, 2.0],
            "voxel_size": [1.0, 1.0, 4.0],
            "anchor_dims_hwl": [1.5, 1.6, 3.9],
            "anchor_rotations_deg": [0.0, 90.0],
        },
    )


def test_zero_residual_decodes_to_anchor() -> None:
    encoder = _encoder()
    feature = torch.zeros(1, 8, 2, 4)
    anchors = encoder._anchors(2, 4, feature)
    deltas = torch.zeros(1, 14, 2, 4)
    decoded = delta_to_boxes3d(deltas, anchors)
    torch.testing.assert_close(decoded[0], anchors.reshape(-1, 7))
    expected_bias = torch.full_like(encoder.local_cls_head.bias, -4.59511985)
    torch.testing.assert_close(encoder.local_cls_head.bias, expected_bias)


def test_observation_shape_and_roi_gradient() -> None:
    encoder = _encoder()
    local_bev = torch.randn(2, 8, 2, 4, requires_grad=True)
    runtime = {
        "source_mask": torch.tensor([[True, True]]),
        "source_to_batch": torch.tensor([0, 0]),
        "source_slot": torch.tensor([0, 1]),
    }
    output = encoder(local_bev, runtime)
    assert output["obs_feat"].shape == (1, 2, 4, 16)
    assert output["obs_box"].shape == (1, 2, 4, 7)
    assert output["obs_mask"].shape == (1, 2, 4)
    output["obs_feat"].sum().backward()
    assert local_bev.grad is not None
    assert torch.isfinite(local_bev.grad).all()
    assert local_bev.grad.abs().sum() > 0
