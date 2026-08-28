import math

import torch
import pytest

from opencood.loss.colif_loss import ColifLoss
from opencood.models.colif.actor_source_tokenizer import (
    ActorAlignedSourceTokenizer,
)


def _module(**kwargs):
    config = {
        "input_dim": 8,
        "query_dim": 8,
        "max_cav": 2,
        "lidar_range": [-8.0, -8.0, -3.0, 8.0, 8.0, 3.0],
        "roi_grid_size": 5,
        "roi_scale": 2.0,
    }
    config.update(kwargs)
    return ActorAlignedSourceTokenizer(**config)


def _boxes(yaw=0.0):
    return torch.tensor(
        [[[0.0, 0.0, 0.0, 1.6, 2.0, 4.0, yaw]]]
    )


def test_tokenizer_preserves_source_and_spatial_dimensions() -> None:
    output = _module()(
        torch.randn(1, 2, 8, 8, 8),
        torch.tensor([[True, True]]),
        _boxes(),
        torch.ones(1, 1, dtype=torch.bool),
    )

    assert output["tokens"].shape == (1, 1, 2, 25, 8)
    assert output["positions"].shape == (1, 1, 2, 25, 8)
    assert output["token_mask"].shape == (1, 1, 2, 25)
    assert output["sampling_grid"].shape == (1, 1, 25, 2)


def test_masked_partner_has_zero_tokens_and_no_gradient() -> None:
    module = _module().train()
    source = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    output = module(
        source,
        torch.tensor([[True, False]]),
        _boxes(),
        torch.ones(1, 1, dtype=torch.bool),
    )

    assert torch.count_nonzero(output["tokens"][:, :, 1]) == 0
    assert torch.count_nonzero(output["positions"][:, :, 1]) == 0
    assert not output["token_mask"][:, :, 1].any()
    output["tokens"].square().mean().backward()
    assert torch.count_nonzero(source.grad[:, 0]) > 0
    assert torch.count_nonzero(source.grad[:, 1]) == 0


def test_inactive_actor_has_zero_tokens_and_mask() -> None:
    output = _module()(
        torch.randn(1, 2, 8, 8, 8),
        torch.tensor([[True, True]]),
        _boxes(),
        torch.zeros(1, 1, dtype=torch.bool),
    )

    assert torch.count_nonzero(output["tokens"]) == 0
    assert torch.count_nonzero(output["positions"]) == 0
    assert not output["token_mask"].any()


def test_roi_grid_rotates_with_actor_heading() -> None:
    module = _module()
    grid_zero, _, _ = module._sampling_grid(_boxes(yaw=0.0))
    grid_quarter, _, _ = module._sampling_grid(_boxes(yaw=math.pi / 2.0))

    zero_metric = grid_zero * 8.0
    quarter_metric = grid_quarter * 8.0
    expected = torch.stack(
        (-zero_metric[..., 1], zero_metric[..., 0]), dim=-1
    )
    torch.testing.assert_close(quarter_metric, expected, atol=1e-5, rtol=1e-5)


def test_source_permutation_only_permutes_source_tokens() -> None:
    torch.manual_seed(11)
    module = _module().eval()
    source = torch.randn(1, 2, 8, 8, 8)
    mask = torch.tensor([[True, True]])
    valid = torch.ones(1, 1, dtype=torch.bool)

    original = module(source, mask, _boxes(), valid)
    swapped = module(source.flip(1), mask.flip(1), _boxes(), valid)

    torch.testing.assert_close(
        original["tokens"].flip(2), swapped["tokens"]
    )
    torch.testing.assert_close(
        original["positions"].flip(2), swapped["positions"]
    )
    torch.testing.assert_close(
        original["token_mask"].flip(2), swapped["token_mask"]
    )


def test_position_input_preserves_metric_extent_in_mtr_local_frame() -> None:
    module = _module()
    small = _boxes()
    large = _boxes()
    large[..., 5] = 8.0
    _, _, small_position = module._sampling_grid(small)
    _, _, large_position = module._sampling_grid(large)

    assert large_position[..., 0].abs().max() > small_position[..., 0].abs().max()


def test_position_input_matches_mtr_legacy_local_frame_at_nonzero_yaw() -> None:
    module = _module()
    boxes = _boxes(yaw=math.pi / 2.0)
    grid, _, encoded_local = module._sampling_grid(boxes)
    receiver_offset = grid * 8.0
    expected = ColifLoss._receiver_displacement_to_actor(
        receiver_offset,
        boxes[..., 6, None],
    ) / 8.0

    torch.testing.assert_close(encoded_local, expected, atol=1e-6, rtol=1e-6)


def test_invalid_tokenizer_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="roi_grid_size"):
        _module(roi_grid_size=4)
    with pytest.raises(ValueError, match="roi_scale"):
        _module(roi_scale=0.0)
