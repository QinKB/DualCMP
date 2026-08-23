import pytest
import torch

from opencood.models.point_pillar_colif import select_forecast_bev_sources
from opencood.tools import train_colif


def _encoded():
    fused = torch.randn(2, 4, 3, 5, requires_grad=True)
    sources = torch.randn(2, 2, 4, 3, 5)
    return {
        "fused_bev": fused,
        "regrouped_source_bev": sources,
        "cav_mask": torch.tensor([[True, True], [True, False]]),
    }


def test_per_cav_forecast_representation_preserves_source_inputs():
    encoded = _encoded()
    transforms = torch.randn(2, 2, 4, 4)

    source_bev, cav_mask, selected_transforms = select_forecast_bev_sources(
        "per_cav", encoded, transforms
    )

    assert source_bev is encoded["regrouped_source_bev"]
    torch.testing.assert_close(cav_mask, encoded["cav_mask"])
    assert selected_transforms is transforms


def test_fused_forecast_representation_is_one_receiver_frame_source():
    encoded = _encoded()
    transforms = torch.randn(2, 2, 4, 4)

    source_bev, cav_mask, selected_transforms = select_forecast_bev_sources(
        "fused", encoded, transforms
    )

    assert source_bev.shape == (2, 1, 4, 3, 5)
    torch.testing.assert_close(source_bev[:, 0], encoded["fused_bev"])
    torch.testing.assert_close(cav_mask, torch.ones(2, 1, dtype=torch.bool))
    torch.testing.assert_close(
        selected_transforms,
        torch.eye(4).view(1, 1, 4, 4).expand(2, 1, 4, 4),
    )
    source_bev.sum().backward()
    torch.testing.assert_close(
        encoded["fused_bev"].grad,
        torch.ones_like(encoded["fused_bev"]),
    )


def test_forecast_representation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="representation"):
        select_forecast_bev_sources("unknown", _encoded(), None)


def test_train_cli_accepts_seed_and_forecast_representation(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_colif",
            "--hypes_yaml",
            "config.yaml",
            "--seed",
            "42",
            "--forecast-bev-representation",
            "fused",
        ],
    )

    args = train_colif._arguments()

    assert args.seed == 42
    assert args.forecast_bev_representation == "fused"
