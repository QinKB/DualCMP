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
    source_bev, cav_mask = select_forecast_bev_sources(
        "per_cav", encoded
    )

    assert source_bev is encoded["regrouped_source_bev"]
    torch.testing.assert_close(cav_mask, encoded["cav_mask"])


def test_fused_forecast_representation_is_one_receiver_frame_source():
    encoded = _encoded()
    source_bev, cav_mask = select_forecast_bev_sources(
        "fused", encoded
    )

    assert source_bev.shape == (2, 1, 4, 3, 5)
    torch.testing.assert_close(source_bev[:, 0], encoded["fused_bev"])
    torch.testing.assert_close(cav_mask, torch.ones(2, 1, dtype=torch.bool))
    source_bev.sum().backward()
    torch.testing.assert_close(
        encoded["fused_bev"].grad,
        torch.ones_like(encoded["fused_bev"]),
    )


def test_forecast_representation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="representation"):
        select_forecast_bev_sources("unknown", _encoded())


def test_train_cli_accepts_forecast_ablation_switches(monkeypatch):
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
            "--source-aggregation",
            "mean",
            "--source-geometry",
            "off",
            "--forecast-trajectory-parameterization",
            "direct",
        ],
    )

    args = train_colif._arguments()

    assert args.seed == 42
    assert args.forecast_bev_representation == "fused"
    assert args.source_aggregation == "mean"
    assert args.source_geometry == "off"
    assert args.forecast_trajectory_parameterization == "direct"


def test_retired_flat_source_ablation_cli_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_colif",
            "--hypes_yaml",
            "config.yaml",
            "--source-aggregation",
            "mean",
            "--source-geometry",
            "off",
            "--forecast-trajectory-parameterization",
            "direct",
        ],
    )
    args = train_colif._arguments()
    hypes = {
        "model": {
            "args": {
                "actor_cooperative_fusion": {
                    "aggregation": "attention",
                    "geometry_conditioning": True,
                },
                "forecast": {"trajectory_parameterization": "anchored"},
            }
        }
    }

    with pytest.raises(ValueError, match="retired flat source reader"):
        train_colif._apply_forecast_ablation_overrides(hypes, args)


def test_direct_parameterization_cli_remains_supported(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_colif",
            "--hypes_yaml",
            "config.yaml",
            "--forecast-trajectory-parameterization",
            "direct",
        ],
    )
    args = train_colif._arguments()
    hypes = {
        "model": {
            "args": {
                "actor_cooperative_fusion": {},
                "forecast": {"trajectory_parameterization": "anchored"},
            }
        }
    }

    train_colif._apply_forecast_ablation_overrides(hypes, args)

    assert (
        hypes["model"]["args"]["forecast"]["trajectory_parameterization"]
        == "direct"
    )
