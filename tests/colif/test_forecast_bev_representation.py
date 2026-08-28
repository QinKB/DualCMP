import pytest
import torch

from opencood.models.point_pillar_colif import select_forecast_bev_inputs
from opencood.tools import train_colif


def _encoded():
    fused = torch.randn(2, 4, 3, 5, requires_grad=True)
    sources = torch.randn(2, 2, 4, 3, 5)
    return {
        "fused_bev": fused,
        "regrouped_source_bev": sources,
        "cav_mask": torch.tensor([[True, True], [True, False]]),
    }


def test_dual_representation_preserves_consensus_and_source_inputs():
    encoded = _encoded()
    selected = select_forecast_bev_inputs("dual", encoded)
    consensus_bev, consensus_mask = selected["consensus"]
    source_bev, source_mask = selected["source"]

    torch.testing.assert_close(consensus_bev[:, 0], encoded["fused_bev"])
    torch.testing.assert_close(
        consensus_mask,
        torch.ones(2, 1, dtype=torch.bool),
    )
    assert source_bev is encoded["regrouped_source_bev"]
    torch.testing.assert_close(source_mask, encoded["cav_mask"])


def test_fused_forecast_representation_is_one_receiver_frame_source():
    encoded = _encoded()
    selected = select_forecast_bev_inputs("fused", encoded)
    source_bev, cav_mask = selected["consensus"]

    assert source_bev.shape == (2, 1, 4, 3, 5)
    torch.testing.assert_close(source_bev[:, 0], encoded["fused_bev"])
    torch.testing.assert_close(cav_mask, torch.ones(2, 1, dtype=torch.bool))
    source_bev.sum().backward()
    torch.testing.assert_close(
        encoded["fused_bev"].grad,
        torch.ones_like(encoded["fused_bev"]),
    )
    assert selected["source"] == (None, None)


def test_source_ablation_excludes_fused_consensus():
    encoded = _encoded()
    selected = select_forecast_bev_inputs("source", encoded)

    assert selected["consensus"] == (None, None)
    assert selected["source"][0] is encoded["regrouped_source_bev"]
    torch.testing.assert_close(selected["source"][1], encoded["cav_mask"])


def test_forecast_representation_rejects_unknown_mode():
    with pytest.raises(ValueError, match="representation"):
        select_forecast_bev_inputs("unknown", _encoded())


def test_train_cli_accepts_forecast_representation(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_colif",
            "--hypes_yaml",
            "config.yaml",
            "--seed",
            "42",
            "--forecast-bev-representation",
            "dual",
        ],
    )

    args = train_colif._arguments()

    assert args.seed == 42
    assert args.forecast_bev_representation == "dual"
