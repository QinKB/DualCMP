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
            "--lr",
            "0.0005",
            "--forecast-bev-representation",
            "dual",
        ],
    )

    args = train_colif._arguments()

    assert args.seed == 42
    assert args.lr == 5e-4
    assert args.forecast_bev_representation == "dual"


def test_learning_rate_override_updates_optimizer_config():
    hypes = {"optimizer": {"lr": 1e-4}}

    value = train_colif._apply_learning_rate_override(hypes, 5e-4)

    assert value == 5e-4
    assert hypes["optimizer"]["lr"] == 5e-4


@pytest.mark.parametrize("value", (0.0, -1e-4, float("inf"), float("nan")))
def test_learning_rate_override_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="learning rate"):
        train_colif._apply_learning_rate_override(
            {"optimizer": {"lr": 5e-4}}, value
        )


def test_train_cli_rejects_mtr_checkpoint_initialization(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_colif",
            "--hypes_yaml",
            "config.yaml",
            "--init-mtr-checkpoint",
            "mtr.pth",
        ],
    )

    with pytest.raises(SystemExit):
        train_colif._arguments()


def test_grad_accum_override_is_checkpointed_in_config():
    hypes = {"train_params": {"grad_accum_steps": 1}}

    value = train_colif._apply_grad_accum_override(hypes, 4)

    assert value == 4
    assert hypes["train_params"]["grad_accum_steps"] == 4


def test_grad_accum_override_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="grad_accum_steps"):
        train_colif._apply_grad_accum_override(
            {"train_params": {"grad_accum_steps": 1}}, 0
        )
