"""Small decisive checks of the native adapter against pinned UniAD code."""
import numpy as np
import pytest

from opencood.tools.dualcmp.mainstream_forecast_metrics import (
    CONTRACT, _accumulate, _validated_frames, forecast_frame, mainstream_report, official_reference,
)


def frame():
    return {"key": ("scene", 0, 10), "gt_ids": np.array([1, 2]),
            "gt_centers": np.array([[0., 0.], [20., 0.]]),
            "gt_future": np.zeros((2, 50, 2)), "gt_mask": np.ones((2, 50), bool),
            "pred_ids": np.array([10]), "pred_centers": np.zeros((1, 2)),
            "pred_future": np.zeros((1, 6, 50, 2)), "pred_scores": np.array([[.5, .1, .1, .1, .1, .1]]),
            "detection_scores": np.array([.9])}


def test_official_distance_boundaries_and_no_match_counts():
    f = frame()
    for offset, count in ((.999, 1), (1., 0), (1.001, 0)):
        f["pred_centers"][0, 0] = offset
        _, counts = _accumulate([f], 10, 6, 1.)
        assert counts["tp"] == count and counts["gt"] == 2
        assert counts["fp"] == 1 - count
    f["pred_centers"][0, 0] = 2.
    report = mainstream_report([f])
    assert report["model"]["minADE_1s"] is None
    assert report["model"]["EPA_1s"] == pytest.approx(-.5 / (2 + 1e-5))
    assert report["model"]["minFDE_AP_1s"] == 0.


def test_official_mode_selection_mr_is_not_endpoint_and_top1_uses_scores():
    f = frame()
    # Mode 0 has the best ADE; mode 1 has the best endpoint. All modes
    # exceed 2 m somewhere, even though one endpoint is perfect.
    f["pred_future"][0, :, :, 0] = 4.
    f["pred_future"][0, 0, :, 0] = 3.
    f["pred_future"][0, 1, -1, 0] = 0.
    f["pred_scores"][0] = [.1, .1, .1, .1, .1, .5]
    report = mainstream_report([f])
    assert report["model"]["minADE_5s"] == pytest.approx(3.)
    assert report["model"]["minFDE_5s"] == 0.
    assert report["model"]["MR_max_2m_5s"] == 1.
    assert report["top1"]["minADE_5s"] == pytest.approx(4.)
    # Same tensors go through the unmodified official scalar metric funcs.
    metrics = official_reference().prediction_metrics
    from types import SimpleNamespace
    ade, fde, mr = metrics(SimpleNamespace(traj=f["gt_future"][0]),
                           SimpleNamespace(traj=f["pred_future"][0]))
    assert float(ade[0]) == 3. and float(fde[0]) == 0. and float(mr) == 1.


def test_exact_horizons_gt_population_and_epa_keep_missed_targets():
    f = frame()
    f["gt_mask"][1, 10:] = False
    report = mainstream_report([f])
    assert report["coverage"]["horizons"]["1s"]["epa_detection"]["gt"] == 2
    assert report["coverage"]["horizons"]["3s"]["epa_detection"]["gt"] == 1
    assert report["model"]["EPA_1s"] == pytest.approx(1 / (2 + 1e-5))
    assert report["model"]["EPA_5s"] == pytest.approx(1 / (1 + 1e-5))
    f["pred_future"][0, :, 9, 0] = 2.
    report = mainstream_report([f])
    assert report["model"]["minFDE_1s"] == 2.
    assert report["model"]["EPA_1s"] == 0.  # strict FDE < 2, not <= 2
    assert report["model"]["MR_max_2m_1s"] == 0.  # strict MR > 2


def test_ddp_padding_empty_forecasts_and_invalid_exports():
    f = frame()
    assert mainstream_report([f, f])["model"] == mainstream_report([f])["model"]
    bad = dict(f, pred_future=np.full((1, 6, 50, 2), np.nan))
    with pytest.raises(ValueError, match="nonfinite"):
        mainstream_report([bad])
    conflicting = dict(f, detection_scores=np.array([.8]))
    with pytest.raises(ValueError, match="DDP padded"):
        _validated_frames([f, conflicting])
    f.update(pred_ids=np.empty(0, int), pred_centers=np.empty((0, 2)),
             pred_future=np.empty((0, 6, 50, 2)), pred_scores=np.empty((0, 6)),
             detection_scores=np.empty(0))
    result = mainstream_report([f])
    assert result["model"]["minADE_5s"] is None
    assert result["model"]["EPA_5s"] == 0.
    assert result["coverage"]["horizons"]["5s"]["epa_detection"]["fn"] == 2
    assert CONTRACT["epa_center_distance_m"] == 2.


def test_native_export_anchors_predictions_at_runtime_not_gt():
    import torch
    from types import SimpleNamespace
    output = {"state": SimpleNamespace(runtime_id=torch.tensor([[10, 11]])),
        "track_box": torch.tensor([[[.5, 0., 0., 1., 1., 1., 0.], [50., 0., 0., 1., 1., 1., 0.]]]),
        "track_exist_logits": torch.tensor([[2., 1.]]),
        "forecast": {"forecast_coordinate_frame": "current_receiver",
            "forecast_valid_mask": torch.tensor([[True, True]]),
            "pred_trajs": torch.zeros(1, 2, 6, 50, 5),
            "pred_scores": torch.full((1, 2, 6), 1 / 6.)}}
    target = {"forecast_coordinate_frame": "current_receiver",
        "current_mask": torch.tensor([[True, True]]), "actor_ids": torch.tensor([[1, 2]]),
        "current_boxes": torch.tensor([[[0., 0., 0., 1., 1., 1., 0.], [20., 0., 0., 1., 1., 1., 0.]]]),
        "future_xy": torch.zeros(1, 2, 50, 2), "future_mask": torch.ones(1, 2, 50, dtype=torch.bool)}
    exported = forecast_frame(output, target, 0, ("scene", 0, 10))
    assert exported["pred_future"][0, 0, 0, 0] == .5
    assert exported["gt_future"][0, 0, 0] == 0.
    report = mainstream_report([exported])
    assert report["model"]["minADE_1s"] == pytest.approx(.5)
    # The prediction at exactly 50 m is out of the official vehicle range.
    counts = report["coverage"]["horizons"]["1s"]["epa_detection"]
    assert counts["predictions"] == 1 and counts["gt"] == 2 and counts["fn"] == 1


def test_score_ordered_one_to_one_and_invalid_future_not_hidden_misses():
    f = frame()
    f.update(pred_ids=np.array([10, 11]), pred_centers=np.zeros((2, 2)),
        pred_future=np.zeros((2, 6, 50, 2)), pred_scores=np.full((2, 6), 1 / 6.),
        detection_scores=np.array([.8, .9]))
    _, counts = _accumulate([f], 50, 6, 1.)
    assert counts["tp"] == 1 and counts["fp"] == 1 and counts["fn"] == 1
    assert mainstream_report([f])["model"]["EPA_5s"] == pytest.approx(.5 / (2 + 1e-5))
    f["gt_mask"][0, 9] = False
    report = mainstream_report([f])
    counts = report["coverage"]["horizons"]["1s"]["epa_detection"]
    assert counts["gt"] == 1 and counts["ignored_unscorable_predictions"] == 2
    assert counts["fp"] == 0 and counts["fn"] == 1


def test_formal_entries_csv_keeps_only_primary_forecast_metrics():
    from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import PRIMARY_METRICS
    from opencood.tools.dualcmp import train_forecast_ddp as b, train_joint_ddp as c
    assert b.evaluate_forecast_training_ddp is c.evaluate_forecast_training_ddp
    assert b.FORECAST_METRICS == c.FORECAST_METRICS == PRIMARY_METRICS
    assert [key for key in b.CSV_FIELDS if key.startswith("test_")] == ["test_" + key for key in PRIMARY_METRICS]
    assert [key for key in c.CSV_FIELDS if key.startswith("test_forecast_")] == ["test_forecast_" + key for key in PRIMARY_METRICS]
    assert all("minFDE_AP" not in key and "top1" not in key for key in b.CSV_FIELDS + c.CSV_FIELDS)
    assert "test_MR_3.6" in b.LEGACY_CSV_FIELDS
