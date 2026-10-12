import numpy as np
import pytest

from opencood.tools.dualcmp import cmp_adapted_forecast_metrics as metrics


def frame(pred_scores=(0.9, 0.1), gt_count=1):
    return {"key": ("scene", 0, 10),
            "gt_ids": np.arange(gt_count) + 1,
            "gt_boxes": np.asarray([[0., 0., 0., 2., 2., 4., 0.]] * gt_count).reshape(gt_count, 7),
            "gt_future": np.zeros((gt_count, 50, 2)),
            "gt_mask": np.ones((gt_count, 50), dtype=bool),
            "pred_ids": np.arange(len(pred_scores)) + 10,
            "pred_boxes": np.asarray([[0., 0., 0., 2., 2., 4., 0.]] * len(pred_scores)).reshape(len(pred_scores), 7),
            "pred_future": np.zeros((len(pred_scores), 6, 50, 2)),
            "pred_scores": np.asarray([[1 / 6.] * 6] * len(pred_scores)).reshape(len(pred_scores), 6),
            "detection_scores": np.asarray(pred_scores),
            "history_mask": np.ones((len(pred_scores), 11), dtype=bool),
            "history_steps": np.full(len(pred_scores), 11)}


def test_official_metric_parity_and_cmp_endpoint_indices():
    f = frame(pred_scores=(0.9,))
    # Exact evaluator endpoints: FDE index 10/30/49 and 5s ADE excludes 49.
    f["pred_future"][0, :, 10, 0] = 1.
    f["pred_future"][0, :, 30, 0] = 2.
    f["pred_future"][0, :, 49, 0] = 4.
    report = metrics.cmp_adapted_report([f])
    expected, expected_count = metrics._official_metrics([(f, 0, 0)])
    assert expected_count == report["model"]["count"] == 1
    for name in metrics.PRIMARY_METRICS + tuple(
            key for key in metrics.MR_METRICS if key not in metrics.PRIMARY_METRICS):
        assert report["model"][name] == pytest.approx(expected[name])
    _, _, official = metrics._official_functions()

    class Identity:
        def eval(self):
            pass

        def __call__(self, records):
            return records

    class Logger:
        def info(self, *args, **kwargs):
            pass

    raw = official(Identity(), [[{"scenario_name": "scene", "timestamp_idx": 10,
                    "object_id": 1, "ego_cav_id": "0", "gt_trajs": f["gt_future"][0],
                    "pred_trajs": f["pred_future"][0], "pred_scores": f["pred_scores"][0],
                    "center_gt_trajs_mask": f["gt_mask"][0],
                    "center_gt_final_valid_idx": 49}]], 0, Logger())
    assert report["model"]["minADE_5s"] == pytest.approx(raw["minADE"])
    assert report["model"]["minFDE_5s"] == pytest.approx(raw["minFDE"])
    assert report["model"]["MR_3.6"] == pytest.approx(raw["MR"])
    assert report["model"]["minFDE_1s"] == 1.
    assert report["model"]["minFDE_3s"] == 2.
    assert report["model"]["minFDE_5s"] == 4.
    assert report["model"]["minADE_5s"] == pytest.approx(3 / 49)  # index 49 is excluded
    assert "not official CMP end-to-end evaluation" in report["evaluation_protocol"]["scope"]
    assert "not the CMP paper headline" in report["top1"]["scope"]


def test_score_order_pairing_uses_correct_prediction_slot_and_counts_fp_fn():
    f = frame(pred_scores=(0.2, 0.9), gt_count=1)
    f["pred_future"][0] = 9.
    f["pred_future"][1] = 1.
    result = metrics.cmp_adapted_report([f])
    assert result["model"]["minADE_5s"] == pytest.approx(2 ** 0.5)
    assert result["coverage"]["tp"] == 1
    assert result["coverage"]["fp"] == 1
    assert result["coverage"]["fn"] == 0
    assert result["coverage"]["matched_rows_provenance"]["eligible_rows"] == 1
    assert result["coverage"]["matched_rows_provenance"]["prediction_identity"] == "receiver-local runtime pred_ids"


def test_future_and_history_eligibility_happens_after_matching():
    f = frame(pred_scores=(0.9, 0.8, 0.7), gt_count=3)
    f["gt_mask"][1, -1] = False
    f["history_mask"][2, 0] = False
    f["history_steps"][2] = 10
    result = metrics.cmp_adapted_report([f])
    cov = result["coverage"]
    assert cov["current_gt"] == 3 and cov["predictions"] == 3
    assert cov["tp"] == 3 and cov["fn"] == 0
    assert cov["complete_gt"] == 2 and cov["eligible_complete"] == 1
    assert cov["rejected_incomplete_future"] == 1
    assert cov["rejected_short_history"] == 1
    assert cov["missed_complete"] == 1
    assert result["model"]["count"] == 1


def test_empty_population_returns_null_without_calling_official_metric(monkeypatch):
    f = frame(pred_scores=(), gt_count=0)
    monkeypatch.setattr(metrics, "_official_metrics", lambda pairs: (_ for _ in ()).throw(AssertionError()))
    result = metrics.cmp_adapted_report([f])
    assert result["model"]["count"] == 0
    names = metrics.PRIMARY_METRICS + tuple(
        key for key in metrics.MR_METRICS if key not in metrics.PRIMARY_METRICS)
    assert all(result[group][name] is None for group in ("model", "top1") for name in names)


def test_ddp_padding_deduplicates_and_conflicts_fail_closed():
    f = frame(pred_scores=(0.9,))
    assert metrics.cmp_adapted_report([f, f])["coverage"]["prediction_keys"] == 1
    conflict = dict(f, detection_scores=np.asarray([0.8]))
    with pytest.raises(ValueError, match="DDP padded"):
        metrics.cmp_adapted_report([f, conflict])


def test_validation_rejects_bad_shape_nonfinite_or_nonpositive_dimensions():
    f = frame(pred_scores=(0.9,))
    bad = dict(f, pred_future=np.full((1, 6, 49, 2), np.nan))
    with pytest.raises(ValueError, match="shape"):
        metrics.cmp_adapted_report([bad])
    bad = dict(f, pred_future=np.full((1, 6, 50, 2), np.nan))
    with pytest.raises(ValueError, match="nonfinite"):
        metrics.cmp_adapted_report([bad])
    f["pred_boxes"][0, 3] = 0
    with pytest.raises(ValueError, match="positive"):
        metrics.cmp_adapted_report([f])


def test_iou05_experiment_only_changes_pairing_gate_and_preserves_production():
    from eval_cmp_iou05_ddp import report_iou05, IOU05_CONTRACT
    f = frame(pred_scores=(0.9,))
    f["pred_boxes"][0, 1] = 0.5  # Same-size box: BEV IoU is 0.6.
    f["pred_future"][0, :, :, 0] = 1.
    old_contract = dict(metrics.CONTRACT)
    result = report_iou05([f])
    assert result["evaluation_protocol"] == IOU05_CONTRACT
    assert result["model"]["count"] == 1
    assert result["model"]["minADE_5s"] == pytest.approx(1.)
    assert result["iou07_control"]["model"]["count"] == 0
    assert result["matching_population_delta"]["added_iou05_pairs"] == 1
    assert result["iou_only_adapter"]["official_max_difference"] < 1e-8
    assert metrics.CONTRACT == old_contract
    assert metrics.cmp_adapted_report([f], iou_threshold=0.7)["model"]["count"] == 0
    assert metrics.cmp_adapted_report([f])["model"]["count"] == 1


def test_default_iou05_and_explicit_legacy_iou07_protocols():
    f = frame(pred_scores=(0.9,))
    production = metrics.cmp_adapted_report([f])
    legacy = metrics.cmp_adapted_report([f], iou_threshold=0.7)
    assert metrics.PRIMARY_METRICS == (
        "minADE_1s", "minADE_3s", "minADE_5s", "minFDE_1s",
        "minFDE_3s", "minFDE_5s", "MR_2.0")
    assert tuple(metrics.MR_METRICS) == ("MR_3.6", "MR_3.0", "MR_2.0", "MR_1.8")
    assert production["metric_version"] == "cmp_gtfree_adapter_iou05_mr2_v1"
    assert production["evaluation_protocol"] == metrics.CONTRACT
    assert legacy["metric_version"] == "cmp_gtfree_adapter_v1"
    assert legacy["evaluation_protocol"] == metrics.LEGACY_CONTRACT
    assert set(metrics.MR_METRICS) <= production["model"].keys()
    assert set(metrics.MR_METRICS) <= legacy["model"].keys()
    with pytest.raises(ValueError, match="0.5 or 0.7"):
        metrics.cmp_adapted_report([f], iou_threshold=0.6)


def test_mr2_uses_strictly_greater_than_two_metres():
    exact = frame(pred_scores=(0.9,))
    exact["pred_future"][0, :, 49, 0] = 2.0
    over = frame(pred_scores=(0.9,))
    over["pred_future"][0, :, 49, 0] = 2.0001
    assert metrics.cmp_adapted_report([exact])["model"]["MR_2.0"] == 0.0
    assert metrics.cmp_adapted_report([over])["model"]["MR_2.0"] == 1.0
