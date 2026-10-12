from types import SimpleNamespace

import torch

from opencood.loss.forecast_association import ASSOCIATION_STATUS
from opencood.tools.dualcmp.eval_forecast_ddp import (
    _validated_forecast_report,
    _validated_forecast_rows,
)


def _box(x=0.0):
    return [x, 0.0, 0.0, 1.5, 2.0, 4.0, 0.0]


def _output(boxes, runtime_ids, history_steps):
    boxes = torch.tensor([boxes], dtype=torch.float32)
    batch, slots = boxes.shape[:2]
    return {
        "state": SimpleNamespace(
            runtime_id=torch.tensor([runtime_ids]),
            valid_mask=torch.ones(batch, slots, dtype=torch.bool),
            active_mask=torch.ones(batch, slots, dtype=torch.bool),
            miss_count=torch.zeros(batch, slots, dtype=torch.long),
        ),
        "track_box": boxes,
        "track_exist_logits": torch.zeros(batch, slots),
        "forecast": {
            "forecast_coordinate_frame": "current_receiver",
            "forecast_valid_mask": torch.ones(batch, slots, dtype=torch.bool),
            "forecast_history_steps": torch.tensor([history_steps]),
            "pred_trajs": torch.zeros(batch, slots, 6, 50, 2),
        },
    }


def _association_target(boxes, actor_ids):
    boxes = torch.tensor([boxes], dtype=torch.float32)
    return {
        "current_boxes": boxes,
        "actor_ids": torch.tensor([actor_ids]),
        "current_mask": torch.ones(1, len(actor_ids), dtype=torch.bool),
    }


def _batch(final_boxes, actor_ids, future_xy=None, future_mask=None, history_boxes=None, history_ids=None):
    count = len(actor_ids)
    final = _association_target(final_boxes, actor_ids)
    final.update({
        "forecast_coordinate_frame": "current_receiver",
        "future_xy": torch.zeros(1, count, 50, 2) if future_xy is None else future_xy,
        "future_mask": torch.ones(1, count, 50, dtype=torch.bool) if future_mask is None else future_mask,
        "current_velocity": torch.zeros(1, count, 2),
    })
    prior = _association_target(
        final_boxes if history_boxes is None else history_boxes,
        actor_ids if history_ids is None else history_ids,
    )
    return {"association_target_frames": [prior, _association_target(final_boxes, actor_ids)],
            "target_frames": [final]}


def _key():
    return [("scene", 0, 10)]


def test_admitted_forecast_metrics_and_failed_associations_have_explicit_counts():
    outputs = [
        _output([_box(0), _box(100)], [10, 11], [2, 2]),
        _output([_box(0), _box(100)], [10, 11], [2, 2]),
    ]
    batch = _batch([_box(0), _box(200)], [101, 202])
    rows, coverage, records, assigned = _validated_forecast_rows(outputs, batch, _key())
    report = _validated_forecast_report(rows, coverage, records)

    assert assigned.tolist() == [[0, -1]]
    assert report["model"]["count"] == 1
    for metric in ("minADE_1s", "minADE_3s", "minADE_5s",
                   "minFDE_1s", "minFDE_3s", "minFDE_5s", "MR_3.6"):
        assert report["model"][metric] == 0.0
    assert report["association"]["unmatched_predictions"] == 1
    assert report["association"]["unmatched_current_gt"] == 1
    assert report["association"]["missed_complete"] == 1
    assert report["association"]["complete_future_coverage"] == 0.5
    assert report["association"]["forecast_precision"] == 0.5
    assert report["association"]["current_actor_recall"] == 0.5


def test_strict_observed_history_metric_is_unavailable_below_eleven_steps():
    outputs = [
        _output([_box()], [10], [2]),
        _output([_box()], [10], [2]),
    ]
    batch = _batch([_box()], [101])
    rows, coverage, records, _ = _validated_forecast_rows(outputs, batch, _key())
    report = _validated_forecast_report(rows, coverage, records)

    assert report["model"]["count"] == 1
    assert report["strict_observed_history_model"]["count"] == 0
    assert report["strict_observed_history_model"]["minADE_1s"] is None


def test_runtime_coasts_are_not_counted_as_measured_detection_history():
    outputs = [_output([_box()], [10], [11]) for _ in range(11)]
    outputs[3]["state"].miss_count.fill_(1)
    batch = _batch([_box()], [101])
    batch["association_target_frames"] = [_association_target([_box()], [101]) for _ in range(11)]
    rows, coverage, records, _ = _validated_forecast_rows(outputs, batch, _key())
    report = _validated_forecast_report(rows, coverage, records)
    assert records[0]["history_steps"] == 11
    assert records[0]["detection_observation_steps"] == 10
    assert report["model"]["count"] == 1
    assert report["strict_observed_history_model"]["count"] == 0


def test_causal_runtime_identity_switch_is_rejected_without_fake_metrics():
    outputs = [
        _output([_box(0)], [10], [11]),
        _output([_box(20)], [10], [11]),
    ]
    batch = _batch([_box(0), _box(20)], [101, 202])
    rows, coverage, records, assigned = _validated_forecast_rows(outputs, batch, _key())
    report = _validated_forecast_report(rows, coverage, records)

    assert assigned.tolist() == [[-1]]
    assert records[-1]["match_status"] == ASSOCIATION_STATUS[4]
    assert rows == []
    assert report["metric_status"] == "no_valid_forecast_pairs"
    assert report["model"]["count"] == 0
    assert report["model"]["minADE_1s"] is None
    assert report["strict_observed_history_model"]["minADE_1s"] is None


def test_ddp_padding_duplicates_do_not_inflate_rows_coverage_or_records():
    outputs = [
        _output([_box(0), _box(100)], [10, 11], [11, 11]),
        _output([_box(0), _box(100)], [10, 11], [11, 11]),
    ]
    batch = _batch([_box(0), _box(200)], [101, 202])
    rows, coverage, records, _ = _validated_forecast_rows(outputs, batch, _key())
    single = _validated_forecast_report(rows, coverage, records)
    padded = _validated_forecast_report(rows + rows, coverage + coverage, records + records)

    assert padded["model"] == single["model"]
    assert padded["association"] == single["association"]
    assert padded["coverage"] == single["coverage"]
    assert len(padded["association_records"]) == len(single["association_records"]) == 2


def test_headline_and_cmp_legacy_metrics_keep_distinct_endpoint_conventions():
    outputs = [
        _output([_box()], [10], [11]),
        _output([_box()], [10], [11]),
    ]
    errors = torch.arange(1, 51, dtype=torch.float32)
    outputs[-1]["forecast"]["pred_trajs"][0, 0, :, :, 0] = errors
    batch = _batch([_box()], [101])
    rows, coverage, records, _ = _validated_forecast_rows(outputs, batch, _key())
    report = _validated_forecast_report(rows, coverage, records)

    exact = report["model"]
    legacy = report["cmp_legacy_same_population_model"]
    assert exact["count"] == legacy["count"] == 1
    assert exact["minADE_1s"] == legacy["minADE_1s"] == 5.5
    assert exact["minFDE_1s"] == 10.0
    assert legacy["minFDE_1s"] == 11.0
    assert exact["minADE_3s"] == legacy["minADE_3s"] == 15.5
    assert exact["minFDE_3s"] == 30.0
    assert legacy["minFDE_3s"] == 31.0
    assert exact["minADE_5s"] == 25.5
    assert legacy["minADE_5s"] == 25.0
    assert exact["minFDE_5s"] == legacy["minFDE_5s"] == 50.0
