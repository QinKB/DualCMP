"""Verify final native inference artifacts, not just isolated metric functions."""
import argparse
import csv
import json
import math
from pathlib import Path

from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import (
    CONTRACT, LEGACY_CONTRACT, LEGACY_PRIMARY_METRICS, MR_METRICS, PRIMARY_METRICS,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--completion", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iou05-baseline", help="Prior IoU0.7 full report for the same checkpoint")
    args = parser.parse_args()
    path = Path(args.report)
    report = json.loads(path.read_text())
    completion = json.loads(Path(args.completion).read_text())
    assert completion["exit_code"] == 0
    from eval_cmp_iou05_ddp import HISTORICAL_IOU05_CONTRACT, IOU05_CONTRACT
    protocol = report["metric_version"]
    if protocol == CONTRACT["protocol"]:
        assert report["evaluation_protocol"] == CONTRACT
        report_primary = PRIMARY_METRICS
    elif protocol == LEGACY_CONTRACT["protocol"]:
        assert report["evaluation_protocol"] == LEGACY_CONTRACT
        report_primary = LEGACY_PRIMARY_METRICS
    elif protocol == HISTORICAL_IOU05_CONTRACT["protocol"]:
        assert report["evaluation_protocol"] == HISTORICAL_IOU05_CONTRACT
        report_primary = LEGACY_PRIMARY_METRICS
    else:
        raise AssertionError("unsupported CMP evaluation protocol: " + str(protocol))
    if args.iou05_baseline:
        assert report["evaluation_protocol"] in (IOU05_CONTRACT, HISTORICAL_IOU05_CONTRACT)
        baseline = json.loads(Path(args.iou05_baseline).read_text())
        control = report["iou07_control"]
        assert control["evaluation_protocol"] == LEGACY_CONTRACT
        assert baseline["evaluation_protocol"] == LEGACY_CONTRACT
        assert report["checkpoint_provenance"]["checkpoint_sha256"] == baseline["checkpoint_provenance"]["checkpoint_sha256"]
        assert control["coverage"] == baseline["coverage"]
        for name in LEGACY_PRIMARY_METRICS + tuple(
                key for key in MR_METRICS if key not in LEGACY_PRIMARY_METRICS):
            assert abs(control["model"][name] - baseline["model"][name]) <= 1e-5
        assert report["iou_only_adapter"]["official_max_difference"] <= 1e-5
        adapter = report["iou_only_adapter"]
        assert adapter.get("production_sources_mutated_at_runtime",
                           adapter.get("production_sources_changed")) is False
        cache = report["prediction_cache"]
        import hashlib
        assert cache["frames"] == 2906
        assert hashlib.sha256(Path(cache["path"]).read_bytes()).hexdigest() == cache["sha256"]
    assert report["full_test"] and report["split"] == "test"
    assert report["dataset_samples"] == report["coverage"]["prediction_keys"] == 2906
    assert report["checkpoint_epoch"] == 10 and report["checkpoint_step"] == 25551
    assert report["checkpoint_training_evaluation_protocol"]["protocol"] == "uniad_distance_v1"
    assert report["original_checkpoint_sha256"] == "e09ad3ba79e29d072e43614c7f80d70d94e10228318a79e89f0b04dfac499495"
    assert "EPA_5s" not in report["model"]
    cov, model, top1 = report["coverage"], report["model"], report["top1"]
    assert cov["tp"] + cov["fp"] == cov["predictions"]
    assert cov["tp"] + cov["fn"] == cov["current_gt"]
    assert cov["eligible_complete"] == model["count"] == top1["count"] > 0
    assert model["count"] + cov["rejected_incomplete_future"] + cov["rejected_short_history"] == cov["tp"]
    assert cov["missed_complete"] == cov["complete_gt"] - model["count"]
    for name in report_primary:
        assert math.isfinite(model[name]) and model[name] >= 0
        assert top1[name] + 1e-8 >= model[name]
    for group in (model, top1):
        rates = [group[name] for name in MR_METRICS]
        assert all(0 <= rate <= 1 for rate in rates) and rates == sorted(rates)
    with path.with_suffix(".csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["evaluation_protocol", "checkpoint_epoch", "checkpoint_step"] + list(report_primary)
        rows = list(reader)
    assert len(rows) == 1 and rows[0]["evaluation_protocol"] == report["metric_version"]
    for name in report_primary:
        assert abs(float(rows[0][name]) - model[name]) < 1e-12
    result = {"status": "passed", "scope": "native full-test inference, CMP-adapter protocol and JSON/CSV integration",
        "evaluation_protocol": report["evaluation_protocol"],
        "checkpoint_epoch": report["checkpoint_epoch"], "checkpoint_step": report["checkpoint_step"],
        "original_checkpoint_sha256": report["original_checkpoint_sha256"],
        "model": model, "coverage": cov, "resources": report["resources"],
        "complete_future_coverage": model["count"] / cov["complete_gt"],
        "not_claimed": ["official CMP end-to-end equivalence", "leakage-free split", "new training or overfit acceptance", "Phase-C CUDA acceptance"]}
    destination = Path(args.output)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
