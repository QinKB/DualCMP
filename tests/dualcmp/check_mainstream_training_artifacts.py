"""Accept native checkpoint -> independent inference -> complete JSON/CSV."""
import argparse
import csv
import json
import math
from pathlib import Path

from opencood.tools.dualcmp.mainstream_forecast_metrics import CONTRACT as UNIAD_CONTRACT, PRIMARY_METRICS as UNIAD_METRICS
from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import CONTRACT as CMP_CONTRACT, PRIMARY_METRICS as CMP_METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--fresh-report", required=True)
    parser.add_argument("--joint", action="store_true")
    args = parser.parse_args()
    directory = Path(args.directory)
    report = json.loads((directory / "test_forecast_epoch_001.json").read_text())
    fresh = json.loads(Path(args.fresh_report).read_text())
    assert report.get("metric_version") in (CMP_CONTRACT["protocol"], UNIAD_CONTRACT["protocol"])
    assert report["metric_version"] == report["evaluation_protocol"]["protocol"]
    assert fresh["metric_version"] == fresh["evaluation_protocol"]["protocol"]
    cmp_adapter = report.get("metric_version") == CMP_CONTRACT["protocol"]
    contract = CMP_CONTRACT if cmp_adapter else UNIAD_CONTRACT
    primary_metrics = CMP_METRICS if cmp_adapter else UNIAD_METRICS
    prefix = "test_forecast_" if args.joint else "test_"
    with (directory / "metrics.csv").open() as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    epoch = [row for row in rows if row["kind"] == "epoch"]
    assert len(epoch) == 1 and int(epoch[0]["global_step"]) == 1
    forecast_fields = [key for key in fields if key.startswith(prefix) and (
        not args.joint or key.startswith("test_forecast_"))]
    assert forecast_fields == [prefix + key for key in primary_metrics]
    assert all(report["evaluation_protocol"][key] == value for key, value in contract.items())
    assert report["metric_version"] == fresh["metric_version"]
    assert report["world_size"] == fresh["world_size"]
    assert report["coverage"]["prediction_keys"] == fresh["coverage"]["prediction_keys"] == 16
    differences = {}
    for key in primary_metrics:
        value = report["model"][key]
        assert value is not None and math.isfinite(value), (key, report["coverage"])
        assert abs(float(epoch[0][prefix + key]) - value) < 1e-12
        differences[key] = abs(fresh["model"][key] - value)
        assert differences[key] <= 1e-4, (key, differences)
    if cmp_adapter:
        counts = report["coverage"]
        assert counts == fresh["coverage"]
        assert counts["current_gt"] > 0 and counts["tp"] + counts["fn"] == counts["current_gt"]
        assert counts["tp"] + counts["fp"] == counts["predictions"]
        assert report["model"]["count"] == counts["eligible_complete"] > 0
        assert counts["tp"] == counts["eligible_complete"] + counts["rejected_incomplete_future"] + counts["rejected_short_history"]
        assert counts["missed_complete"] == counts["complete_gt"] - counts["eligible_complete"]
        assert all("MR_" + threshold in group for group in (report["model"], report["top1"])
                   for threshold in ("3.6", "3.0", "2.0", "1.8"))
    for seconds in (() if cmp_adapter else (1, 3, 5)):
        suffix = "%ds" % seconds
        counts = report["coverage"]["horizons"][suffix]
        assert counts == fresh["coverage"]["horizons"][suffix]
        det, joint = counts["epa_detection"], counts["epa_joint"]
        assert det["gt"] > 0 and det["tp"] + det["fn"] == det["gt"]
        assert det["tp"] + det["fp"] == det["predictions"]
        assert abs(report["model"]["EPA_" + suffix] - (joint["tp"] - .5 * det["fp"]) / (det["gt"] + 1e-5)) < 1e-12
        for metrics in (report["model"], report["top1"]):
            assert all(key + "_" + suffix in metrics for key in ("minADE", "minFDE", "MR_max_2m"))
        assert "minFDE_AP_" + suffix in report["model"]
    log = (directory / "job.log").read_text()
    assert '"kind": "forecast_evaluation"' in log and '"evaluation_protocol"' in log
    result = {"status": "passed", "scope": "evaluation and logging integration, not model-quality superiority",
        "model": report["model"], "fresh_inference_max_difference": max(differences.values()),
        "csv_forecast_fields": forecast_fields, "coverage": report["coverage"], "resources": fresh["resources"]}
    (directory / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
