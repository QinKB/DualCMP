"""Frozen real-model learnability gate, not a held-out superiority claim."""
import argparse
import csv
import json
import math
from pathlib import Path

from opencood.tools.dualcmp.cmp_adapted_forecast_metrics import CONTRACT, PRIMARY_METRICS

THRESHOLDS = {"minADE_1s": .5, "minFDE_1s": .75,
              "minADE_3s": .75, "minFDE_3s": 1.25,
              "minADE_5s": 1., "minFDE_5s": 1.5, "MR_2.0": .1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    root = Path(args.directory)
    train = json.loads((root / "report.json").read_text())
    evaluation = json.loads((root / "train_eval.json").read_text())
    assert train["steps"] == 400 and train["global_batch"] == 4 and train["ddp_sync"]
    assert train["source_feature_encoder"] == "motion_correlation_v1"
    assert train["source_fusion_location"] == "actor_pre_scene_and_decoder"
    assert train["source_actor_readout_grad_norm_min_rank"] > 0
    if train["provenance"].get("source_motion_contract", "").endswith("norm_v5"):
        assert train["source_actor_normalization_grad_norm_min_rank"] > 0
    assert train["source_query_fusion_grad_norm_min_rank"] > 0
    assert all(value > 0 and math.isfinite(value) for value in train["source_motion_gradient_min_rank"])
    assert train["forecast_mean_weight"] == 5 and train["provenance"]["initial_lr"] == .001
    schedule = train["provenance"]["lr_schedule"]
    assert schedule in ("constant_v1", "cmp_step_v1")
    assert train["last_lr"] == (.001 if schedule == "constant_v1" else .001 / 16)
    if train["provenance"].get("refresh_forecast_bn"):
        calibration = train["provenance"]["forecast_bn_calibration"]
        assert calibration["global_calibration_batch"] == calibration["samples"] == 4
        assert calibration["future_gt_used"] is False and calibration["optimizer_steps"] == 0
        assert calibration["parameters_bitwise_unchanged"] and calibration["non_head_buffers_unchanged"]
        assert calibration["updated_layers"] > 0 and calibration["unvisited_active_layers"] == 0
        assert evaluation["checkpoint_provenance"]["source_tree_sha256"] == train["provenance"]["source_tree_sha256"]
    assert evaluation["evaluation_protocol"] == CONTRACT
    assert evaluation["coverage"]["prediction_keys"] == evaluation["dataset_samples"] == 4
    assert evaluation["coverage"]["eligible_complete"] == evaluation["model"]["count"] >= 10
    assert evaluation["iou_only_adapter"]["official_max_difference"] <= 1e-5
    assert evaluation["coverage"]["complete_gt"] == 14
    for key, limit in THRESHOLDS.items():
        value = evaluation["model"][key]
        assert value is not None and math.isfinite(value) and value <= limit, (key, value, limit)
    with (root / "train_eval.csv").open() as stream:
        fields = csv.DictReader(stream).fieldnames
    assert tuple(fields[-7:]) == PRIMARY_METRICS
    result = {"status": "passed", "thresholds": THRESHOLDS, "model": evaluation["model"],
              "coverage": evaluation["coverage"], "global_batch": train["global_batch"],
              "resources": {"training_peak_mib": train["peak_memory_mib"],
                            "clips_per_second_per_rank": train["clips_per_second_per_rank"]},
              "scope": "four-sample training/checkpoint/fresh-inference learnability; not generalization"}
    (root / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
