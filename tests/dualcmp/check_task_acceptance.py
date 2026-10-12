"""Frozen acceptance gates; do not loosen thresholds after observing results."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def assert_all_forecast_horizons(evaluation):
    metrics = evaluation["diagnostics"]["exact_horizon_model"]
    thresholds = {"minADE_1s": .5, "minFDE_1s": .75,
                  "minADE_3s": .75, "minFDE_3s": 1.25,
                  "minADE_5s": 1., "minFDE_5s": 1.5, "MR_3.6": .1}
    assert evaluation["forecast_association"]["policy"] == "iou_causal_v1"
    assert evaluation["coverage"]["prediction_keys"] == 4
    assert metrics["count"] == evaluation["coverage"]["eligible_complete"] >= 10
    for key, limit in thresholds.items():
        assert metrics[key] is not None and math.isfinite(metrics[key]) and metrics[key] <= limit, (key, metrics)
        assert math.isfinite(evaluation["model"][key]), (key, evaluation["model"])
    failures = evaluation["association"]
    assert failures["eligible_complete"] + failures["missed_complete"] == failures["complete"] == 14
    assert failures["matched_current"] + failures["unmatched_predictions"] == failures["predicted_forecasts"]
    assert failures["matched_current"] + failures["unmatched_current_gt"] == failures["current_gt"]
    assert failures["forecast_precision"] >= .5 and failures["current_actor_recall"] >= .5, failures
    return thresholds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("legacy", "learning", "final"), required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--training-directory", default="runs/accept-source-aligned-20260930")
    parser.add_argument("--joint-directory", default="runs/accept-source-aligned-joint-20260930")
    parser.add_argument("--all-horizons", action="store_true", help="full repaired-pipeline quality and coverage gate")
    args = parser.parse_args()
    root = Path(args.directory)
    if args.phase == "legacy":
        for name in ("source", "fused"):
            before = read("runs/audit-task-fitness-20260930/" + name + ".json")
            after = read(root / ("legacy_" + name + ".json"))
            for key, value in before["variants"]["normal"]["model"].items():
                assert abs(after["variants"]["normal"]["model"][key] - value) < 1e-6, (name, key)
            assert all(item["clip_online_max_abs_error"] < 1e-3 for item in after["streaming"]), after["streaming"]
            if name == "source":
                assert all(item["source_times"] == 11 for item in after["streaming"])
                assert all(sum(sum(source) for source in item["attention_mass"]) > 0 for item in after["diagnostics"])
        print("legacy checkpoint, metric, attention and online-equivalence gates passed")
    else:
        train = read(Path(args.training_directory) / "report.json")
        evaluation = read(root / "train_eval.json")
        metrics = evaluation["diagnostics"]["exact_horizon_model"]
        assert train["steps"] == 400 and train["ddp_sync"] and train["global_batch"] == 4
        assert train["source_temporal_grad_norm_min_rank"] > 0
        assert train["source_query_fusion_grad_norm_min_rank"] > 0
        assert train["persistent_query_position_grad_norm_min_rank"] > 0
        assert evaluation["sequence_policy"]["sha256"] == train["provenance"]["sequence_policy"]["sha256"]
        assert evaluation["sequence_policy"]["label_identity"] == train["provenance"]["sequence_policy"]["label_identity"]
        assert evaluation["coverage"]["eligible_complete"] >= 10
        assert evaluation["coverage"]["complete"] == 14
        assert metrics["minADE_5s"] <= 1.0 and metrics["minFDE_5s"] <= 1.5 and metrics["MR_3.6"] <= .1, metrics
        if args.all_horizons:
            assert_all_forecast_horizons(evaluation)
        if args.phase == "final":
            audit = read(root / "aligned.json")
            assert all(item["source_times"] == 11 and item["clip_online_max_abs_error"] < 1e-3
                       for item in audit["streaming"]), audit["streaming"]
            assert all(item["boundary_stream_max_history"] <= 4 and item["boundary_stream_online_error"] < 1e-3
                       for item in audit["streaming"]), audit["streaming"]
            phase_c = read(Path(args.joint_directory) / "report.json")
            assert phase_c["steps"] == 1 and phase_c["ddp_sync"]
            assert phase_c["forecast_to_evidence_bev_grad_norm_min_rank"] > 0
            assert phase_c["joint_detector_grad_norm_min_rank"] > 0
            joint_eval = read(root / "joint_eval.json")
            assert joint_eval["coverage"]["complete"] == 14 and joint_eval["coverage"]["eligible_complete"] >= 10
            joint_metrics = joint_eval["diagnostics"]["exact_horizon_model"]
            assert joint_metrics["minADE_5s"] <= 1.0 and joint_metrics["minFDE_5s"] <= 1.5 and joint_metrics["MR_3.6"] <= .1, joint_metrics
            if args.all_horizons:
                assert_all_forecast_horizons(joint_eval)
        gate = {"status": "passed", "phase": args.phase, "overfit_exact_horizon_metrics": metrics,
                "training_global_batch": train["global_batch"], "peak_memory_mib": train["peak_memory_mib"]}
        if args.all_horizons:
            gate["thresholds"] = assert_all_forecast_horizons(evaluation)
            gate["association"] = evaluation["association"]
        (root / ("acceptance_" + args.phase + ".json")).write_text(json.dumps(gate, indent=2))
        print(json.dumps(gate))


if __name__ == "__main__":
    main()
