"""UniAD's official evaluator, with an explicit V2V4Real horizon adapter.

No model, loss, causal training assignment, or historical evaluator is changed.
The external Apache-2.0 reference is hash-pinned rather than reimplemented.
"""
import hashlib
import importlib.util
import os
import sys
import types
from functools import lru_cache
from pathlib import Path

import numpy as np

PROTOCOL = "uniad_distance_v1"
PRIMARY_METRICS = tuple("%s_%ds" % (kind, seconds)
    for kind in ("minADE", "minFDE", "MR_max_2m") for seconds in (1, 3, 5)) + ("EPA_5s",)
REFERENCE_FILES = {
    "metric_utils.py": "d2c2dfceec1acc086a3e3161478035e0b61546ea5d77099be1749079fef9938f",
    "eval_utils.py": "846e115bcfc7ef198d9d4367d74568f5e1eac0fd69f7ec92b93e0e2b1c604783",
    "nuscenes_eval_motion.py": "9649af875089ae785901da58b4f2d1ccfdcde241e5283fd064554ef6343c1e53",
}
CONTRACT = {
    "protocol": PROTOCOL, "reference_commit": "609ee083ea51c3521c323f1279dfc4cee0e60467",
    "reference": "https://github.com/OpenDriveLab/UniAD",
    "conditional_center_distance_m": 1.0, "epa_center_distance_m": 2.0,
    "future_fde_threshold_m": 2.0, "mr_max_displacement_threshold_m": 2.0,
    "epa_false_positive_weight": 0.5, "epa_denominator_epsilon": 1e-5,
    "distance_comparison": "strict_less_than", "mr_comparison": "strict_greater_than",
    "range_m": 50.0, "max_predictions_per_frame": 500,
    "modes": [1, 6], "min_recall": 0.1, "min_precision": 0.1,
    "mode_selection": "independent_minADE_minFDE; MR=min_mode(max_valid_time_error)",
    "matching": "score_ordered_nearest_unused_current_xy; no causal GT ownership",
    "aggregation": "official_101_recall_bin_calc_tp_and_calc_ap",
    "frame_period_s": 0.1, "horizons_s": [1, 3, 5],
    "population_adapter": "current annotated vehicles; complete GT prefix per exact horizon; ignore predictions closest to unscorable GT",
    "dataset_adapter": "V2V4Real 10Hz 1/3/5s, not the official nuScenes benchmark; point-count filtering unavailable",
    "empty_match_adapter": "retain GT and FP counts; null displacement errors instead of upstream 1.0 no-match sentinel",
}


@lru_cache(maxsize=1)
def official_reference():
    root = Path(os.environ.get("DUALCMP_UNIAD_ROOT", str(
        Path(__file__).resolve().parents[4] / "Reference" / "Code" / "UniAD")))
    directory = root / "projects/mmdet3d_plugin/datasets/eval_utils"
    for name, expected in REFERENCE_FILES.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("UniAD reference changed: " + name)
    package = "_dualcmp_official_uniad_eval"
    module = types.ModuleType(package)
    module.__path__ = [str(directory)]
    sys.modules[package] = module
    name = package + ".eval_utils"
    spec = importlib.util.spec_from_file_location(name, str(directory / "eval_utils.py"))
    reference = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference
    spec.loader.exec_module(reference)
    return reference


def forecast_frame(output, target, row, key):
    """Export *all* current GT and emitted forecasts before metric matching."""
    forecast = output["forecast"]
    if forecast.get("forecast_coordinate_frame") != "current_receiver" or target.get("forecast_coordinate_frame") != "current_receiver":
        raise ValueError("mainstream evaluation requires current_receiver coordinates")
    gt = target["current_mask"][row].bool()
    pred = forecast["forecast_valid_mask"][row].bool()
    def array(value):
        return value.detach().cpu().numpy().copy()
    pred_centers = array(output["track_box"][row, pred, :2])
    gt_centers = array(target["current_boxes"][row, gt, :2])
    return {"key": tuple(key), "gt_ids": array(target["actor_ids"][row, gt]),
            "gt_centers": gt_centers,
            "gt_future": array(target["future_xy"][row, gt]) + gt_centers[:, None],
            "gt_mask": array(target["future_mask"][row, gt]).astype(bool),
            "pred_ids": array(output["state"].runtime_id[row, pred]),
            "pred_centers": pred_centers,
            "pred_future": array(forecast["pred_trajs"][row, pred, :, :, :2]) + pred_centers[:, None, None],
            "pred_scores": array(forecast["pred_scores"][row, pred]),
            "detection_scores": array(output["track_exist_logits"][row, pred].sigmoid())}


def _validated_frames(frames):
    unique = {}
    for frame in frames:
        key = tuple(frame["key"])
        if len(key) != 3:
            raise ValueError("frame key must contain scene, receiver, frame")
        n, p = len(frame["gt_ids"]), len(frame["pred_ids"])
        shapes = {"gt_centers": (n, 2), "gt_future": (n, 50, 2),
                  "gt_mask": (n, 50), "pred_centers": (p, 2),
                  "pred_future": (p, 6, 50, 2), "pred_scores": (p, 6),
                  "detection_scores": (p,)}
        for field, shape in shapes.items():
            value = np.asarray(frame[field])
            if value.shape != shape:
                raise ValueError("invalid forecast frame shape: " + field)
            if field != "gt_mask" and not np.isfinite(value).all():
                raise ValueError("nonfinite forecast frame: " + field)
        if len(set(frame["gt_ids"].tolist())) != n or len(set(frame["pred_ids"].tolist())) != p:
            raise ValueError("duplicate current GT/runtime identity")
        if p > CONTRACT["max_predictions_per_frame"]:
            raise ValueError("too many predictions for official evaluator")
        for field in ("pred_scores", "detection_scores"):
            if ((frame[field] < 0) | (frame[field] > 1)).any():
                raise ValueError("metric confidences must be probabilities")
        if key in unique:
            if any(not np.allclose(frame[field], unique[key][field], rtol=0, atol=1e-5)
                   for field in shapes) or any(not np.array_equal(frame[field], unique[key][field]) for field in ("gt_ids", "pred_ids")):
                raise ValueError("DDP padded frame predictions disagree")
        else:
            unique[key] = frame
    return [unique[key] for key in sorted(unique)]


def _boxes(frames, steps, modes, distance):
    from nuscenes.eval.common.data_classes import EvalBoxes
    reference = official_reference()
    ground_truth, predictions = EvalBoxes(), EvalBoxes()
    ignored = 0
    for frame in frames:
        token = repr(frame["key"])
        centers = frame["gt_centers"]
        in_range = np.linalg.norm(centers, axis=-1) < CONTRACT["range_m"]
        eligible = in_range & frame["gt_mask"][:, :steps].all(-1)
        def box(center, future, score):
            return reference.DetectionMotionBox(sample_token=token,
                translation=(float(center[0]), float(center[1]), 0.0), size=(1.0, 1.0, 1.0),
                rotation=(1.0, 0.0, 0.0, 0.0), velocity=(0.0, 0.0),
                detection_name="car", detection_score=float(score), traj=future)
        ground_truth.add_boxes(token, [box(centers[i], frame["gt_future"][i, :steps], -1)
                                      for i in np.flatnonzero(eligible)])
        boxes = []
        for i, center in enumerate(frame["pred_centers"]):
            if np.linalg.norm(center) >= CONTRACT["range_m"]:
                continue
            if len(centers):
                distances = np.linalg.norm(centers - center, axis=-1)
                nearest = int(distances.argmin())
                if distances[nearest] < distance and not eligible[nearest]:
                    ignored += 1
                    continue
            order = np.argsort(-frame["pred_scores"][i], kind="stable")[:modes]
            boxes.append(box(center, frame["pred_future"][i, order, :steps], frame["detection_scores"][i]))
        predictions.add_boxes(token, boxes)
    return ground_truth, predictions, ignored


def _accumulate(frames, steps, modes, distance, joint=False):
    from nuscenes.eval.common.utils import center_distance
    gt, pred, ignored = _boxes(frames, steps, modes, distance)
    reference = official_reference()
    if joint:
        md, tp, fp, _ = reference.accumulate_motion(gt, pred, "car", center_distance,
            sys.modules[reference.__package__ + ".metric_utils"].traj_fde,
            distance, CONTRACT["future_fde_threshold_m"], final_step=steps)
    else:
        md, tp, fp, _ = reference.accumulate(gt, pred, "car", center_distance, distance)
    # Upstream returns 0,0,0 when no pair matches. Those are not real counts.
    n_gt, n_pred = len(gt.all), len(pred.all)
    if not tp:
        fp = n_pred
    return md, {"gt": n_gt, "predictions": n_pred, "tp": int(tp), "fp": int(fp),
                "fn": n_gt - int(tp), "ignored_unscorable_predictions": ignored}


def mainstream_report(frames):
    """Run the pinned official formulas; do not infer GT from admitted rows."""
    from nuscenes.eval.detection.algo import calc_ap, calc_tp
    frames = _validated_frames(frames)
    model, top1, horizons = {}, {}, {}
    for seconds in CONTRACT["horizons_s"]:
        steps = int(round(seconds / CONTRACT["frame_period_s"]))
        populations = {}
        for modes, destination in ((6, model), (1, top1)):
            md, counts = _accumulate(frames, steps, modes, CONTRACT["conditional_center_distance_m"])
            usable = counts["tp"] > 0 and md.max_recall_ind >= 11
            for name, field in (("minADE", "min_ade_err"), ("minFDE", "min_fde_err"), ("MR_max_2m", "miss_rate_err")):
                destination["%s_%ds" % (name, seconds)] = calc_tp(md, 0.1, field) if usable else None
            destination["count_%ds" % seconds] = counts["tp"]
            populations["top%d" % modes] = dict(counts, metric_status="valid" if usable else "insufficient_matched_recall")
        md, joint_counts = _accumulate(frames, steps, 6, 1.0, joint=True)
        detection_md, detection_counts = _accumulate(frames, steps, 6, 2.0)
        _, epa_joint = _accumulate(frames, steps, 6, 2.0, joint=True)
        n_gt = detection_counts["gt"]
        model["EPA_%ds" % seconds] = ((epa_joint["tp"] - 0.5 * detection_counts["fp"]) / (n_gt + 1e-5) if n_gt else None)
        model["minFDE_AP_%ds" % seconds] = calc_ap(md, 0.1, 0.1) if joint_counts["gt"] else None
        horizons[str(seconds) + "s"] = dict(populations,
            minFDE_AP=joint_counts, epa_detection=detection_counts, epa_joint=epa_joint)
    return {"evaluation_protocol": dict(CONTRACT, reference_files_sha256=REFERENCE_FILES),
            "metric_version": PROTOCOL, "model": model, "top1": top1,
            "coverage": {"prediction_keys": len(frames), "horizons": horizons},
            "metric_scope": "UniAD official distance/formulas; declared V2V4Real horizon and GT-eligibility adapter"}
