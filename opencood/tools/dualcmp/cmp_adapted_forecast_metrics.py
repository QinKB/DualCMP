"""CMP forecast metrics adapted to post-model GT pairing in DualCMP."""

import ast
import copy
import hashlib
import importlib.util
import os
import types
from functools import lru_cache
from pathlib import Path

import numpy as np

LEGACY_PROTOCOL = "cmp_gtfree_adapter_v1"
PROTOCOL = "cmp_gtfree_adapter_iou05_mr2_v1"
PRIMARY_METRICS = ("minADE_1s", "minADE_3s", "minADE_5s",
                   "minFDE_1s", "minFDE_3s", "minFDE_5s", "MR_2.0")
LEGACY_PRIMARY_METRICS = ("minADE_1s", "minADE_3s", "minADE_5s",
                          "minFDE_1s", "minFDE_3s", "minFDE_5s", "MR_3.6")
MR_METRICS = ("MR_3.6", "MR_3.0", "MR_2.0", "MR_1.8")
REFERENCE_FILES = {
    "MTR/tools/eval_utils/eval_utils.py": "6af8a7a5415d5a892eb5eeeb60d2b4509aee92783f6c74d541efe4117ba60259",
    "opencood/utils/eval_utils.py": "26018aa01612b53caee79f7aca160132116168a8fa6d58ea168a3911df56b545",
    "opencood/utils/common_utils.py": "a6958c7780e2508398608081b266d280bdabdd705346c513e37074d8af19951f",
}
LEGACY_CONTRACT = {
    "protocol": LEGACY_PROTOCOL,
    "reference": "CMP",
    "reference_commit": "de0f83900abaff47d87bb7b8329a5ff4a369a49d",
    "reference_metric": "MTR/tools/eval_utils/eval_utils.py::eval_one_epoch_custom",
    "reference_files_sha256": REFERENCE_FILES,
    "pairing": "after model output only; every emitted prediction is matched against current native GT",
    "matching": "official score-descending greedy BEV IoU >= 0.7; no future/history eligibility filtering before matching",
    "matching_iou_threshold": 0.7,
    "matching_adapter": "official matched GT IDs are score-ordered; pair each ID with the prediction slot in the same official score order (fixes upstream generator ID-order mismatch)",
    "native_adaptation": "current_receiver absolute XY; GT class and boxes from native current targets; no GT in model forward; no extra 50m radial crop",
    "population": "one-to-one current GT/pred matches with complete_future50 and valid_history11",
    "future_steps": 50,
    "history_steps": 11,
    "frame_period_s": 0.1,
    "modes": 6,
    "mr_thresholds_m": [3.6, 3.0, 2.0, 1.8],
    "missed_target_rule": "excluded from displacement-error denominator; retained in coverage counts; no EPA or synthetic error penalty",
    "history_rule": "forecast_history_mask has 11 valid steps; Kalman coast is allowed, measured 11-frame history is not required",
    "horizon_indices": {"1s": 10, "3s": 30, "5s": 49},
    "ade_5s_indices": "[0:49], as in CMP official evaluator",
    "mode_selection": "independent minimum per horizon and per metric, as in CMP official evaluator",
    "aggregation": "per eligible matched target average",
    "miss_rate": "final displacement strictly greater than threshold; official CMP formula",
    "empty_population": "null metrics and count 0; official metric function is not called",
    "scope": "adapted CMP forecast metrics; not official CMP end-to-end evaluation",
}
CONTRACT = copy.deepcopy(LEGACY_CONTRACT)
CONTRACT.update(protocol=PROTOCOL,
    matching="official score-descending greedy BEV IoU >= 0.5; no future/history eligibility filtering before matching",
    matching_iou_threshold=0.5,
    headline_mr_threshold_m=2.0,
    scope="IoU0.5 adapted CMP forecast metrics; not official CMP end-to-end evaluation")


def _validated_iou_threshold(iou_threshold):
    if (isinstance(iou_threshold, (bool, np.bool_)) or
            not isinstance(iou_threshold, (int, float, np.integer, np.floating)) or
            not np.isfinite(iou_threshold) or float(iou_threshold) not in (0.5, 0.7)):
        raise ValueError("matching IoU threshold must be 0.5 or 0.7")
    return float(iou_threshold)


def _reference_root():
    return Path(os.environ.get("DUALCMP_CMP_ROOT", str(
        Path(__file__).resolve().parents[4] / "Reference" / "Code" / "CMP")))


def _check_references():
    root = _reference_root()
    for relative, expected in REFERENCE_FILES.items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("CMP reference changed: " + relative)
    return root


@lru_cache(maxsize=1)
def _official_functions():
    """Load pinned official functions without importing their licensed modules."""
    import torch
    root = _check_references()
    function_path = root / "MTR/tools/eval_utils/eval_utils.py"
    tree = ast.parse(function_path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name == "eval_one_epoch_custom")
    reference_node = copy.deepcopy(node)
    # Expose the original accumulators at full precision; the official public
    # return omits the shorter horizons and three MR thresholds.
    node.body[-1] = ast.Return(value=ast.Dict(
        keys=[ast.Constant(k) for k in (
            "minADE_1s", "minADE_3s", "minADE_5s", "minFDE_1s",
            "minFDE_3s", "minFDE_5s", "MR_3.6", "MR_3.0", "MR_2.0",
            "MR_1.8", "count")],
        values=[ast.BinOp(ast.Name(n, ast.Load()), ast.Div(), ast.Name("trajectories_count", ast.Load()))
                for n in ("total_distance_error_1s", "total_distance_error_3s",
                          "total_distance_error", "total_final_distance_error_1s",
                          "total_final_distance_error_3s", "total_final_distance_error",
                          "total_miss_count_3_6", "total_miss_count_3",
                          "total_miss_count_2", "total_miss_count_1_8")]
                + [ast.Name("trajectories_count", ast.Load())]))
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"np": np, "torch": torch, "tqdm": types.SimpleNamespace(tqdm=lambda x: x),
                 "OPV2VMultiEgoDataset": types.SimpleNamespace(
                     generate_prediction_dicts=lambda rows, output_path=None: [rows])}
    reference_namespace = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[reference_node], type_ignores=[])),
                 str(function_path), "exec"), reference_namespace)
    exec(compile(module, str(function_path), "exec"), namespace)

    eval_path = root / "opencood/utils/eval_utils.py"
    eval_tree = ast.parse(eval_path.read_text())
    match_node = next(item for item in eval_tree.body if isinstance(item, ast.FunctionDef)
                      and item.name == "caluclate_tp_fp")
    match_module = ast.fix_missing_locations(ast.Module(body=[match_node], type_ignores=[]))
    common_path = root / "opencood/utils/common_utils.py"
    spec = importlib.util.spec_from_file_location("_dualcmp_cmp_common_utils", str(common_path))
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    match_namespace = {"np": np, "common_utils": common}
    exec(compile(match_module, str(eval_path), "exec"), match_namespace)
    return (namespace["eval_one_epoch_custom"], match_namespace["caluclate_tp_fp"],
            reference_namespace["eval_one_epoch_custom"])


def _array(value):
    return value.detach().cpu().numpy().copy()


def forecast_frame(output, target, row, key):
    """Export all native current GT and every emitted forecast for later pairing."""
    forecast = output["forecast"]
    if forecast.get("forecast_coordinate_frame") != "current_receiver":
        raise ValueError("CMP adapter requires current_receiver forecasts")
    if target.get("forecast_coordinate_frame") != "current_receiver":
        raise ValueError("CMP adapter requires current_receiver GT")
    gt_mask = target["current_mask"][row].bool()
    pred_mask = forecast["forecast_valid_mask"][row].bool()
    gt_centers = _array(target["current_boxes"][row, gt_mask])
    pred_boxes = _array(output["track_box"][row, pred_mask])
    future_relative = _array(target["future_xy"][row, gt_mask])
    gt_future = future_relative + gt_centers[:, None, :2]
    predicted = _array(forecast["pred_trajs"][row, pred_mask, :, :, :2])
    pred_centers = pred_boxes[:, :2]
    return {
        "key": tuple(key),
        "gt_ids": _array(target["actor_ids"][row, gt_mask]),
        "gt_boxes": gt_centers,
        "gt_future": gt_future,
        "gt_mask": _array(target["future_mask"][row, gt_mask]).astype(bool),
        "pred_ids": _array(output["state"].runtime_id[row, pred_mask]),
        "pred_boxes": pred_boxes,
        "pred_future": predicted + pred_centers[:, None, None, :2],
        "pred_scores": _array(forecast["pred_scores"][row, pred_mask]),
        "detection_scores": _array(output["track_exist_logits"][row, pred_mask].sigmoid()),
        "history_mask": _array(output["state"].forecast_history_mask[row, pred_mask]).astype(bool),
        "history_steps": _array(forecast["forecast_history_steps"][row, pred_mask]),
    }


def _validated_frames(frames):
    unique = {}
    for frame in frames:
        key = tuple(frame["key"])
        if len(key) != 3:
            raise ValueError("frame key must contain scene, receiver, frame")
        n, p = len(frame["gt_ids"]), len(frame["pred_ids"])
        shapes = {"gt_boxes": (n, 7), "gt_future": (n, 50, 2), "gt_mask": (n, 50),
                  "pred_boxes": (p, 7), "pred_future": (p, 6, 50, 2),
                  "pred_scores": (p, 6), "detection_scores": (p,),
                  "history_mask": (p, 11), "history_steps": (p,)}
        for name, shape in shapes.items():
            value = np.asarray(frame[name])
            if value.shape != shape:
                raise ValueError("invalid forecast frame shape: " + name)
            if name not in ("gt_mask", "history_mask") and not np.isfinite(value).all():
                raise ValueError("nonfinite forecast frame: " + name)
        for name in ("gt_mask", "history_mask"):
            if np.asarray(frame[name]).dtype != np.bool_:
                raise ValueError("forecast masks must be boolean: " + name)
        if (np.any(frame["gt_boxes"][:, 3:6] <= 0) or
                np.any(frame["pred_boxes"][:, 3:6] <= 0)):
            raise ValueError("box dimensions must be positive")
        if np.any(frame["history_steps"] < 0) or np.any(frame["history_steps"] > 11):
            raise ValueError("history steps must lie in [0, 11]")
        for name in ("gt_ids", "pred_ids", "history_steps"):
            values = np.asarray(frame[name])
            if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
                raise ValueError("identities/history steps must be nonnegative integers: " + name)
        if not np.array_equal(frame["history_steps"], np.asarray(frame["history_mask"]).sum(-1)):
            raise ValueError("forecast history steps and native history mask disagree")
        if len(set(np.asarray(frame["gt_ids"]).tolist())) != n or len(set(np.asarray(frame["pred_ids"]).tolist())) != p:
            raise ValueError("duplicate current GT/runtime identity")
        if (np.any((frame["pred_scores"] < 0) | (frame["pred_scores"] > 1)) or
                np.any((frame["detection_scores"] < 0) | (frame["detection_scores"] > 1))):
            raise ValueError("metric confidences must be probabilities")
        if key in unique:
            old = unique[key]
            fields = tuple(shapes)
            if (any(not np.array_equal(frame[name], old[name]) for name in ("gt_ids", "pred_ids")) or
                    any(not np.allclose(frame[name], old[name], rtol=0, atol=1e-6)
                        for name in fields if name not in ("gt_ids", "pred_ids"))):
                raise ValueError("DDP padded frame predictions disagree")
        else:
            unique[key] = frame
    return [unique[key] for key in sorted(unique)]


def _corners(boxes):
    import torch
    from opencood.utils.box_utils import boxes_to_corners_3d
    if not len(boxes):
        return torch.empty((0, 8, 3), dtype=torch.float32)
    return boxes_to_corners_3d(torch.as_tensor(boxes, dtype=torch.float32), order="hwl")


def _pair_frames(frames, iou_threshold=0.5):
    iou_threshold = _validated_iou_threshold(iou_threshold)
    _, matcher, _ = _official_functions()
    pairs, coverage = [], {"current_gt": 0, "predictions": 0, "tp": 0,
                           "fp": 0, "fn": 0, "complete_gt": 0,
                           "eligible_complete": 0, "missed_complete": 0,
                           "rejected_incomplete_future": 0,
                           "rejected_short_history": 0}
    for frame in frames:
        gt_ids, pred_ids = np.asarray(frame["gt_ids"]), np.asarray(frame["pred_ids"])
        gt_boxes, pred_boxes = np.asarray(frame["gt_boxes"]), np.asarray(frame["pred_boxes"])
        n, p = len(gt_ids), len(pred_ids)
        coverage["current_gt"] += n
        coverage["predictions"] += p
        stat = {iou_threshold: {"fp": [], "tp": [], "gt": 0, "score": [], "matched_indices": []}}
        import torch
        matcher(_corners(pred_boxes), torch.as_tensor(frame["detection_scores"], dtype=torch.float32),
                _corners(gt_boxes), stat, iou_threshold,
                gt_object_id_tensor=torch.as_tensor(gt_ids))
        matched_ids = stat[iou_threshold]["matched_indices"]
        order = np.argsort(-np.asarray(frame["detection_scores"], dtype=np.float32))
        coverage["tp"] += sum(x >= 0 for x in matched_ids)
        coverage["fp"] += sum(x < 0 for x in matched_ids)
        coverage["fn"] += n - sum(x >= 0 for x in matched_ids)
        gt_slot_by_id = {value.item() if hasattr(value, "item") else value: i
                         for i, value in enumerate(gt_ids)}
        complete = np.asarray(frame["gt_mask"]).all(axis=1)
        coverage["complete_gt"] += int(complete.sum())
        frame_pairs = []
        for pred_slot, gt_id in zip(order, matched_ids):
            if gt_id < 0:
                continue
            gt_slot = gt_slot_by_id[gt_id]
            if not complete[gt_slot]:
                coverage["rejected_incomplete_future"] += 1
                continue
            if not np.asarray(frame["history_mask"])[pred_slot].all() or frame["history_steps"][pred_slot] != 11:
                coverage["rejected_short_history"] += 1
                continue
            coverage["eligible_complete"] += 1
            frame_pairs.append((pred_slot, gt_slot))
            pairs.append((frame, pred_slot, gt_slot))
        coverage["missed_complete"] += int(complete.sum()) - sum(
            1 for pred_slot, gt_slot in frame_pairs if complete[gt_slot])
    return pairs, coverage


def _official_metrics(pairs):
    if not pairs:
        return {key: None for key in PRIMARY_METRICS + tuple(
            key for key in MR_METRICS if key not in PRIMARY_METRICS)}, 0
    evaluator, _, _ = _official_functions()
    records = []
    for pair_index, (frame, pred_slot, gt_slot) in enumerate(pairs):
        records.append({"scenario_name": str(frame["key"][0]),
                        "timestamp_idx": int(frame["key"][2]),
                        "object_id": int(frame["gt_ids"][gt_slot]),
                        "ego_cav_id": str(frame["key"][1]),
                        "gt_trajs": frame["gt_future"][gt_slot],
                        "pred_trajs": frame["pred_future"][pred_slot],
                        "pred_scores": frame["pred_scores"][pred_slot],
                        "center_gt_trajs_mask": frame["gt_mask"][gt_slot],
                        "center_gt_final_valid_idx": 49})
    class Identity:
        def eval(self):
            return None

        def __call__(self, batch):
            return batch

    class Logger:
        def info(self, *args, **kwargs):
            return None

    result = evaluator(Identity(), [records], 0, Logger())
    metrics = dict(result)
    metrics.pop("count")
    return metrics, int(result["count"])


def _top1_diagnostic(pairs):
    top1 = []
    for frame, pred_slot, gt_slot in pairs:
        mode = int(np.argmax(frame["pred_scores"][pred_slot]))
        row = dict(frame)
        row["pred_future"] = np.repeat(frame["pred_future"][pred_slot:pred_slot + 1, mode:mode + 1], 6, axis=1)
        row["pred_scores"] = np.ones((1, 6), dtype=np.float32) / 6
        row["gt_future"] = frame["gt_future"][gt_slot:gt_slot + 1]
        row["gt_mask"] = frame["gt_mask"][gt_slot:gt_slot + 1]
        row["gt_ids"] = frame["gt_ids"][gt_slot:gt_slot + 1]
        row["pred_ids"] = frame["pred_ids"][pred_slot:pred_slot + 1]
        row["gt_boxes"] = frame["gt_boxes"][gt_slot:gt_slot + 1]
        row["pred_boxes"] = frame["pred_boxes"][pred_slot:pred_slot + 1]
        row["history_mask"] = frame["history_mask"][pred_slot:pred_slot + 1]
        row["history_steps"] = frame["history_steps"][pred_slot:pred_slot + 1]
        row["detection_scores"] = frame["detection_scores"][pred_slot:pred_slot + 1]
        top1.append((row, 0, 0))
    return _official_metrics(top1)[0]


def cmp_adapted_report(frames, iou_threshold=0.5):
    """Return adapted CMP forecast metrics after post-output current-box pairing."""
    iou_threshold = _validated_iou_threshold(iou_threshold)
    contract = CONTRACT if iou_threshold == 0.5 else LEGACY_CONTRACT
    protocol = PROTOCOL if iou_threshold == 0.5 else LEGACY_PROTOCOL
    frames = _validated_frames(frames)
    pairs, coverage = _pair_frames(frames, iou_threshold=iou_threshold)
    if pairs:
        model, count = _official_metrics(pairs)
        top1 = _top1_diagnostic(pairs)
    else:
        model = dict((key, None) for key in PRIMARY_METRICS + tuple(
            key for key in MR_METRICS if key not in PRIMARY_METRICS))
        top1 = dict(model)
        count = 0
    model["count"] = count
    top1["count"] = count
    top1["scope"] = "same eligible matched targets; score-argmax diagnostic, not the CMP paper headline"
    if coverage["eligible_complete"] != count:
        raise RuntimeError("eligible-complete coverage must equal metric count")
    if (coverage["tp"] + coverage["fp"] != coverage["predictions"]
            or coverage["tp"] + coverage["fn"] != coverage["current_gt"]
            or count + coverage["rejected_incomplete_future"]
            + coverage["rejected_short_history"] != coverage["tp"]):
        raise RuntimeError("CMP-adapter pairing/population counts do not conserve targets")
    coverage.update({"prediction_keys": len(frames), "forecast_coverage": {
        "emitted_predictions": coverage["predictions"], "matched_complete_history": count,
        "ratio": (count / coverage["predictions"] if coverage["predictions"] else None)},
        "matched_rows_provenance": {
            "frame_key": ["scene", "receiver", "frame"],
            "target_identity": "native gt_ids",
            "prediction_identity": "receiver-local runtime pred_ids",
            "pairing": "pinned CMP IoU matcher GT IDs joined to prediction slots in official score order",
            "eligible_rows": count}})
    return {"evaluation_protocol": dict(contract), "metric_version": protocol,
            "metric_scope": contract["scope"],
            "reference_files_sha256": dict(REFERENCE_FILES), "model": model,
            "top1": top1, "coverage": coverage}
