"""Fresh-process strict-online DualCMP inference over receiver frames.

This entry point consumes only ``runtime_frames`` from the sequence dataset.
Targets are neither constructed nor passed to the model.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import sys

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.data_utils.v2v4real.datasets.dualcmp_target_builder import load_tracking_label_file, transform_tracking_boxes
from opencood.data_utils.v2v4real.post_processor import build_postprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.tools.train_utils import to_device
from opencood.tools.dualcmp.ab3dmot_metrics import tracking_frame_prediction
from opencood.tools.dualcmp.protocol import AUDIT_DENOMINATORS, AUDIT_THRESHOLDS, METRIC_THRESHOLDS, WINDOW, WINDOW_FRAMES, checkpoint_detector_compression

_SHA1_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_sha256():
    return _sha256_file(Path(__file__).resolve().with_name("protocol.py"))


def _checkpoint_provenance(checkpoint, payload=None, allow_dirty_smoke=False):
    """Reject checkpoints that cannot prove their code and protocol origin."""
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu") if payload is None else payload
    if not isinstance(payload, dict) or not isinstance(payload.get("provenance"), dict):
        raise RuntimeError("checkpoint provenance metadata is required for DualCMP evaluation")
    provenance = payload["provenance"]
    if provenance.get("source_state") == "dirty-smoke":
        if not allow_dirty_smoke:
            raise RuntimeError("dirty-smoke checkpoint requires explicit allow_dirty_smoke")
        is_joint = "init_checkpoint_sha256" in provenance
        if is_joint:
            from opencood.tools.dualcmp.train_joint_ddp import _provenance
        else:
            from opencood.tools.dualcmp.train_forecast_ddp import _provenance
        current = _provenance()
        checkpoint_protocol = provenance.get("protocol_sha256")
        if checkpoint_protocol is None and is_joint:
            checkpoint_protocol = provenance.get("init_provenance", {}).get(
                "protocol_sha256"
            )
        current_protocol = _protocol_sha256()
        if (
            current.get("source_state") != "dirty-smoke"
            or any(
                provenance.get(key) != current.get(key)
                for key in ("base_sha", "source_tree_sha256")
            )
            or checkpoint_protocol != current_protocol
        ):
            raise RuntimeError("dirty-smoke checkpoint source manifest does not match current task sources")
        return {"checkpoint_sha256": _sha256_file(checkpoint), "source_state": "dirty-smoke", "base_sha": provenance["base_sha"], "source_tree_sha256": provenance["source_tree_sha256"], "checkpoint_protocol_sha256": checkpoint_protocol, "current_protocol_sha256": current_protocol}
    source_sha = provenance.get("source_sha")
    protocol_sha = provenance.get("protocol_sha256")
    if not isinstance(source_sha, str) or not _SHA1_RE.match(source_sha):
        raise RuntimeError("checkpoint provenance source_sha must be a 40-character hexadecimal SHA")
    if not isinstance(protocol_sha, str) or not re.match(r"^[0-9a-fA-F]{64}$", protocol_sha):
        raise RuntimeError("checkpoint provenance protocol_sha256 must be a SHA256 digest")
    current_protocol_sha = _protocol_sha256()
    if protocol_sha.lower() != current_protocol_sha:
        raise RuntimeError("checkpoint protocol SHA256 does not match the current frozen protocol")
    return {"checkpoint_sha256": _sha256_file(checkpoint),
            "source_sha": source_sha.lower(), "checkpoint_protocol_sha256": protocol_sha.lower(),
            "current_protocol_sha256": current_protocol_sha}


def _model_from_checkpoint(hypes, checkpoint, device, forecast_args=None,
                           allow_dirty_smoke=False):
    postprocessor = build_postprocessor(hypes["postprocess"], train=False)
    target_args = hypes["postprocess"]["target_args"]
    model = DualCmpClipModel(hypes["model"]["args"], postprocessor.generate_anchor_box(),
                             hypes["preprocess"]["cav_lidar_range"],
                             proposal_score_threshold=target_args["score_threshold"],
                             proposal_nms_thresh=hypes["postprocess"]["nms_thresh"],
                             emission_threshold=target_args["score_threshold"],
                             forecast_args=forecast_args)
    payload = torch.load(checkpoint, map_location="cpu")
    _checkpoint_provenance(checkpoint, payload, allow_dirty_smoke=allow_dirty_smoke)
    state_dict = payload.get("model", payload.get("model_state_dict", payload))
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key[7:] if key.startswith("module.") else key: value
                      for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


class OnlineDualCMP(object):
    """Per-receiver state owner with explicit scene-change reset."""

    def __init__(self, model, device):
        self.model, self.device = model, torch.device(device)
        self._states = {}

    @torch.no_grad()
    def step(self, runtime_frame, run_forecast=False):
        """Consume one already-collated runtime frame and emit one prediction."""
        if "target_frames" in runtime_frame or "fused_label_dict" in runtime_frame:
            raise ValueError("online inference accepts runtime data only")
        runtime_frame = to_device(runtime_frame, self.device)
        if len(runtime_frame["scene_id"]) != 1:
            raise ValueError("online inference requires one receiver frame per call")
        key = (str(runtime_frame["scene_id"][0]), int(runtime_frame["receiver_id"][0]))
        state = self._states.get(key)
        outputs, state = self.model([runtime_frame], state=state, run_forecast=run_forecast)
        self._last_output = outputs[-1]
        self._states[key] = state.detached()
        active = state.active_mask[0]
        prediction = tracking_frame_prediction(
            int(runtime_frame["receiver_frame_idx"][0]), state.runtime_id[0, active],
            state.box[0, active], state.score[0, active])
        if run_forecast and "forecast" in outputs[-1]:
            forecast = outputs[-1]["forecast"]; indices = torch.nonzero(active & forecast["forecast_valid_mask"][0], as_tuple=False).flatten()
            prediction["forecast_runtime_ids"] = state.runtime_id[0, indices].tolist()
            prediction["forecast_scores"] = forecast["pred_scores"][0, indices].detach().cpu().tolist()
            prediction["forecast_trajs"] = forecast["pred_trajs"][0, indices, :, :, :2].detach().cpu().tolist()
        return prediction

    def reset_scene(self, scene_id):
        scene_id = str(scene_id)
        self._states = {key: value for key, value in self._states.items() if key[0] != scene_id}


def run_receiver_frames(hypes, checkpoint, scene_id, receiver_id, start_frame, frame_count, device):
    """Load a fresh runtime-only dataset and process contiguous frames once."""
    hypes = dict(hypes)
    hypes["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": {
        "stage": "tracking", "tracking_frames": 3, "runtime_only": True,
        "runtime_forbid_gt": True, "receiver_ids": [int(receiver_id)],
        "frame_period_s": 0.1, "max_actors": 128, "max_cav": 2,
    }}
    dataset = DualCMPSequenceDataset(hypes, train=False)
    keys = dataset.index.receiver_frame_keys(str(scene_id), str(receiver_id))
    selected = [key for key in keys if int(key) >= int(start_frame)][:int(frame_count)]
    if len(selected) != int(frame_count):
        raise ValueError("requested runtime window is not available")
    online = OnlineDualCMP(_model_from_checkpoint(hypes, checkpoint, device), device)
    predictions = []
    for key in selected:
        runtime = dataset._runtime_frame(str(scene_id), str(receiver_id), key)
        predictions.append(online.step(dataset._collate_runtime_frame([runtime])))
    return {(str(scene_id), int(receiver_id)): predictions}


def _ab3d_geometry(repository_root):
    root = Path(repository_root) / "AB3Dmot"
    for path in (str(root), str(root / "Xinshuo_PyToolbox"), str(root / "scripts/KITTI")):
        if path not in sys.path:
            sys.path.insert(0, path)
    from AB3DMOT_libs.box import Box3D
    from AB3DMOT_libs.dist_metrics import iou
    return Box3D, iou


def _window_ground_truth(repository_root, entry, box_class):
    source = (Path(repository_root) / "AB3Dmot/scripts/KITTI/v2v4real_label/test" /
              (entry["scene_id"] + "-" + str(entry["receiver_id"]) + ".txt"))
    if not source.is_file():
        raise FileNotFoundError("frozen-window GT missing: %s" % source)
    frames = {}
    needed = set(range(entry["start_frame"] - 1, entry["end_frame"] + 4))
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or int(fields[0]) not in needed:
            continue
        frame, track_id = int(fields[0]), int(fields[1])
        frames.setdefault(frame, {})[track_id] = box_class(
            x=float(fields[13]), y=float(fields[14]), z=float(fields[15]),
            h=float(fields[10]), w=float(fields[11]), l=float(fields[12]),
            ry=float(fields[16]))
    return frames


def _prediction_boxes(frame, box_class):
    return {int(runtime_id): box_class(x=float(box[0]), y=float(box[2]), z=float(box[1]),
            h=float(box[3]), w=float(box[4]), l=float(box[5]), ry=float(box[6]))
            for runtime_id, box in zip(frame["runtime_ids"], frame["boxes"])}


def _prediction_in_reference(frame, receiver_pose, reference_pose):
    """Convert emitted receiver-local boxes only at the metric boundary."""
    boxes = frame["boxes"]
    if boxes:
        boxes = transform_tracking_boxes(np.asarray(boxes, dtype=np.float32),
                                         receiver_pose, reference_pose).tolist()
    return {"frame_index": frame["frame_index"], "runtime_ids": frame["runtime_ids"],
            "boxes": boxes, "scores": frame["scores"]}


def _match_frame(predictions, targets, iou_fn):
    pred_ids, gt_ids = list(predictions), list(targets)
    matches, overlaps = {}, {}
    if pred_ids and gt_ids:
        def overlap(target, prediction):
            values = ("x", "y", "z", "h", "w", "l", "ry")
            if all(abs(getattr(target, name) - getattr(prediction, name)) < 1e-6 for name in values):
                return 1.0
            try:
                return float(iou_fn(target, prediction, metric="iou_3d"))
            except ValueError:
                return 0.0
        matrix = np.asarray([[overlap(targets[gt_id], predictions[pred_id])
                              for pred_id in pred_ids] for gt_id in gt_ids])
        rows, cols = linear_sum_assignment(1.0 - matrix)
        for row, col in zip(rows, cols):
            if matrix[row, col] >= 0.25:
                matches[gt_ids[row]] = pred_ids[col]
        for row, gt_id in enumerate(gt_ids):
            overlaps[gt_id] = int((matrix[row] >= 0.25).sum())
    return matches, overlaps


def _gt_conditioned_audit(repository_root, predictions):
    """Count evaluator-GT duplicate matches; lifecycle is label-target-only."""
    box_class, iou_fn = _ab3d_geometry(repository_root)
    totals = {"duplicate": 0, "duplicate_denominator": 0}
    for entry in WINDOW:
        timeline = (entry["scene_id"], entry["receiver_id"])
        frames = predictions[timeline]
        raw = list(range(entry["start_frame"], entry["end_frame"] + 1))
        if [int(frame["frame_index"]) for frame in frames] != raw:
            raise ValueError("evaluation predictions do not exactly match frozen receiver window")
        gt = _window_ground_truth(repository_root, entry, box_class)
        for raw_frame, prediction in zip(raw, frames):
            _, overlap = _match_frame(_prediction_boxes(prediction, box_class),
                                      gt.get(raw_frame, {}), iou_fn)
            for gt_id in gt.get(raw_frame, {}):
                totals["duplicate"] += max(0, overlap.get(gt_id, 0) - 1)
                totals["duplicate_denominator"] += 1
    if totals["duplicate_denominator"] != AUDIT_DENOMINATORS["duplicate"]:
        raise RuntimeError("evaluator-GT duplicate denominator mismatch")
    return {"duplicate": "%d/%d" % (totals["duplicate"], totals["duplicate_denominator"])}


def _target_lifecycle_audit(hypes, predictions, repository_root):
    """Target-only lifecycle audit from the evaluator-native label source."""
    box_class, iou_fn = _ab3d_geometry(repository_root)
    totals = {"states": 0, "first_frame_states": 0, "birth": 0, "birth_tp": 0,
              "keep": 0, "keep_tp": 0, "delete": 0, "delete_tp": 0}
    for entry in WINDOW:
        receiver, scene = entry["receiver_id"], entry["scene_id"]
        label_root = hypes.get("fusion", {}).get("args", {}).get("tracking_label_root")
        if not label_root:
            raise EnvironmentError("target lifecycle audit requires tracking_label_root")
        raw_target = load_tracking_label_file(str(Path(label_root) / (scene + "-" + str(receiver) + ".txt")))
        target = {}
        for raw in range(entry["start_frame"] - 1, entry["end_frame"] + 1):
            if raw not in raw_target:
                raise ValueError("tracking label missing audit frame %d" % raw)
            target[raw] = {actor: box_class(x=float(box[0]), y=float(box[2]), z=float(box[1]),
                                            h=float(box[3]), w=float(box[4]), l=float(box[5]), ry=float(box[6]))
                           for actor, box in raw_target[raw].items()}
        matched = {}
        for index, prediction in enumerate(predictions[(scene, receiver)]):
            raw = int(prediction["frame_index"]); current, _ = _match_frame(_prediction_boxes(prediction, box_class), target[raw], iou_fn); matched[raw] = current
            current_ids, previous_ids = set(target[raw]), set(target.get(raw - 1, {}))
            totals["states"] += len(current_ids)
            if index == 0: totals["first_frame_states"] += len(current_ids)
            else:
                for actor in current_ids - previous_ids:
                    totals["birth"] += 1; totals["birth_tp"] += int(actor in current)
                for actor in current_ids & previous_ids:
                    totals["keep"] += 1; totals["keep_tp"] += int(actor in matched.get(raw - 1, {}) and current.get(actor) == matched[raw - 1][actor])
                if raw + 2 <= entry["end_frame"]:
                    for actor in previous_ids:
                        if actor not in current_ids and actor not in target.get(raw + 1, {}) and actor not in target.get(raw + 2, {}):
                                totals["delete"] += 1
                                prior = matched.get(raw - 1, {}).get(actor); third = _prediction_boxes(predictions[(scene, receiver)][index + 2], box_class)
                                totals["delete_tp"] += int(prior is not None and prior not in third)
    for key, expected in AUDIT_DENOMINATORS.items():
        if key != "duplicate" and totals[key] != expected: raise RuntimeError("target audit denominator %s=%d expected=%d" % (key, totals[key], expected))
    return totals


def run_fixed_evaluation(hypes, checkpoint, device, repository_root="."):
    """Fresh strict-online 2x24-frame evaluation and GT-conditioned audit."""
    provenance = _checkpoint_provenance(checkpoint)
    label_root = hypes.get("fusion", {}).get("args", {}).get("tracking_label_root")
    if not label_root:
        raise EnvironmentError("fixed DualCMP evaluation requires tracking_label_root")
    predictions = {}
    for entry in WINDOW:
        receiver, scene_id = entry["receiver_id"], entry["scene_id"]
        hypes_copy = dict(hypes)
        hypes_copy["fusion"] = {"core_method": "DualCMPSequenceDataset", "args": {
            "stage": "tracking", "tracking_frames": 3, "runtime_only": True,
            "runtime_forbid_gt": True, "receiver_ids": [receiver], "frame_period_s": 0.1,
            "max_actors": 128, "max_cav": 2, "tracking_label_root": label_root,
        }}
        dataset = DualCMPSequenceDataset(hypes_copy, train=False)
        keys = [key for key in dataset.index.receiver_frame_keys(str(scene_id), str(receiver))
                if entry["start_frame"] <= int(key) <= entry["end_frame"]]
        if len(keys) != WINDOW_FRAMES or [int(key) for key in keys] != list(range(entry["start_frame"], entry["end_frame"] + 1)):
            raise ValueError("fixed evaluation requires 24 contiguous frames per receiver")
        online = OnlineDualCMP(_model_from_checkpoint(hypes_copy, checkpoint, device), device)
        frames = []
        for key in keys:
            runtime = dataset._runtime_frame(str(scene_id), str(receiver), key)
            prediction = online.step(dataset._collate_runtime_frame([runtime]))
            reference_pose = dataset._pose(dataset.index.receiver_record(str(scene_id), "0", key))
            frames.append(_prediction_in_reference(prediction, runtime["receiver_pose"], reference_pose))
        predictions[(str(scene_id), receiver)] = frames
    from opencood.tools.dualcmp.ab3dmot_metrics import compute_ab3dmot_metrics
    lifecycle = _target_lifecycle_audit(hypes, predictions, repository_root)
    duplicate = _gt_conditioned_audit(repository_root, predictions)
    duplicate_count, duplicate_denominator = [int(value) for value in duplicate["duplicate"].split("/")]
    lifecycle.update({"duplicate": duplicate_count, "duplicate_denominator": duplicate_denominator})
    rates = {"birth": lifecycle["birth_tp"] / lifecycle["birth"],
             "keep": lifecycle["keep_tp"] / lifecycle["keep"],
             "delete": lifecycle["delete_tp"] / lifecycle["delete"]}
    passed = {"birth": lifecycle["birth_tp"] >= AUDIT_THRESHOLDS["birth_tp"],
              "keep": lifecycle["keep_tp"] >= AUDIT_THRESHOLDS["keep_same_runtime"],
              "delete": lifecycle["delete_tp"] >= AUDIT_THRESHOLDS["delete_tp"],
              "duplicate": duplicate_count <= AUDIT_THRESHOLDS["duplicate_max"]}
    metrics = compute_ab3dmot_metrics(predictions, repository_root=repository_root,
                                      ground_truth_root=label_root)
    metric_pass = {key: metrics[key] >= threshold for key, threshold in METRIC_THRESHOLDS.items()
                   if key != "identity_switches_max"}
    metric_pass["identity_switches"] = metrics["identity_switches"] <= METRIC_THRESHOLDS["identity_switches_max"]
    combined = dict(metric_pass); combined.update(passed)
    return {"provenance": provenance, "metrics": metrics, "audit": {"numerator": lifecycle, "rate": rates,
            "pass": combined, "overall_pass": all(combined.values())}}


def main():
    parser = argparse.ArgumentParser(description="DualCMP fresh strict-online inference")
    parser.add_argument("--hypes", default="opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--receiver", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluate-output", default=None)
    args = parser.parse_args()
    if "V2V4REAL_ROOT" not in os.environ:
        raise EnvironmentError("V2V4REAL_ROOT must point to the V2V4Real split root")
    if "DUALCMP_TRACKING_LABEL_ROOT" not in os.environ:
        raise EnvironmentError("DUALCMP_TRACKING_LABEL_ROOT must point to evaluator-native labels")
    hypes = load_yaml(args.hypes)
    hypes["root_dir"] = str(Path(os.environ["V2V4REAL_ROOT"]) / "test")
    hypes["validate_dir"] = hypes["root_dir"]
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
    hypes["model"]["args"]["compression"] = checkpoint_detector_compression(
        checkpoint_payload
    )
    hypes["fusion"] = {"args": {"tracking_label_root": os.environ["DUALCMP_TRACKING_LABEL_ROOT"]}}
    if args.evaluate_output:
        import json
        result = run_fixed_evaluation(hypes, args.checkpoint, args.device)
        Path(args.evaluate_output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
    else:
        if args.scene is None or args.start_frame is None:
            raise ValueError("--scene and --start-frame are required for one-receiver smoke inference")
        result = run_receiver_frames(hypes, args.checkpoint, args.scene, args.receiver,
                                     args.start_frame, args.frames, args.device)
        print({key: len(value) for key, value in result.items()})


if __name__ == "__main__":
    main()
