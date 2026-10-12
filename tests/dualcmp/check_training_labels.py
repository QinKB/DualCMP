"""Real DMSTrack train-label contract for DualCMP target-only supervision."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.distributed as dist

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.data_utils.v2v4real.datasets.dualcmp_target_builder import (
    load_tracking_label_file, transform_tracking_boxes)
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.online import (
    OnlineDualCMP, _model_from_checkpoint, _prediction_in_reference,
    _checkpoint_provenance)
from opencood.tools.dualcmp.protocol import TRAIN_LABEL_WINDOW
from opencood.tools.dualcmp.ab3dmot_metrics import compute_ab3dmot_metrics
from opencood.utils import box_utils_v2v4real


def check_mapping():
    root = os.environ["V2V4REAL_ROOT"]
    labels = os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT"]
    h = load_yaml("opencood/hypes_yaml/point_pillar_v2v4real.yaml")
    h["root_dir"] = str(Path(root) / "train")
    h["fusion"] = {"args": {"stage": "tracking", "tracking_frames": 3,
        "runtime_only": False, "runtime_forbid_gt": True, "receiver_ids": [0, 1],
        "tracking_label_root": labels, "tracking_label_layout": "dmstrack_train"}}
    dataset = DualCMPSequenceDataset(h, train=True)
    scene = "testoutput_CAV_data_2022-03-17-10-43-13"
    keys = dataset.index.receiver_frame_keys(scene, "0")
    assert keys[0] == "000001" and len(keys) == 410
    expected = load_tracking_label_file(str(Path(labels) / "0008.txt"))[0]
    reference_pose = dataset._pose(dataset.index.receiver_record(scene, "0", keys[0]))
    for receiver in ("0", "1"):
        pose = dataset._pose(dataset.index.receiver_record(scene, receiver, keys[0]))
        actual = dataset._actors(scene, receiver, keys[0], pose)
        ids = sorted(expected)
        transformed = transform_tracking_boxes(
            np.stack([expected[actor_id] for actor_id in ids]), reference_pose, pose)
        assert sorted(actual) == ids
        torch.testing.assert_close(torch.as_tensor([actual[actor_id] for actor_id in ids]),
                                   torch.as_tensor(transformed), atol=3e-6, rtol=0)
    print("DMSTrack train labels: scene8 raw000001 -> label0 PASS")


def evaluate(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise RuntimeError("fresh tracking-label evaluation requires two GPU processes")
    if not args.output:
        raise ValueError("--output is required with --checkpoint")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    payload = torch.load(args.checkpoint, map_location="cpu")
    provenance = _checkpoint_provenance(
        args.checkpoint, payload, allow_dirty_smoke=args.allow_dirty_smoke)
    if payload["provenance"].get("tracking_label_contract") != "dmstrack_train":
        raise ValueError("checkpoint does not use the native training-label contract")
    h, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    h["root_dir"] = h["validate_dir"] = str(Path(os.environ["V2V4REAL_ROOT"]) / "train")
    h["fusion"] = {"args": {
        "stage": "tracking", "tracking_frames": 3, "runtime_only": True,
        "runtime_forbid_gt": True, "receiver_ids": [rank], "split_name": "train",
        "shuffle_training_points": False, "sequence_augmentation": False,
        "tracking_label_layout": "dmstrack_train",
        "tracking_label_root": os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT"]}}
    dataset = DualCMPSequenceDataset(h, train=False)
    model = _model_from_checkpoint(h, args.checkpoint, device,
                                   allow_dirty_smoke=args.allow_dirty_smoke)
    online = OnlineDualCMP(model, device)
    kf_output = {}
    def capture_kf(_module, _inputs, state):
        kf_output["state"] = state
    hook = model.tracker.register_forward_hook(capture_kf)
    # Same untouched upstream metric source used by trace_tracking_ddp.py.
    metric_path = Path(args.reference) / "opencood/utils/eval_utils.py"
    spec = importlib.util.spec_from_file_location("official_detection_metrics", metric_path)
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    stats = {stage: {t: {"tp": [], "fp": [], "gt": 0} for t in (0.5, 0.7)}
             for stage in ("detector_proposals", "learned_tracks", "same_ids_kf_boxes",
                           "same_ids_kf_yaw", "same_ids_kf_xy", "same_ids_kf_size")}
    entry = TRAIN_LABEL_WINDOW[rank]
    scene, receiver = entry["scene_id"], str(entry["receiver_id"])
    label_path, raw_keys = dataset.dmstrack_train_labels[scene]
    keys = [key for key in raw_keys if entry["start_frame"] <= int(key) <= entry["end_frame"]]
    if len(keys) != 24:
        raise RuntimeError("frozen train-label window must contain 24 frames per receiver")
    frames = []
    for key in keys:
        runtime = dataset._runtime_frame(scene, receiver, key)
        prediction = online.step(dataset._collate_runtime_frame([runtime]))
        actors = dataset._actors(scene, receiver, key, runtime["receiver_pose"])
        gt = torch.as_tensor(np.asarray(list(actors.values())), device=device).reshape(-1, 7)
        gt = box_utils_v2v4real.boxes_to_corners_3d(gt, order="hwl")
        proposals = online._last_output["proposals"]
        selected = proposals["mask"][0]
        state = online._last_output["state"]
        active = state.active_mask[0]
        kf_state = kf_output["state"]
        if kf_state.runtime_id[0, active].tolist() != prediction["runtime_ids"]:
            raise RuntimeError("same-ID geometry comparison lost runtime ownership")
        learned = torch.tensor(prediction["boxes"], device=device).reshape(-1, 7)
        scores = torch.tensor(prediction["scores"], device=device)
        kf_boxes = kf_state.box[0, active]
        geometry = [("same_ids_kf_boxes", kf_boxes, scores)]
        for stage, selection in (("same_ids_kf_yaw", slice(6, 7)),
                                 ("same_ids_kf_xy", slice(0, 2)),
                                 ("same_ids_kf_size", slice(3, 6))):
            boxes = learned.clone()
            boxes[:, selection] = kf_boxes[:, selection]
            geometry.append((stage, boxes, scores))
        for stage, boxes, scores in [
            ("detector_proposals", proposals["boxes"][0, selected], proposals["scores"][0, selected]),
            ("learned_tracks", learned, scores)] + geometry:
            corners = box_utils_v2v4real.boxes_to_corners_3d(boxes, order="hwl")
            for threshold in stats[stage]:
                metrics.caluclate_tp_fp(corners, scores, gt, stats[stage], threshold)
        reference_pose = dataset._pose(dataset.index.receiver_record(scene, "0", key))
        frame = _prediction_in_reference(prediction, runtime["receiver_pose"], reference_pose)
        # The released label file uses scene-local ordinals, not raw keys.
        frame["frame_index"] = raw_keys.index(key)
        frames.append(frame)
    hook.remove()
    gathered = [None, None]
    dist.all_gather_object(gathered, (receiver, frames, stats))
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix="dualcmp-train-labels-") as temp:
            for receiver_id, _, _ in gathered:
                (Path(temp) / (scene + "-" + receiver_id + ".txt")).symlink_to(label_path.resolve())
            tracking = compute_ab3dmot_metrics(
                {(scene, receiver_id): predictions for receiver_id, predictions, _ in gathered},
                repository_root=".", ground_truth_root=temp)
        report = {"scope": "train-window", "window": list(TRAIN_LABEL_WINDOW),
                  "frames": sum(len(item[1]) for item in gathered), "world_size": 2,
                  "provenance": provenance, "tracking": tracking,
                  "ap_gt_scope": "native labels cropped to receiver range", "stages": {}}
        for stage in stats:
            report["stages"][stage] = {}
            for threshold in stats[stage]:
                merged = {key: sum((group[2][stage][threshold][key] for group in gathered),
                                   0 if key == "gt" else []) for key in ("tp", "fp", "gt")}
                tp_count, fp_count = sum(merged["tp"]), sum(merged["fp"])
                report["stages"][stage][str(threshold)] = {
                    "AP": metrics.calculate_ap({threshold: merged}, threshold)[0],
                    "tp": tp_count, "fp": fp_count, "gt": merged["gt"]}
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--allow-dirty-smoke", action="store_true")
    parser.add_argument("--reference", default="../Reference/Code/V2V4Real")
    args = parser.parse_args()
    if args.checkpoint:
        evaluate(args)
    else:
        check_mapping()


if __name__ == "__main__":
    main()
