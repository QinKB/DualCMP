"""Read-only 4-clip detector/tracker association audit on two GPUs."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from opencood.data_utils.dualcmp_dataloader import build_dataloader
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import DualCMPSequenceDataset
from opencood.loss.forecast_association import (
    ASSOCIATION_STATUS,
    VALIDATED_ASSOCIATION,
    associate_forecast_clip,
    associate_forecast_frame,
    association_contract,
    forecast_bev_iou,
    forecast_metric_version,
)
from opencood.tools.dualcmp.aligned_protocol import ALIGNED_FORECAST_WINDOW
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.online import _model_from_checkpoint
from opencood.tools.train_utils import to_device


def _cpu_state(state):
    return {name: getattr(state, name).detach().cpu().clone()
            for name in ("box", "score", "valid_mask", "active_mask", "runtime_id")}


def _proposal_output(proposals):
    scores, valid = proposals["scores"], proposals["mask"]
    ids = torch.arange(scores.shape[1], device=scores.device).expand_as(scores)
    state = SimpleNamespace(runtime_id=ids, valid_mask=valid, active_mask=valid)
    logits = torch.logit(scores.clamp(1e-6, 1.0 - 1e-6))
    return {"state": state, "track_box": proposals["boxes"],
            "track_exist_logits": logits}


def _stage_rows(stage, boxes, scores, predicted_mask, runtime_ids, target,
                assigned, diagnostic, keys):
    rows, errors, current_gt_count = [], [], 0
    for row, key in enumerate(keys):
        gt_indices = torch.nonzero(target["current_mask"][row], as_tuple=False).flatten()
        pred_indices = torch.nonzero(predicted_mask[row], as_tuple=False).flatten()
        complete = target["future_mask"][row].all(-1) & target["current_mask"][row]
        complete_indices = set(torch.nonzero(complete, as_tuple=False).flatten().tolist())
        current_gt_count += len(gt_indices)
        current_boxes = target["current_boxes"][row, gt_indices]
        pred_boxes = boxes[row, pred_indices]
        if len(pred_indices) and len(gt_indices):
            overlap = forecast_bev_iou(pred_boxes, current_boxes)
            distances = torch.cdist(pred_boxes[:, :2].float(), current_boxes[:, :2].float())
        else:
            overlap = boxes.new_zeros((len(pred_indices), len(gt_indices)))
            distances = boxes.new_zeros((len(pred_indices), len(gt_indices)))

        pred_to_gt = {}
        prediction_rows = []
        for local_pred, slot_tensor in enumerate(pred_indices):
            slot = int(slot_tensor)
            gt_index = int(assigned[row, slot])
            matched = gt_index >= 0
            if matched:
                local_gt = int(torch.nonzero(gt_indices == gt_index, as_tuple=False)[0])
                pred_box, gt_box = boxes[row, slot], target["current_boxes"][row, gt_index]
                center_error = float(torch.linalg.vector_norm(pred_box[:2] - gt_box[:2]))
                errors.append(center_error)
                pred_to_gt[gt_index] = slot
                size_error = (pred_box[3:6] - gt_box[3:6]).abs().cpu().tolist()
                yaw_error = float(torch.atan2(torch.sin(pred_box[6] - gt_box[6]),
                                              torch.cos(pred_box[6] - gt_box[6])).abs())
                matched_iou = float(diagnostic["matched_iou"][row, slot])
            else:
                local_gt, center_error, size_error, yaw_error, matched_iou = None, None, None, None, 0.0
            prediction_rows.append({
                "prediction_id": int(runtime_ids[row, slot]),
                "score": float(scores[row, slot]),
                "center_xy": boxes[row, slot, :2].cpu().tolist(),
                "matched_actor_id": (int(target["actor_ids"][row, gt_index]) if matched else None),
                "match_status": ASSOCIATION_STATUS[int(diagnostic["status"][row, slot])],
                "matched_iou": matched_iou,
                "best_iou": (float(overlap[local_pred].max()) if len(gt_indices) else 0.0),
                "center_error_m": center_error,
                "size_abs_error_hwl_m": size_error,
                "yaw_abs_error_rad": yaw_error,
            })

        gt_rows, eligible_complete = [], 0
        for local_gt, gt_tensor in enumerate(gt_indices):
            gt_index = int(gt_tensor)
            if gt_index in complete_indices and gt_index in pred_to_gt:
                eligible_complete += 1
            if len(pred_indices):
                nearest_local = int(distances[:, local_gt].argmin())
                nearest = float(distances[nearest_local, local_gt])
                best_iou = float(overlap[:, local_gt].max())
                nearest_id = int(runtime_ids[row, pred_indices[nearest_local]])
            else:
                nearest, best_iou, nearest_id = None, 0.0, None
            slot = pred_to_gt.get(gt_index)
            gt_rows.append({
                "actor_id": int(target["actor_ids"][row, gt_index]),
                "complete_future50": gt_index in complete_indices,
                "matched_prediction_id": (int(runtime_ids[row, slot]) if slot is not None else None),
                "best_iou": best_iou,
                "nearest_prediction_id": nearest_id,
                "nearest_center_distance_m": nearest,
                "gt_center_xy": target["current_boxes"][row, gt_index, :2].cpu().tolist(),
            })
        rows.append({
            "scene": key[0], "receiver": key[1], "prediction_frame": key[2],
            "current_gt": len(gt_indices), "predicted": len(pred_indices),
            "matched": len(pred_to_gt), "complete": len(complete_indices),
            "eligible_complete": eligible_complete,
            "matched_center_error_m": {
                "count": len([x for x in prediction_rows if x["center_error_m"] is not None]),
                "mean": (sum(errors[-len(pred_to_gt):]) / len(pred_to_gt) if pred_to_gt else None),
            },
            "predictions": prediction_rows, "current_gt_actors": gt_rows,
        })
    return {"stage": stage, "current_gt": current_gt_count,
            "predicted": sum(x["predicted"] for x in rows),
            "matched": sum(x["matched"] for x in rows),
            "complete": sum(x["complete"] for x in rows),
            "eligible_complete": sum(x["eligible_complete"] for x in rows),
            "matched_center_error_m": {
                "count": len(errors), "mean": sum(errors) / len(errors) if errors else None,
                "maximum": max(errors) if errors else None,
            }, "clips": rows}


def _audit_batch(model, batch, device):
    pre_states = []
    handle = model.tracker.register_forward_hook(
        lambda _module, _inputs, output: pre_states.append(_cpu_state(output)))
    try:
        with torch.inference_mode():
            outputs, _ = model(batch["runtime_frames"], run_forecast=False)
    finally:
        handle.remove()
    if len(outputs) != 11 or len(pre_states) != 11:
        raise RuntimeError("expected eleven model outputs and pre-learned tracker states")
    target = batch["target_frames"][-1]
    association_targets = batch["association_target_frames"]
    if not torch.equal(association_targets[-1]["actor_ids"], target["actor_ids"]):
        raise RuntimeError("final causal association IDs disagree with the final target")
    if not torch.equal(association_targets[-1]["current_mask"], target["current_mask"]):
        raise RuntimeError("final causal association mask disagrees with the final target")
    keys = [(str(scene), int(receiver), int(frame)) for (scene, receiver), frame in zip(
        batch["scene_receiver_keys"], batch["runtime_frames"][-1]["receiver_frame_idx"])]
    final = outputs[-1]

    proposal = final["proposals"]
    raw_output = _proposal_output(proposal)
    raw_assigned, _, raw_diag = associate_forecast_frame(raw_output, target)
    raw = _stage_rows("raw_detector_proposals", proposal["boxes"], proposal["scores"],
                      proposal["mask"], raw_output["state"].runtime_id, target,
                      raw_assigned, raw_diag, keys)

    pre = pre_states[-1]
    pre_boxes = pre["box"].to(device)
    pre_scores = pre["score"].to(device)
    pre_valid = pre["valid_mask"].to(device)
    pre_active = pre["active_mask"].to(device) & pre_valid
    pre_ids = pre["runtime_id"].to(device)
    pre_state = SimpleNamespace(runtime_id=pre_ids, valid_mask=pre_valid,
                                 active_mask=pre_active)
    pre_logits = torch.logit(pre_scores.clamp(1e-6, 1.0 - 1e-6))
    pre_output = {"state": pre_state, "track_box": pre_boxes,
                  "track_exist_logits": pre_logits}
    pre_assigned, _, pre_diag = associate_forecast_frame(pre_output, target)
    kalman = _stage_rows("kalman_before_learned_update", pre_boxes, pre_scores,
                         pre_active, pre_ids, target, pre_assigned, pre_diag, keys)

    learned_assigned, learned_records = associate_forecast_clip(outputs, association_targets)
    learned_state = final["state"]
    learned_active = getattr(learned_state, "active_mask", learned_state.valid_mask) & learned_state.valid_mask
    learned_active = learned_active & (
        learned_state.forecast_history_mask.sum(-1) >= model.forecast_head.min_forecast_history_steps)
    learned = _stage_rows("learned_after_update", final["track_box"],
                          final["track_exist_logits"].sigmoid(), learned_active,
                          learned_state.runtime_id, target, learned_assigned,
                          learned_records[-1], keys)
    return {"clip_keys": keys, "pre_learned_state_frames": len(pre_states),
            "raw_detector": raw, "kalman_before_learned": kalman,
            "learned_after_update": learned}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("refusing to overwrite audit output: %s" % output)
    if "LOCAL_RANK" not in os.environ:
        raise RuntimeError("perception audit requires torchrun --nproc_per_node=2")
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if world_size != 2:
        raise RuntimeError("perception audit requires exactly two ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    payload = torch.load(args.checkpoint, map_location="cpu")
    hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    root = Path(os.environ["V2V4REAL_ROOT"]) / "train"
    hypes["root_dir"] = hypes["validate_dir"] = str(root)
    hypes["fusion"] = {"args": {
        "stage": "forecast", "tracking_frames": 3, "forecast_history_frames": 10,
        "future_frames": 50, "runtime_only": False, "runtime_forbid_gt": True,
        "receiver_ids": [0, 1], "max_actors": 128, "max_cav": 2,
        "tracking_label_root": os.environ["DUALCMP_TRAIN_TRACKING_LABEL_ROOT"],
        "tracking_label_layout": "dmstrack_train", "split_name": "train",
        "partner_delay_ms": 100.0, "shuffle_training_points": False,
        "include_forecast_association_history": True,
        "forecast_coordinate_frame": "current_receiver",
        "window": list(ALIGNED_FORECAST_WINDOW),
    }}
    dataset = DualCMPSequenceDataset(hypes, train=False)
    if len(dataset) != 4:
        raise RuntimeError("aligned perception audit requires four train clips")
    _, loader, _ = build_dataloader(
        dataset, 2, True, workers=args.workers, rank=rank, world_size=world_size,
        seed=0, training=False, collate_fn=dataset.collate_batch_test, pin_memory=True)
    model = _model_from_checkpoint(hypes, args.checkpoint, device,
                                   allow_dirty_smoke=True)
    if getattr(model, "forecast_association_policy", None) != VALIDATED_ASSOCIATION:
        raise RuntimeError("checkpoint must use validated causal forecast association")
    local = []
    with torch.inference_mode():
        for host_batch in loader:
            batch = to_device(host_batch, device)
            local.append(_audit_batch(model, batch, device))
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank == 0:
        clips = [item for group in gathered for item in group]
        if sum(len(item["clip_keys"]) for item in clips) != 4:
            raise RuntimeError("distributed audit did not produce exactly four clips")
        report = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "forecast_association": association_contract(VALIDATED_ASSOCIATION),
            "forecast_metric_version": forecast_metric_version(VALIDATED_ASSOCIATION),
            "box_format": "[x,y,z,h,w,l,yaw] in current receiver coordinates",
            "iou_threshold": 0.7,
            "runtime_frames_per_clip": 11,
            "delay_ms": 100.0,
            "batch_per_rank": 2,
            "world_size": world_size,
            "clip_keys": [key for clip in clips for key in clip["clip_keys"]],
            "clips": clips,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"clips": 4,
            "stages": {name: {
                "current_gt": sum(clip[name]["current_gt"] for clip in clips),
                "predicted": sum(clip[name]["predicted"] for clip in clips),
                "matched": sum(clip[name]["matched"] for clip in clips),
                "complete": sum(clip[name]["complete"] for clip in clips),
                "eligible_complete": sum(clip[name]["eligible_complete"] for clip in clips),
            } for name in ("raw_detector", "kalman_before_learned", "learned_after_update")},
            "output": str(output)}, separators=(",", ":")), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
