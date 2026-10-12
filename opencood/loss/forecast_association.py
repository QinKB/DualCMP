"""Target-only forecast association; never consumed by the online model.

Reuse CMP's score-ordered BEV-IoU admission and JDT3D's loss-local identity
ownership. Detection/tracking set-prediction matching remains unchanged.
"""
import torch
from mmcv.ops import box_iou_rotated


LEGACY_ASSOCIATION = "legacy_hungarian_v1"
VALIDATED_ASSOCIATION = "iou_causal_v1"
ASSOCIATION_POLICIES = (LEGACY_ASSOCIATION, VALIDATED_ASSOCIATION)
FORECAST_IOU_THRESHOLD = 0.7
ASSOCIATION_STATUS = {
    0: "inactive", 1: "matched", 2: "unmatched_iou",
    3: "owner_absent", 4: "identity_conflict", 5: "duplicate_owner",
}


def association_contract(policy):
    if policy not in ASSOCIATION_POLICIES:
        raise ValueError("unknown forecast association policy: " + str(policy))
    return {"policy": policy, "iou_threshold": (
        FORECAST_IOU_THRESHOLD if policy == VALIDATED_ASSOCIATION else None),
        "geometry": "receiver_frame_rotated_bev_hwl",
        "ownership": "causal_target_only_runtime_id" if policy == VALIDATED_ASSOCIATION else "legacy_loss_local"}


def forecast_metric_version(policy):
    association_contract(policy)
    return "exact_10hz_v1" if policy == VALIDATED_ASSOCIATION else "cmp_legacy_v1"


@torch.no_grad()
def forecast_bev_iou(predicted, ground_truth):
    """Existing MMCV CPU/CUDA geometry, verified against CMP polygons."""
    if (predicted.ndim != 2 or ground_truth.ndim != 2
            or predicted.shape[-1] != 7 or ground_truth.shape[-1] != 7
            or predicted.device != ground_truth.device):
        raise ValueError("forecast IoU requires same-device [N,7] H-W-L boxes")
    for boxes in (predicted, ground_truth):
        if not torch.isfinite(boxes).all() or (boxes[:, 3:6] <= 0).any():
            raise ValueError("forecast association boxes must be finite with positive dimensions")
    if not len(predicted) or not len(ground_truth):
        return predicted.new_zeros((len(predicted), len(ground_truth)))
    order = [0, 1, 5, 4, 6]
    left, right = predicted[:, order].float().contiguous(), ground_truth[:, order].float().contiguous()
    # The native flag names refer to image axes. clockwise=True uses the
    # [cos,-sin; sin,cos] matrix used by CMP's LiDAR corner construction.
    # Flipping this flag produces a measured .307 IoU mismatch on rotated boxes.
    result = box_iou_rotated(left, right, clockwise=True)
    if not torch.isfinite(result).all():
        raise ValueError("forecast association IoU must be finite")
    return result.clamp(0.0, 1.0)


@torch.no_grad()
def associate_forecast_frame(output, target, bindings=None):
    """Admit only plausible current pairs, without changing runtime identities.

    Once a runtime owns a GT identity in this causal window it cannot acquire
    another identity while that runtime survives. A failed match remains a
    negative/unknown observation, not supervision for a different vehicle.
    """
    state, boxes = output["state"], output["track_box"].detach()
    runtime_ids = state.runtime_id
    active = getattr(state, "active_mask", state.valid_mask) & state.valid_mask
    gt_mask, gt_boxes, gt_ids = target["current_mask"], target["current_boxes"], target["actor_ids"]
    if (boxes.shape != (*runtime_ids.shape, 7) or active.shape != runtime_ids.shape
            or gt_boxes.shape != (*gt_mask.shape, 7) or gt_ids.shape != gt_mask.shape
            or gt_mask.shape[0] != boxes.shape[0]):
        raise ValueError("forecast association batch/box/identity shapes disagree")
    if bindings is None:
        bindings = [dict() for _ in range(boxes.shape[0])]
    if len(bindings) != boxes.shape[0]:
        raise ValueError("forecast ownership batch size disagrees")
    assigned = torch.full_like(runtime_ids, -1)
    matched_iou = boxes.new_zeros(runtime_ids.shape)
    best_iou = boxes.new_zeros(runtime_ids.shape)
    status = torch.zeros_like(runtime_ids)
    scores = output["track_exist_logits"].detach().sigmoid()
    if scores.shape != runtime_ids.shape or not torch.isfinite(scores).all():
        raise ValueError("forecast association scores must be finite [B,Q]")
    for row in range(boxes.shape[0]):
        for slot, (runtime, _) in list(bindings[row].items()):
            if not bool(state.valid_mask[row, slot]) or int(runtime_ids[row, slot]) != runtime:
                bindings[row].pop(slot)
        slots = torch.nonzero(active[row], as_tuple=False).flatten().tolist()
        candidates = torch.nonzero(gt_mask[row], as_tuple=False).flatten().tolist()
        ids = gt_ids[row, candidates].tolist()
        if len(ids) != len(set(ids)) or any(actor < 0 for actor in ids):
            raise ValueError("valid forecast GT identities must be unique and nonnegative")
        overlap = forecast_bev_iou(boxes[row, slots], gt_boxes[row, candidates])
        if candidates:
            best_iou[row, slots] = overlap.max(dim=1).values
        remaining = set(range(len(candidates)))
        ordered = sorted(range(len(slots)), key=lambda index: (-float(scores[row, slots[index]]), slots[index]))
        for local_slot in ordered:
            slot = slots[local_slot]
            owner = bindings[row].get(slot)
            if owner is not None:
                if owner[1] not in ids:
                    status[row, slot] = 3
                    continue
                local_gt = ids.index(owner[1])
                if local_gt not in remaining:
                    status[row, slot] = 5
                    continue
            elif remaining:
                local_gt = max(sorted(remaining), key=lambda index: float(overlap[local_slot, index]))
            else:
                status[row, slot] = 2
                continue
            if float(overlap[local_slot, local_gt]) < FORECAST_IOU_THRESHOLD:
                conflict = owner is not None and any(
                    float(overlap[local_slot, index]) >= FORECAST_IOU_THRESHOLD for index in remaining)
                status[row, slot] = 4 if conflict else 2
                continue
            index = candidates[local_gt]
            assigned[row, slot], matched_iou[row, slot] = index, overlap[local_slot, local_gt]
            status[row, slot] = 1
            bindings[row][slot] = (int(runtime_ids[row, slot]), int(gt_ids[row, index]))
            remaining.remove(local_gt)
    return assigned, bindings, {"matched_iou": matched_iou, "best_iou": best_iou, "status": status}


@torch.no_grad()
def associate_forecast_clip(outputs, targets):
    """Replay only causal current-frame annotations; no future GT is read."""
    if not outputs or len(outputs) != len(targets):
        raise ValueError("forecast association requires equal nonempty causal frame lists")
    bindings, records = None, []
    for output, target in zip(outputs, targets):
        assigned, bindings, record = associate_forecast_frame(output, target, bindings)
        records.append(record)
    return assigned, records
