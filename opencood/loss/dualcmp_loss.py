"""Detector-plus-causal actor supervision using repository loss primitives."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from opencood.loss.point_pillar_loss import PointPillarLoss, WeightedSmoothL1Loss


class DualCmpLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.detector_loss = PointPillarLoss(args["detector"])
        self.reg = WeightedSmoothL1Loss()

    def forward(self, outputs, targets):
        zero = outputs[0]["psm"].sum() * 0.0
        det = exist = box = velocity = zero
        match_count = 0
        bindings = [dict() for _ in range(outputs[0]["state"].box.shape[0])]
        for output, target in zip(outputs, targets):
            det = det + self.detector_loss(output, target["fused_label_dict"])
            state = output["state"]
            gt_mask, gt_boxes, gt_ids = target["current_mask"], target["current_boxes"], target["actor_ids"]
            # Target-only existing-first then Hungarian newborn assignment.
            # Runtime IDs never read GT; bindings are loss-local only.
            assigned = torch.full_like(state.runtime_id, -1)
            for row in range(state.box.shape[0]):
                remaining = set(torch.nonzero(gt_mask[row], as_tuple=False).flatten().tolist())
                for slot, binding in list(bindings[row].items()):
                    previous_runtime_id, previous_id = binding
                    if (not bool(state.valid_mask[row, slot]) or
                            int(state.runtime_id[row, slot]) != previous_runtime_id):
                        # Deleted slots can be reused by a newborn runtime ID.
                        bindings[row].pop(slot)
                        continue
                    found = torch.nonzero((gt_ids[row] == previous_id) & gt_mask[row], as_tuple=False).flatten()
                    if found.numel():
                        index = int(found[0]); assigned[row, slot] = index; remaining.discard(index)
                    else:
                        bindings[row].pop(slot)
                slots = [slot for slot in torch.nonzero(state.valid_mask[row], as_tuple=False).flatten().tolist() if assigned[row, slot] < 0]
                if slots and remaining:
                    gt = sorted(remaining)
                    pred = output["track_box"][row, slots]
                    candidate_targets = gt_boxes[row, gt]
                    cls_cost = -output["track_exist_logits"][row, slots].sigmoid()[:, None]
                    scale = candidate_targets[:, 3:6].mean().clamp_min(1.0)
                    cost = 2.0 * cls_cost + 0.25 * (pred[:, None] - candidate_targets[None]).abs().mean(-1) / scale
                    left, right = linear_sum_assignment(cost.detach().cpu().numpy())
                    for local_slot, local_gt in zip(left.tolist(), right.tolist()):
                        slot, index = slots[local_slot], gt[local_gt]
                        assigned[row, slot] = index
                        bindings[row][slot] = (int(state.runtime_id[row, slot]), int(gt_ids[row, index]))
                        remaining.discard(index)
            positive = assigned >= 0
            match_count += int(positive.sum().item())
            logits = output["track_exist_logits"]
            exist = exist + self.detector_loss.cls_loss_func(logits[..., None], positive.float()[..., None], torch.ones_like(state.score)).mean()
            if positive.any():
                rows, slots = torch.nonzero(positive, as_tuple=True); gt_index = assigned[rows, slots]
                box = box + self.reg(output["track_box"][rows, slots], gt_boxes[rows, gt_index]).mean()
                gt_velocity = target["current_velocity"][rows, gt_index]
                velocity = velocity + self.reg(output["track_velocity"][rows, slots], gt_velocity).mean()
        total = det + exist + box + velocity
        return {"total_loss": total, "L_det": det, "L_exist": exist, "L_box": box, "L_velocity": velocity, "match_count": match_count}
