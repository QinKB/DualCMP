"""Detector-plus-causal actor supervision using repository loss primitives."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from opencood.loss.point_pillar_loss import PointPillarLoss, WeightedSmoothL1Loss
from opencood.loss.dualcmp_gmm_loss import nearest_mode_gmm_loss


def update_identity_bindings(output, target, bindings):
    """Existing DualCmpLoss ownership/Hungarian logic, reusable target-only."""
    state = output["state"]; gt_mask, gt_boxes, gt_ids = target["current_mask"], target["current_boxes"], target["actor_ids"]
    assigned = torch.full_like(state.runtime_id, -1)
    for row in range(state.box.shape[0]):
        remaining=set(torch.nonzero(gt_mask[row],as_tuple=False).flatten().tolist()); owned=set()
        for slot,binding in list(bindings[row].items()):
            runtime,actor=binding
            if not bool(state.valid_mask[row,slot]) or int(state.runtime_id[row,slot])!=runtime: bindings[row].pop(slot); continue
            owned.add(slot); found=torch.nonzero((gt_ids[row]==actor)&gt_mask[row],as_tuple=False).flatten()
            if found.numel(): index=int(found[0]); assigned[row,slot]=index; remaining.discard(index)
        slots=[slot for slot in torch.nonzero(state.valid_mask[row],as_tuple=False).flatten().tolist() if assigned[row,slot]<0 and slot not in owned]
        if slots and remaining:
            gt=sorted(remaining); pred=output["track_box"][row,slots]; candidates=gt_boxes[row,gt]
            cost=2.0*(-output["track_exist_logits"][row,slots].sigmoid()[:,None])+0.25*(pred[:,None]-candidates[None]).abs().mean(-1)/candidates[:,3:6].mean().clamp_min(1.0)
            left,right=linear_sum_assignment(cost.detach().cpu().numpy())
            for local_slot,local_gt in zip(left.tolist(),right.tolist()):
                slot,index=slots[local_slot],gt[local_gt]; assigned[row,slot]=index; bindings[row][slot]=(int(state.runtime_id[row,slot]),int(gt_ids[row,index]))
    return assigned,bindings


class DualCmpLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.detector_loss = PointPillarLoss(args["detector"])
        self.reg = WeightedSmoothL1Loss()

    def forward(self, outputs, targets):
        zero = outputs[0]["psm"].sum() * 0.0
        det = exist = box = velocity = zero
        gmm = mode = zero
        match_count = 0
        bindings = [dict() for _ in range(outputs[0]["state"].box.shape[0])]
        for output, target in zip(outputs, targets):
            det = det + self.detector_loss(output, target["fused_label_dict"])
            state = output["state"]
            gt_mask, gt_boxes, gt_ids = target["current_mask"], target["current_boxes"], target["actor_ids"]
            # JDT3D TrackingLoss::loss_single_frame first inherits the identity
            # owned by each existing track.  A disappeared owner remains an
            # existence negative (matched_gt_idxes=-2 there), rather than an
            # unmatched proposal that may acquire another GT identity.
            # Runtime IDs never read GT; bindings are loss-local only.
            assigned, bindings = update_identity_bindings(output, target, bindings)
            positive = assigned >= 0
            match_count += int(positive.sum().item())
            logits = output["track_exist_logits"]
            exist = exist + self.detector_loss.cls_loss_func(logits[..., None], positive.float()[..., None], torch.ones_like(state.score)).mean()
            if positive.any():
                rows, slots = torch.nonzero(positive, as_tuple=True); gt_index = assigned[rows, slots]
                box = box + self.reg(output["track_box"][rows, slots], gt_boxes[rows, gt_index]).mean()
                gt_velocity = target["current_velocity"][rows, gt_index]
                velocity = velocity + self.reg(output["track_velocity"][rows, slots], gt_velocity).mean()
            # CoLIF ColifLoss::_forecast_loss consumes the existing loss-local
            # current-frame pairs only; future labels never enter the model.
            forecast = output.get("forecast")
            if forecast is not None and positive.any() and "future_mask" in target:
                valid_rows, valid_slots = torch.nonzero(
                    positive & forecast["forecast_valid_mask"], as_tuple=True)
                if valid_rows.numel():
                    gt_index = assigned[valid_rows, valid_slots]
                    future_receiver = (target["current_boxes"][valid_rows, gt_index, None, :2]
                                       + target["future_xy"][valid_rows, gt_index]
                                       - output["track_box"][valid_rows, valid_slots, None, :2])
                    heading = output["track_box"][valid_rows, valid_slots, 6]
                    # CoLIF@6b8788e compatibility with the released V2V4Real
                    # CMP intention points: tracking remains radians, but the
                    # MTR-local target uses degree-numeric trigonometry.
                    cmp_heading = torch.rad2deg(heading)
                    cosine, sine = torch.cos(cmp_heading)[:, None], torch.sin(cmp_heading)[:, None]
                    future_actor = torch.stack((cosine * future_receiver[..., 0] + sine * future_receiver[..., 1],
                                                -sine * future_receiver[..., 0] + cosine * future_receiver[..., 1]), dim=-1)
                    layer_scores = forecast["loss_pred_scores_layers"][valid_rows, valid_slots]
                    layer_trajs = forecast["loss_pred_trajs_local_layers"][valid_rows, valid_slots]
                    layer_intentions = forecast["loss_intention_points_layers"][valid_rows, valid_slots]
                    layer_losses = [nearest_mode_gmm_loss(
                        layer_scores[:, layer], layer_trajs[:, layer], future_actor,
                        target["future_mask"][valid_rows, gt_index],
                        layer_intentions[:, layer])
                        for layer in range(layer_scores.shape[1])]
                    gmm = gmm + torch.stack([item["gmm"] for item in layer_losses]).mean() / layer_losses[-1]["gmm_count"].clamp_min(1.0)
                    mode = mode + torch.stack([item["mode"] for item in layer_losses]).mean() / layer_losses[-1]["mode_count"].clamp_min(1.0)
        total = det + exist + box + velocity + gmm + mode
        return {"total_loss": total, "L_det": det, "L_exist": exist, "L_box": box,
                "L_velocity": velocity, "L_gmm": gmm, "L_mode": mode,
                "match_count": match_count}
