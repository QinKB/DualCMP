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

    @staticmethod
    def _future_relative_to_track(future_xy, gt_current_xy, track_current_xy):
        """Copy CoLIF@6b8788e: express GT future around predicted track."""
        return gt_current_xy + future_xy - track_current_xy

    @staticmethod
    def _receiver_displacement_to_actor(displacement, heading):
        """Copy CoLIF@6b8788e and CMP's legacy degree-numeric boundary."""
        cmp_heading = torch.rad2deg(heading)
        cosine, sine = torch.cos(cmp_heading), torch.sin(cmp_heading)
        x, y = displacement[..., 0], displacement[..., 1]
        return torch.stack((cosine * x + sine * y, -sine * x + cosine * y), dim=-1)

    def _dense_context_future_loss(self, output, target, assigned, zero):
        """Copy CoLIF@6b8788e's official CMP per-center dense objective."""
        forecast = output.get("forecast")
        if forecast is None:
            return zero, zero.detach().clone()
        dense = forecast.get("packed_dense_trajs_local")
        entries = forecast.get("packed_entries", ())
        context_slots = forecast.get("packed_context_slots")
        headings = forecast.get("packed_center_heading")
        if dense is None or context_slots is None or headings is None or not dense.shape[0]:
            return zero, zero.detach().clone()
        center_sum = dense.sum() * 0.0
        center_count = dense.new_zeros(())
        for row, (batch_index, center_slot) in enumerate(entries):
            center_xy = output["track_box"][batch_index, center_slot, :2]
            predictions, targets, masks = [], [], []
            for context_index, query_slot in enumerate(context_slots[row].tolist()):
                if query_slot < 0:
                    continue
                target_index = int(assigned[batch_index, query_slot])
                if target_index < 0:
                    continue
                future_mask = target["future_mask"][batch_index, target_index]
                if not bool(future_mask.any()):
                    continue
                future_receiver = self._future_relative_to_track(
                    target["future_xy"][batch_index, target_index],
                    target["current_boxes"][batch_index, target_index, :2],
                    center_xy,
                )
                predictions.append(dense[row, context_index])
                targets.append(
                    self._receiver_displacement_to_actor(
                        future_receiver, headings[row]
                    )
                )
                masks.append(future_mask)
            if not predictions:
                continue
            prediction = torch.stack(predictions)
            row_loss = nearest_mode_gmm_loss(
                prediction.new_zeros(prediction.shape[0], 1),
                prediction[:, None],
                torch.stack(targets),
                torch.stack(masks),
            )
            center_sum = center_sum + row_loss["gmm"] / row_loss[
                "gmm_count"
            ].clamp_min(1.0)
            center_count = center_count + 1.0
        return center_sum, center_count

    def forward(self, outputs, targets):
        zero = outputs[0]["psm"].sum() * 0.0
        det = exist = box = velocity = zero
        gmm_sum = mode_sum = dense_gmm_sum = zero
        gmm_count = mode_count = dense_gmm_count = zero.detach().clone()
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
                valid_velocity = target["velocity_mask"][rows, gt_index]
                if valid_velocity.any():
                    velocity = velocity + self.reg(
                        output["track_velocity"][rows[valid_velocity], slots[valid_velocity]],
                        target["current_velocity"][rows[valid_velocity], gt_index[valid_velocity]],
                    ).mean()
            # CoLIF ColifLoss::_forecast_loss consumes the existing loss-local
            # current-frame pairs only; future labels never enter the model.
            forecast = output.get("forecast")
            if forecast is not None and positive.any() and "future_mask" in target:
                valid_rows, valid_slots = torch.nonzero(
                    positive & forecast["forecast_valid_mask"], as_tuple=True)
                if valid_rows.numel():
                    gt_index = assigned[valid_rows, valid_slots]
                    future_receiver = self._future_relative_to_track(
                        target["future_xy"][valid_rows, gt_index],
                        target["current_boxes"][valid_rows, gt_index, None, :2],
                        output["track_box"][valid_rows, valid_slots, None, :2],
                    )
                    heading = output["track_box"][valid_rows, valid_slots, 6]
                    future_actor = self._receiver_displacement_to_actor(
                        future_receiver, heading[:, None]
                    )
                    layer_scores = forecast["loss_pred_scores_layers"][valid_rows, valid_slots]
                    layer_trajs = forecast["loss_pred_trajs_local_layers"][valid_rows, valid_slots]
                    layer_intentions = forecast["loss_intention_points_layers"][valid_rows, valid_slots]
                    layer_losses = [nearest_mode_gmm_loss(
                        layer_scores[:, layer], layer_trajs[:, layer], future_actor,
                        target["future_mask"][valid_rows, gt_index],
                        layer_intentions[:, layer])
                        for layer in range(layer_scores.shape[1])]
                    gmm_sum = gmm_sum + torch.stack(
                        [item["gmm"] for item in layer_losses]
                    ).mean()
                    mode_sum = mode_sum + torch.stack(
                        [item["mode"] for item in layer_losses]
                    ).mean()
                    gmm_count = gmm_count + layer_losses[-1]["gmm_count"]
                    mode_count = mode_count + layer_losses[-1]["mode_count"]
                dense_sum, dense_count = self._dense_context_future_loss(
                    output, target, assigned, zero
                )
                dense_gmm_sum = dense_gmm_sum + dense_sum
                dense_gmm_count = dense_gmm_count + dense_count
        gmm = gmm_sum / gmm_count.clamp_min(1.0)
        mode = mode_sum / mode_count.clamp_min(1.0)
        dense_gmm = dense_gmm_sum / dense_gmm_count.clamp_min(1.0)
        total = det + exist + box + velocity + gmm + mode + dense_gmm
        return {"total_loss": total, "L_det": det, "L_exist": exist, "L_box": box,
                "L_velocity": velocity, "L_gmm": gmm, "L_mode": mode,
                "L_dense_gmm": dense_gmm,
                "S_gmm": gmm_sum, "C_gmm": gmm_count,
                "S_mode": mode_sum, "C_mode": mode_count,
                "S_dense_gmm": dense_gmm_sum,
                "C_dense_gmm": dense_gmm_count,
                "match_count": match_count}
