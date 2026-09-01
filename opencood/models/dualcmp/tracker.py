"""GT-free existing-first actor lifecycle over decoded detector proposals."""
from __future__ import annotations

from dataclasses import replace

import torch

from .geometry import propagate_state
from .ab3dmot_assignment import associate_car_giou
from .kalman import box_to_measurement, initialize, mean_to_box, orientation_correction, predict, update


class ActorTracker(torch.nn.Module):
    def __init__(self, max_miss=2, duplicate_distance=1.0):
        super().__init__()
        self.max_miss = int(max_miss)
        self.duplicate_distance = float(duplicate_distance)
        if self.max_miss < 0 or self.duplicate_distance <= 0:
            raise ValueError("invalid tracker max_miss/duplicate_distance")

    def forward(self, state, proposals, current_pose, current_time_s, max_dt_s):
        state, valid_rows, _ = propagate_state(state, current_pose, current_time_s, max_dt_s)
        mean, covariance = state.filter_mean.clone(), state.filter_cov.clone()
        valid_before = state.valid_mask & valid_rows[:, None]
        if valid_before.any():
            predicted_mean, predicted_covariance = predict(mean[valid_before], covariance[valid_before])
            mean[valid_before], covariance[valid_before] = predicted_mean, predicted_covariance
        box, score = mean_to_box(mean), state.score.clone()
        velocity = mean[..., 7:9].clone()
        valid, active = state.valid_mask.clone(), torch.zeros_like(state.active_mask)
        miss, age, runtime_id = state.miss_count.clone(), state.age.clone(), state.runtime_id.clone()
        next_id = state.next_runtime_id.clone()
        last_update = state.last_update_time_s.clone()
        for batch_index in range(box.shape[0]):
            if not valid_rows[batch_index]:
                continue
            candidate = torch.nonzero(proposals["mask"][batch_index], as_tuple=False).flatten()
            assigned = set()
            slots = torch.nonzero(valid[batch_index], as_tuple=False).flatten().tolist()
            matched = {}
            if slots and candidate.numel():
                local = associate_car_giou(proposals["boxes"][batch_index, candidate],
                                           box[batch_index, slots])
                matched = {slots[track_index]: int(candidate[proposal_index])
                           for track_index, proposal_index in local.items()}
            # Existing actors first: bundled AB3DMOT Car GIoU Hungarian.
            for slot in slots:
                proposal_index = matched.get(slot)
                if proposal_index is not None:
                    measurement = box_to_measurement(proposals["boxes"][batch_index, proposal_index])
                    previous_mean = mean[batch_index, slot].clone()
                    previous_covariance = covariance[batch_index, slot]
                    corrected_yaw, measurement_yaw = orientation_correction(
                        previous_mean[3], measurement[3])
                    previous_mean = torch.cat((previous_mean[:3], corrected_yaw.reshape(1), previous_mean[4:]))
                    measurement = torch.cat((measurement[:3], measurement_yaw.reshape(1), measurement[4:]))
                    mean[batch_index, slot], covariance[batch_index, slot] = update(
                        previous_mean, previous_covariance, measurement)
                    box[batch_index, slot] = mean_to_box(mean[batch_index, slot])
                    velocity[batch_index, slot] = mean[batch_index, slot, 7:9]
                    score[batch_index, slot] = proposals["scores"][batch_index, proposal_index]
                    miss[batch_index, slot] = 0; age[batch_index, slot] += 1; active[batch_index, slot] = True; last_update[batch_index, slot] = current_time_s[batch_index]; assigned.add(proposal_index); continue
                miss[batch_index, slot] += 1
                if miss[batch_index, slot] >= self.max_miss:
                    valid[batch_index, slot] = False; runtime_id[batch_index, slot] = -1; last_update[batch_index, slot] = float("nan")
            # Newborns consume unused proposals outside the existing local
            # duplicate-distance gate; association itself is AB3 GIoU above.
            for proposal_index in candidate.tolist():
                if proposal_index in assigned:
                    continue
                existing = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
                if existing.numel() and torch.linalg.vector_norm(box[batch_index, existing, :2] - proposals["boxes"][batch_index, proposal_index, :2], dim=1).min() <= self.duplicate_distance:
                    continue
                free = torch.nonzero(~valid[batch_index], as_tuple=False).flatten()
                if not free.numel(): break
                slot = int(free[0]); valid[batch_index, slot] = True; active[batch_index, slot] = True
                measurement = box_to_measurement(proposals["boxes"][batch_index, proposal_index])
                mean[batch_index, slot], covariance[batch_index, slot] = initialize(measurement)
                box[batch_index, slot] = mean_to_box(mean[batch_index, slot]); velocity[batch_index, slot] = mean[batch_index, slot, 7:9]
                score[batch_index, slot] = proposals["scores"][batch_index, proposal_index]
                runtime_id[batch_index, slot] = next_id[batch_index]; next_id[batch_index] += 1; miss[batch_index, slot] = 0; age[batch_index, slot] = 1; last_update[batch_index, slot] = current_time_s[batch_index]
        history_mask = state.history_mask & valid[:, :, None]
        history_box = torch.where(valid[:, :, None, None], state.history_box,
                                  torch.zeros_like(state.history_box))
        mean = torch.where(valid[:, :, None], mean, torch.zeros_like(mean))
        covariance = torch.where(valid[:, :, None, None], covariance, torch.zeros_like(covariance))
        result = replace(state, box=box, velocity=velocity, score=score, valid_mask=valid, active_mask=active, miss_count=miss, age=age, runtime_id=runtime_id, next_runtime_id=next_id, last_update_time_s=last_update, history_mask=history_mask, history_box=history_box, filter_mean=mean, filter_cov=covariance)
        result.validate()
        return result
