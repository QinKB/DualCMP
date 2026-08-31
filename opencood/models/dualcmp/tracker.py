"""GT-free existing-first actor lifecycle over decoded detector proposals."""
from __future__ import annotations

from dataclasses import replace

import torch

from .geometry import propagate_state


class ActorTracker(torch.nn.Module):
    def __init__(self, max_miss=2, match_distance=4.0, duplicate_distance=1.0):
        super().__init__()
        self.max_miss = int(max_miss)
        self.match_distance = float(match_distance)
        self.duplicate_distance = float(duplicate_distance)
        if self.max_miss < 0 or self.match_distance <= 0 or self.duplicate_distance <= 0:
            raise ValueError("invalid tracker lifecycle thresholds")

    def forward(self, state, proposals, current_pose, current_time_s, max_dt_s):
        state, valid_rows, _ = propagate_state(state, current_pose, current_time_s, max_dt_s)
        box, score = state.box.clone(), state.score.clone()
        valid, active = state.valid_mask.clone(), torch.zeros_like(state.active_mask)
        miss, age, runtime_id = state.miss_count.clone(), state.age.clone(), state.runtime_id.clone()
        next_id = state.next_runtime_id.clone()
        last_update = state.last_update_time_s.clone()
        for batch_index in range(box.shape[0]):
            if not valid_rows[batch_index]:
                continue
            candidate = torch.nonzero(proposals["mask"][batch_index], as_tuple=False).flatten()
            assigned = set()
            # Existing actors first: each actor consumes nearest unused proposal.
            for slot in torch.nonzero(valid[batch_index], as_tuple=False).flatten().tolist():
                if candidate.numel():
                    distances = torch.linalg.vector_norm(proposals["boxes"][batch_index, candidate, :2] - box[batch_index, slot, :2], dim=1)
                    best = int(torch.argmin(distances))
                    proposal_index = int(candidate[best])
                    if distances[best] <= self.match_distance and proposal_index not in assigned:
                        box[batch_index, slot] = proposals["boxes"][batch_index, proposal_index]
                        score[batch_index, slot] = proposals["scores"][batch_index, proposal_index]
                        miss[batch_index, slot] = 0; age[batch_index, slot] += 1; active[batch_index, slot] = True; last_update[batch_index, slot] = current_time_s[batch_index]; assigned.add(proposal_index); continue
                miss[batch_index, slot] += 1
                if miss[batch_index, slot] > self.max_miss:
                    valid[batch_index, slot] = False; runtime_id[batch_index, slot] = -1; last_update[batch_index, slot] = float("nan")
            # Newborns only consume still-unused, non-duplicate proposals.
            for proposal_index in candidate.tolist():
                if proposal_index in assigned:
                    continue
                existing = torch.nonzero(valid[batch_index], as_tuple=False).flatten()
                if existing.numel() and torch.linalg.vector_norm(box[batch_index, existing, :2] - proposals["boxes"][batch_index, proposal_index, :2], dim=1).min() <= self.duplicate_distance:
                    continue
                free = torch.nonzero(~valid[batch_index], as_tuple=False).flatten()
                if not free.numel(): break
                slot = int(free[0]); valid[batch_index, slot] = True; active[batch_index, slot] = True
                box[batch_index, slot] = proposals["boxes"][batch_index, proposal_index]; score[batch_index, slot] = proposals["scores"][batch_index, proposal_index]
                runtime_id[batch_index, slot] = next_id[batch_index]; next_id[batch_index] += 1; miss[batch_index, slot] = 0; age[batch_index, slot] = 1; last_update[batch_index, slot] = current_time_s[batch_index]
        history_mask = state.history_mask & valid[:, :, None]
        history_box = torch.where(valid[:, :, None, None], state.history_box,
                                  torch.zeros_like(state.history_box))
        result = replace(state, box=box, score=score, valid_mask=valid, active_mask=active, miss_count=miss, age=age, runtime_id=runtime_id, next_runtime_id=next_id, last_update_time_s=last_update, history_mask=history_mask, history_box=history_box)
        result.validate()
        return result
