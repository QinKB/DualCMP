"""Runtime-only lifecycle policy for persistent query slots."""

from __future__ import annotations

import torch


class QueryLifecycle:
    def __init__(
        self,
        birth_threshold: float = 0.45,
        keep_threshold: float = 0.25,
        miss_ttl: int = 4,
        birth_suppression_radius: float = 2.0,
        max_active_queries: int = 32,
    ):
        if not 0.0 < keep_threshold <= birth_threshold < 1.0:
            raise ValueError("Require 0 < keep_threshold <= birth_threshold < 1")
        if miss_ttl < 0 or birth_suppression_radius <= 0.0 or max_active_queries < 1:
            raise ValueError("Invalid lifecycle TTL or suppression radius")
        self.birth_threshold = float(birth_threshold)
        self.keep_threshold = float(keep_threshold)
        self.miss_ttl = int(miss_ttl)
        self.birth_suppression_radius = float(birth_suppression_radius)
        self.max_active_queries = int(max_active_queries)

    @torch.no_grad()
    def __call__(
        self,
        scores: torch.Tensor,
        boxes: torch.Tensor,
        previous_active: torch.Tensor,
        previous_miss: torch.Tensor,
        previous_runtime_id: torch.Tensor,
        next_runtime_id: torch.Tensor,
        birth_candidate_mask: torch.Tensor | None = None,
        measurement_observed_mask: torch.Tensor | None = None,
    ) -> dict:
        if birth_candidate_mask is None:
            birth_candidate_mask = torch.ones_like(previous_active)
        if birth_candidate_mask.shape != previous_active.shape:
            raise ValueError("birth candidate mask shape disagrees with queries")
        if measurement_observed_mask is None:
            measurement_observed_mask = torch.ones_like(previous_active)
        if measurement_observed_mask.shape != previous_active.shape:
            raise ValueError("measurement observed mask disagrees with queries")
        # The sparse CMP measurement assignment is one-to-one.  A nearby
        # dense heatmap peak may raise several query scores, but only the query
        # actually assigned that measurement is observed.  Ignoring this mask
        # lets duplicate tracks reset their miss TTL from the same actor.
        observed = (
            previous_active
            & measurement_observed_mask
            & (scores >= self.keep_threshold)
        )
        miss = torch.where(
            previous_active & ~observed,
            previous_miss + 1,
            torch.zeros_like(previous_miss),
        )
        survive = previous_active & (observed | (miss <= self.miss_ttl))
        births = torch.zeros_like(previous_active)

        for batch_index in range(scores.shape[0]):
            occupied = boxes[batch_index, survive[batch_index], :2].tolist()
            candidates = torch.where(
                (~previous_active[batch_index])
                & birth_candidate_mask[batch_index]
                & (scores[batch_index] >= self.birth_threshold)
            )[0]
            candidates = candidates[
                torch.argsort(scores[batch_index, candidates], descending=True)
            ]
            capacity = max(
                0,
                self.max_active_queries - int(survive[batch_index].sum()),
            )
            accepted = 0
            for slot_tensor in candidates:
                if accepted >= capacity:
                    break
                slot = int(slot_tensor)
                center = boxes[batch_index, slot, :2]
                if occupied:
                    occupied_tensor = center.new_tensor(occupied)
                    if torch.cdist(center[None], occupied_tensor).min() < self.birth_suppression_radius:
                        continue
                births[batch_index, slot] = True
                occupied.append(center.tolist())
                accepted += 1

        active = survive | births
        runtime_id = previous_runtime_id.clone()
        next_id = next_runtime_id.clone()
        for batch_index in range(scores.shape[0]):
            birth_slots = torch.where(births[batch_index])[0]
            count = int(birth_slots.numel())
            if count:
                runtime_id[batch_index, birth_slots] = torch.arange(
                    int(next_id[batch_index]),
                    int(next_id[batch_index]) + count,
                    device=scores.device,
                    dtype=torch.long,
                )
                next_id[batch_index] += count
        runtime_id[~active] = -1
        age = torch.where(
            births,
            torch.ones_like(previous_miss),
            torch.where(active, torch.ones_like(previous_miss), torch.zeros_like(previous_miss)),
        )
        return {
            "observed_mask": observed,
            "birth_mask": births,
            "survive_mask": survive,
            "active_mask": active,
            "miss_count": torch.where(active, miss, torch.zeros_like(miss)),
            "runtime_id": runtime_id,
            "next_runtime_id": next_id,
            "initial_age": age,
        }
