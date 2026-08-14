"""Persistent cooperative query update, birth, memory and lifecycle."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from opencood.models.colif.forecast_association import ForecastAssociation
from opencood.models.colif.source_memory import SourceMemoryUpdater


class PersistentQueryBank(nn.Module):
    def __init__(
        self, query_dim=256, memory_dim=128, max_sources=2, memory_len=6,
        miss_ttl=4, birth_threshold=0.5, observation_score_threshold=0.2,
        birth_merge_distance=2.0, sinkhorn_iterations=5,
        association_match_threshold=-6.0, birth_group_threshold=0.5,
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.max_sources = int(max_sources)
        self.miss_ttl = int(miss_ttl)
        self.birth_threshold = float(birth_threshold)
        self.observation_score_threshold = float(observation_score_threshold)
        self.birth_merge_distance = float(birth_merge_distance)
        self.birth_group_threshold = float(birth_group_threshold)
        self.association = ForecastAssociation(
            query_dim=query_dim,
            max_sources=max_sources,
            sinkhorn_iterations=sinkhorn_iterations,
            match_threshold=association_match_threshold,
        )
        self.evidence_fusion = nn.Linear(query_dim * 2, query_dim)
        self.update_cell = nn.GRUCell(query_dim, query_dim)
        self.birth_init = nn.Sequential(
            nn.Linear(query_dim, query_dim), nn.ReLU(inplace=True),
            nn.Linear(query_dim, query_dim),
        )
        self.existence = nn.Linear(query_dim, 1)
        self.source_memory = SourceMemoryUpdater(
            query_dim, memory_dim, max_sources, memory_len
        )

    @staticmethod
    def _weighted_read(weights, values):
        denominator = weights.sum(-1, keepdim=True)
        read = torch.einsum("bqo,bod->bqd", weights, values)
        return read / denominator.clamp_min(1e-6), denominator

    def _birth_groups(
        self, batch_index, candidates, aligned, group_logits, active_centers
    ):
        candidate_score = (
            candidates["probability"] * candidates["score"]
        )
        order = torch.argsort(candidate_score, descending=True)
        indices = candidates["index"][order]
        consumed = set()
        groups = []
        for seed in indices.tolist():
            if seed in consumed:
                continue
            seed_center = aligned["box"][batch_index, seed, :2]
            if active_centers.numel():
                if torch.norm(active_centers - seed_center, dim=-1).min() < self.birth_merge_distance:
                    consumed.add(seed)
                    continue
            group = [seed]
            consumed.add(seed)
            seed_source = int(aligned["source"][batch_index, seed])
            for candidate in indices.tolist():
                if candidate in consumed:
                    continue
                if int(aligned["source"][batch_index, candidate]) == seed_source:
                    continue
                distance = torch.norm(
                    aligned["box"][batch_index, candidate, :2] - seed_center
                )
                group_probability = torch.sigmoid(
                    group_logits[batch_index, seed, candidate]
                )
                if (
                    distance < self.birth_merge_distance
                    and group_probability >= self.birth_group_threshold
                ):
                    group.append(candidate)
                    consumed.add(candidate)
            groups.append(group)
        return groups

    def forward(self, state, aligned, prior_box, query_prior, motion_prior=None):
        batch_size, max_queries, _ = query_prior.shape
        observations = aligned["feat"]
        pre_update_active_mask = state.active_mask.clone()
        pre_update_runtime_id = state.runtime_id.clone()
        pre_update_generation = state.generation.clone()

        association = self.association(
            state, aligned, prior_box, query_prior, motion_prior
        )
        assignment = association["assignment"]
        observation_read, weight_sum = self._weighted_read(assignment, observations)
        observation_box_for_query = aligned["box"].unsqueeze(1).expand(
            -1, max_queries, -1, -1
        )
        if association["motion_center_at_capture"] is not None:
            observation_box_for_query = observation_box_for_query.clone()
            observation_box_for_query[..., :2] = (
                observation_box_for_query[..., :2]
                + prior_box[..., None, :2]
                - association["motion_center_at_capture"]
            )
        box_denominator = assignment.sum(-1, keepdim=True)
        box_read = torch.einsum(
            "bqo,bqod->bqd", assignment, observation_box_for_query
        ) / box_denominator.clamp_min(1e-6)
        observed = weight_sum.squeeze(-1) > 0.05

        memory_state = replace(state, query_feat=query_prior)
        memory, memory_mask, memory_time, memory_read = self.source_memory(
            memory_state, assignment, aligned
        )
        update_input = self.evidence_fusion(
            torch.cat((observation_read, memory_read), dim=-1)
        )
        updated = self.update_cell(
            update_input.reshape(-1, self.query_dim),
            query_prior.reshape(-1, self.query_dim),
        ).view(batch_size, max_queries, self.query_dim)
        query_feat = torch.where(observed.unsqueeze(-1), updated, query_prior)
        ref_box = torch.where(observed.unsqueeze(-1), box_read, prior_box)

        active_mask = state.active_mask.clone()
        miss_count = torch.where(
            observed,
            torch.zeros_like(state.miss_count),
            state.miss_count + active_mask.long(),
        )
        age = state.age + active_mask.long()
        expired = active_mask & (miss_count > self.miss_ttl)
        active_mask = active_mask & ~expired
        age = torch.where(expired, torch.zeros_like(age), age)
        runtime_id = state.runtime_id.clone()
        runtime_id[expired] = -1
        generation = state.generation.clone()
        next_runtime_id = state.next_runtime_id.clone()

        birth_slot_mask = torch.zeros_like(active_mask)
        birth_features = torch.zeros_like(query_feat)
        birth_boxes = torch.zeros_like(ref_box)
        birth_assignment = torch.zeros_like(assignment)
        birth_group_indices = []
        for batch_index in range(batch_size):
            candidate_indices = torch.nonzero(
                aligned["mask"][batch_index]
                & (association["birth_probability"][batch_index] >= self.birth_threshold)
                & (aligned["score"][batch_index] >= self.observation_score_threshold),
                as_tuple=False,
            ).flatten()
            if not candidate_indices.numel():
                birth_group_indices.append([])
                continue
            active_centers = ref_box[batch_index, active_mask[batch_index], :2]
            groups = self._birth_groups(
                batch_index,
                {
                    "index": candidate_indices,
                    "probability": association["birth_probability"][
                        batch_index, candidate_indices
                    ],
                    "score": aligned["score"][batch_index, candidate_indices],
                },
                aligned,
                association["birth_group_logits"],
                active_centers,
            )
            allocated_groups = []
            for group in groups:
                free_slots = torch.nonzero(
                    ~active_mask[batch_index], as_tuple=False
                ).flatten()
                if free_slots.numel():
                    slot = int(free_slots[0])
                else:
                    recyclable = torch.nonzero(
                        active_mask[batch_index] & ~observed[batch_index],
                        as_tuple=False,
                    ).flatten()
                    if not recyclable.numel():
                        break
                    recycle_score = (
                        torch.sigmoid(state.exist_logit[batch_index, recyclable])
                        - 0.1 * miss_count[batch_index, recyclable].to(query_feat.dtype)
                    )
                    slot = int(recyclable[torch.argmin(recycle_score)])
                group_tensor = torch.tensor(
                    group, device=query_feat.device, dtype=torch.long
                )
                group_weights = aligned["score"][batch_index, group_tensor]
                group_weights = group_weights / group_weights.sum().clamp_min(1e-6)
                feature = torch.einsum(
                    "o,od->d", group_weights,
                    observations[batch_index, group_tensor],
                )
                box = torch.einsum(
                    "o,od->d", group_weights,
                    aligned["box"][batch_index, group_tensor],
                )
                birth_slot_mask[batch_index, slot] = True
                birth_features[batch_index, slot] = self.birth_init(feature)
                birth_boxes[batch_index, slot] = box
                birth_assignment[batch_index, slot, group_tensor] = 1.0
                active_mask[batch_index, slot] = True
                miss_count[batch_index, slot] = 0
                age[batch_index, slot] = 1
                runtime_id[batch_index, slot] = next_runtime_id[batch_index]
                next_runtime_id[batch_index] += 1
                generation[batch_index, slot] += 1
                allocated_groups.append((slot, tuple(group)))
            birth_group_indices.append(allocated_groups)

        query_feat = torch.where(
            birth_slot_mask.unsqueeze(-1), birth_features, query_feat
        )
        ref_box = torch.where(birth_slot_mask.unsqueeze(-1), birth_boxes, ref_box)
        if birth_slot_mask.any():
            birth_memory_state = replace(
                state,
                query_feat=query_feat,
                source_memory=memory,
                source_memory_mask=memory_mask,
                source_memory_time=memory_time,
            )
            memory, memory_mask, memory_time, _ = self.source_memory(
                birth_memory_state, birth_assignment, aligned
            )

        existence_logit = self.existence(query_feat).squeeze(-1)
        lifecycle_state = torch.zeros_like(miss_count)
        lifecycle_state[active_mask & (miss_count == 0)] = 1
        lifecycle_state[active_mask & (miss_count > 0)] = 2
        next_state = replace(
            state,
            query_feat=query_feat,
            ref_box=ref_box,
            exist_logit=existence_logit,
            active_mask=active_mask,
            lifecycle_state=lifecycle_state,
            age=age,
            miss_count=miss_count,
            runtime_id=runtime_id,
            generation=generation,
            next_runtime_id=next_runtime_id,
            source_memory=memory,
            source_memory_mask=memory_mask,
            source_memory_time=memory_time,
        )
        return next_state, {
            "association_logits": association["similarity"],
            "assignment_probability": assignment,
            "birth_probability": association["birth_probability"],
            "null_probability": association["null_probability"],
            "birth_group_logits": association["birth_group_logits"],
            "birth_pair_valid": association["birth_pair_valid"],
            "motion_log_likelihood": association["motion_log_likelihood"],
            "birth_assignment": birth_assignment,
            "birth_group_indices": birth_group_indices,
            "observed_mask": observed,
            "existence_logit": existence_logit,
            "lifecycle_state": lifecycle_state,
            "pre_update_active_mask": pre_update_active_mask,
            "pre_update_runtime_id": pre_update_runtime_id,
            "pre_update_generation": pre_update_generation,
        }
