"""Forecast-conditioned cooperative observation-to-query association."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class ForecastAssociation(nn.Module):
    """Share one score function between differentiable training and deployment."""

    def __init__(
        self, query_dim=256, max_sources=2, sinkhorn_iterations=5,
        match_threshold=-6.0,
    ):
        super().__init__()
        self.query_dim = int(query_dim)
        self.max_sources = int(max_sources)
        self.sinkhorn_iterations = int(sinkhorn_iterations)
        self.match_threshold = float(match_threshold)
        self.query_projection = nn.Linear(query_dim, query_dim)
        self.observation_projection = nn.Linear(query_dim, query_dim)
        self.group_projection = nn.Linear(query_dim, query_dim)
        self.birth_logit = nn.Parameter(torch.tensor(0.0))
        self.appearance_weight = nn.Parameter(torch.tensor(1.0))
        self.geometry_weight = nn.Parameter(torch.tensor(0.05))
        self.motion_weight = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _interpolate_motion(motion_prior, capture_time_s):
        centers = motion_prior["mode_centers"]
        covariance = motion_prior["mode_covariance"]
        base = motion_prior["base_center"]
        batch, queries, modes, horizon, _ = centers.shape
        base_centers = base[:, :, None, None].expand(batch, queries, modes, 1, 2)
        centers = torch.cat((base_centers, centers), dim=3)
        base_covariance = torch.eye(
            2, device=covariance.device, dtype=covariance.dtype
        ).view(1, 1, 1, 1, 2, 2).expand(batch, queries, modes, 1, 2, 2)
        covariance = torch.cat((base_covariance * 0.25, covariance), dim=3)
        position = (
            (capture_time_s - motion_prior["previous_time_s"].unsqueeze(-1))
            / float(motion_prior["frame_period_s"])
        ).clamp(0.0, float(horizon))
        lower = torch.floor(position).long()
        upper = (lower + 1).clamp(max=horizon)
        batch_index = torch.arange(batch, device=centers.device)[:, None]
        low_center = centers[batch_index, :, :, lower].permute(0, 2, 1, 3, 4)
        high_center = centers[batch_index, :, :, upper].permute(0, 2, 1, 3, 4)
        low_covariance = covariance[batch_index, :, :, lower].permute(
            0, 2, 1, 3, 4, 5
        )
        high_covariance = covariance[batch_index, :, :, upper].permute(
            0, 2, 1, 3, 4, 5
        )
        fraction = (position - lower.to(position.dtype))[:, None, :, None, None]
        interpolated_center = low_center + fraction * (high_center - low_center)
        fraction_cov = fraction.unsqueeze(-1)
        interpolated_covariance = (
            low_covariance + fraction_cov * (high_covariance - low_covariance)
        )
        return interpolated_center, interpolated_covariance

    def _motion_log_likelihood(self, aligned, motion_prior):
        if motion_prior is None:
            return (
                aligned["box"].new_zeros(
                    aligned["box"].shape[0], 1, aligned["box"].shape[1]
                ),
                None,
            )
        center, covariance = self._interpolate_motion(
            motion_prior, aligned["capture_time_s"]
        )
        difference = aligned["box"][:, None, :, None, :2] - center
        covariance = covariance + torch.eye(
            2, device=covariance.device, dtype=covariance.dtype
        ).view(1, 1, 1, 1, 2, 2) * 1e-3
        # Analytic 2-D Gaussian avoids unstable inverse/logdet gradients under
        # CUDA autocast while preserving the full correlated covariance.
        var_x = covariance[..., 0, 0].float().clamp_min(1e-3)
        var_y = covariance[..., 1, 1].float().clamp_min(1e-3)
        cross = covariance[..., 0, 1].float()
        determinant = (var_x * var_y - cross.square()).clamp_min(1e-4)
        dx = difference[..., 0].float()
        dy = difference[..., 1].float()
        mahalanobis = (
            var_y * dx.square() - 2.0 * cross * dx * dy + var_x * dy.square()
        ) / determinant
        log_determinant = determinant.log()
        component = -0.5 * (
            mahalanobis + log_determinant + 2.0 * math.log(2.0 * math.pi)
        )
        component = component + motion_prior["mode_log_probability"].float()[:, :, None]
        likelihood = torch.logsumexp(component, dim=-1).clamp(-50.0, 10.0)
        if not torch.isfinite(likelihood).all():
            raise FloatingPointError("Non-finite full-mixture motion likelihood")
        mode_probability = motion_prior["mode_log_probability"].softmax(-1)
        expected_center = torch.einsum(
            "bqk,bqokd->bqod", mode_probability, center
        )
        return likelihood, expected_center

    def _soft_assignment(self, similarity, birth_score, active_mask, aligned):
        # Sinkhorn normalization is deliberately float32.  Bounded sigmoid
        # weights avoid both float16 exp overflow and ill-conditioned repeated
        # row/column projections for initially untrained logits.
        similarity_float = similarity.float()
        birth_score_float = birth_score.float()
        assignment = torch.zeros_like(similarity_float)
        birth_probability = torch.zeros_like(birth_score_float)
        for source in range(self.max_sources):
            source_mask = aligned["mask"] & (aligned["source"] == source)
            valid = active_mask.unsqueeze(-1) & source_mask.unsqueeze(1)
            weights = torch.sigmoid(similarity_float) * valid.to(
                similarity_float.dtype
            )
            birth = torch.sigmoid(birth_score_float) * source_mask.to(
                birth_score_float.dtype
            )
            for _ in range(self.sinkhorn_iterations):
                column_denominator = weights.sum(1) + birth
                safe_column_denominator = torch.where(
                    source_mask,
                    column_denominator.clamp_min(1e-4),
                    torch.ones_like(column_denominator),
                )
                weights = weights / safe_column_denominator.unsqueeze(1)
                birth = birth / safe_column_denominator
                row_scale = weights.sum(-1, keepdim=True).clamp_min(1.0)
                weights = weights / row_scale
            # The last row projection can leave unclaimed column mass; that is
            # the explicit birth alternative.  Rows and columns are both <= 1.
            birth = (1.0 - weights.sum(1)).clamp(0.0, 1.0) * source_mask
            assignment = assignment + weights
            birth_probability = birth_probability + birth
        null_probability = (1.0 - assignment.sum(-1)).clamp(0.0, 1.0)
        return assignment, birth_probability, null_probability

    def _hard_assignment(self, similarity, active_mask, aligned):
        assignment = torch.zeros_like(similarity)
        birth_probability = aligned["mask"].to(similarity.dtype)
        for batch_index in range(similarity.shape[0]):
            active_queries = torch.nonzero(
                active_mask[batch_index], as_tuple=False
            ).flatten()
            if not active_queries.numel():
                continue
            for source in range(self.max_sources):
                source_observations = torch.nonzero(
                    aligned["mask"][batch_index]
                    & (aligned["source"][batch_index] == source),
                    as_tuple=False,
                ).flatten()
                if not source_observations.numel():
                    continue
                score = similarity[batch_index][active_queries][:, source_observations]
                rows, columns = linear_sum_assignment(-score.detach().float().cpu().numpy())
                for row, column in zip(rows.tolist(), columns.tolist()):
                    if float(score[row, column]) < self.match_threshold:
                        continue
                    query_index = int(active_queries[row])
                    observation_index = int(source_observations[column])
                    assignment[batch_index, query_index, observation_index] = 1.0
                    birth_probability[batch_index, observation_index] = 0.0
        return (
            assignment,
            birth_probability,
            (1.0 - assignment.sum(-1)).clamp(0.0, 1.0),
        )

    def forward(self, state, aligned, prior_box, query_prior, motion_prior=None):
        query_key = F.normalize(self.query_projection(query_prior), dim=-1)
        observation_key = F.normalize(
            self.observation_projection(aligned["feat"]), dim=-1
        )
        appearance = torch.einsum("bqd,bod->bqo", query_key, observation_key)
        displacement = (
            prior_box[..., :2].unsqueeze(2) - aligned["box"][..., :2].unsqueeze(1)
        )
        geometry = displacement.square().sum(-1)
        motion, motion_center_at_capture = self._motion_log_likelihood(
            aligned, motion_prior
        )
        similarity = (
            self.appearance_weight * appearance
            - self.geometry_weight.abs() * geometry
            + self.motion_weight.abs() * motion
            + torch.log(aligned["score"].clamp_min(1e-4)).unsqueeze(1)
        )
        similarity = similarity.masked_fill(~state.active_mask.unsqueeze(-1), -1e4)
        similarity = similarity.masked_fill(~aligned["mask"].unsqueeze(1), -1e4)
        if not torch.isfinite(similarity).all():
            raise FloatingPointError("Non-finite cooperative association similarity")
        birth_score = self.birth_logit + torch.log(aligned["score"].clamp_min(1e-4))
        birth_score = birth_score.masked_fill(~aligned["mask"], -1e4)
        if self.training:
            assignment, birth, null = self._soft_assignment(
                similarity, birth_score, state.active_mask, aligned
            )
        else:
            assignment, birth, null = self._hard_assignment(
                similarity, state.active_mask, aligned
            )

        group_key = F.normalize(self.group_projection(aligned["feat"]), dim=-1)
        group_logits = torch.einsum("bod,bpd->bop", group_key, group_key)
        group_distance = (
            aligned["box"][..., None, :2] - aligned["box"][:, None, :, :2]
        ).square().sum(-1)
        group_logits = 4.0 * group_logits - 0.5 * group_distance
        pair_valid = (
            aligned["mask"].unsqueeze(2) & aligned["mask"].unsqueeze(1)
            & (aligned["source"].unsqueeze(2) != aligned["source"].unsqueeze(1))
        )
        group_logits = group_logits.masked_fill(~pair_valid, -1e4)
        return {
            "similarity": similarity,
            "assignment": assignment,
            "birth_probability": birth,
            "null_probability": null,
            "birth_group_logits": group_logits,
            "birth_pair_valid": pair_valid,
            "motion_log_likelihood": motion,
            "motion_center_at_capture": motion_center_at_capture,
        }
