"""Short-horizon multimodal forecast priors for online actor tracking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.colif.query_geometry import (
    transform_boxes_between_receiver_frames,
)


@dataclass
class ForecastPriorResult:
    """Current-receiver predictive distribution and single-state projection."""

    predicted_box: torch.Tensor
    predicted_velocity: torch.Tensor
    predicted_variance_xy: torch.Tensor
    mode_centers: torch.Tensor
    mode_variances: torch.Tensor
    mode_probabilities: torch.Tensor
    future_valid_mask: torch.Tensor
    selected_forecast_mask: torch.Tensor
    step_index: torch.Tensor


class ForecastMotionPrior(nn.Module):
    """Convert a previous forecast prefix into a current tracking prior.

    Association retains a multimodal distribution.  The query decoder and
    persistent state still require one metric reference, so they use the MAP
    component.  Constant velocity is an explicit mixture component and is the
    exact fallback for invalid or weakly ranked forecasts.
    """

    MODES = (
        "none",
        "association",
        "propagation_detached",
        "propagation_bptt",
        "detached",
        "recurrent_bptt",
    )

    def __init__(
        self,
        mode: str = "none",
        frame_period_s: float = 0.1,
        horizon_steps: int = 5,
        cv_component_weight: float = 0.15,
        cv_std_m: float = 1.0,
        min_std_m: float = 0.25,
        max_std_m: float = 3.0,
        mahalanobis_gate: float = 3.0,
        absolute_gate_m: float = 8.0,
        cv_match_distance_m: float = 4.0,
        proposal_score_weight: float = 0.25,
    ):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("Unknown forecast motion-prior mode: %s" % mode)
        if frame_period_s <= 0.0 or horizon_steps < 1:
            raise ValueError("Forecast prior clock/horizon must be positive")
        if not 0.0 < cv_component_weight < 1.0:
            raise ValueError("CV component weight must lie in (0,1)")
        if not 0.0 < min_std_m <= cv_std_m <= max_std_m:
            raise ValueError("Require 0 < min_std <= cv_std <= max_std")
        if mahalanobis_gate <= 0.0 or absolute_gate_m <= 0.0:
            raise ValueError("Association gates must be positive")
        if cv_match_distance_m <= 0.0 or proposal_score_weight < 0.0:
            raise ValueError("Invalid CV gate or proposal-score weight")
        self.mode = mode
        self.frame_period_s = float(frame_period_s)
        self.horizon_steps = int(horizon_steps)
        self.cv_component_weight = float(cv_component_weight)
        self.cv_variance = float(cv_std_m) ** 2
        self.min_variance = float(min_std_m) ** 2
        self.max_variance = float(max_std_m) ** 2
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.absolute_gate_m = float(absolute_gate_m)
        self.cv_match_distance_m = float(cv_match_distance_m)
        self.proposal_score_weight = float(proposal_score_weight)

    def forward(
        self,
        cv_box: torch.Tensor,
        cv_velocity: torch.Tensor,
        previous_box: torch.Tensor,
        future_mode_probabilities: torch.Tensor,
        future_relative_means: torch.Tensor,
        future_mode_variances: torch.Tensor,
        future_valid_mask: torch.Tensor,
        previous_pose: torch.Tensor,
        current_pose: torch.Tensor,
        previous_time_s: torch.Tensor,
        current_time_s: torch.Tensor,
    ) -> ForecastPriorResult:
        self._validate_shapes(
            cv_box,
            cv_velocity,
            previous_box,
            future_mode_probabilities,
            future_relative_means,
            future_mode_variances,
            future_valid_mask,
        )
        batch, queries, modes = future_mode_probabilities.shape
        available_steps = future_relative_means.shape[-2]
        if available_steps < 1:
            raise ValueError("Forecast state must store at least one future step")

        elapsed = current_time_s - previous_time_s
        raw_step = torch.round(elapsed / self.frame_period_s).long() - 1
        in_horizon = (raw_step >= 0) & (raw_step < available_steps)
        step_index = raw_step.clamp(0, available_steps - 1)
        mean_index = step_index.view(batch, 1, 1, 1, 1).expand(
            batch, queries, modes, 1, 2
        )
        variance_index = step_index.view(batch, 1, 1, 1).expand(
            batch, queries, modes, 1
        )
        selected_means = future_relative_means.gather(-2, mean_index).squeeze(-2)
        selected_variances = future_mode_variances.gather(
            -1, variance_index
        ).squeeze(-1)
        probabilities = future_mode_probabilities
        if self.mode in (
            "association", "propagation_detached", "detached"
        ):
            selected_means = selected_means.detach()
            selected_variances = selected_variances.detach()
            probabilities = probabilities.detach()

        finite = (
            torch.isfinite(selected_means).flatten(-2).all(dim=-1)
            & torch.isfinite(selected_variances).all(dim=-1)
            & torch.isfinite(probabilities).all(dim=-1)
        )
        probability_sum = probabilities.clamp_min(0.0).sum(dim=-1)
        spatial_valid = (
            future_valid_mask
            & in_horizon[:, None]
            & finite
            & (probability_sum > 0.0)
            & (self.mode != "none")
        )
        probabilities = probabilities.clamp_min(0.0)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        selected_variances = selected_variances.clamp(
            self.min_variance, self.max_variance
        )

        mode_boxes = previous_box.unsqueeze(2).expand(-1, -1, modes, -1).clone()
        mode_boxes[..., :2] = mode_boxes[..., :2] + selected_means
        mode_boxes = transform_boxes_between_receiver_frames(
            mode_boxes, previous_pose, current_pose
        )
        future_centers = mode_boxes[..., :2]

        # Keep CV just above one uniform forecast component for arbitrary K;
        # the configured weight is a lower bound calibrated for the main K=6.
        cv_weight = max(
            self.cv_component_weight,
            1.0 / float(modes + 1) + 1e-4,
        )
        cv_prob = torch.full_like(probabilities[..., :1], cv_weight)
        future_prob = probabilities * (1.0 - cv_weight)
        cv_prob = torch.where(spatial_valid[..., None], cv_prob, torch.ones_like(cv_prob))
        future_prob = torch.where(
            spatial_valid[..., None], future_prob, torch.zeros_like(future_prob)
        )
        mixture_probabilities = torch.cat((cv_prob, future_prob), dim=-1)
        mixture_centers = torch.cat(
            (cv_box[..., None, :2], future_centers), dim=-2
        )
        cv_variance = torch.full_like(
            selected_variances[..., :1], self.cv_variance
        )
        mixture_variances = torch.cat((cv_variance, selected_variances), dim=-1)

        selected_component = mixture_probabilities.argmax(dim=-1)
        selected_center = mixture_centers.gather(
            -2,
            selected_component[..., None, None].expand(batch, queries, 1, 2),
        ).squeeze(-2)
        selected_variance = mixture_variances.gather(
            -1, selected_component[..., None]
        ).squeeze(-1)
        use_spatial_propagation = self.mode in (
            "propagation_detached",
            "propagation_bptt",
            "detached",
            "recurrent_bptt",
        )
        predicted_box = cv_box.clone()
        if use_spatial_propagation:
            predicted_box[..., :2] = selected_center
        else:
            selected_variance = torch.full_like(selected_variance, self.cv_variance)
        selected_forecast = (
            spatial_valid & (selected_component > 0) & use_spatial_propagation
        )
        return ForecastPriorResult(
            predicted_box=predicted_box,
            predicted_velocity=cv_velocity,
            predicted_variance_xy=selected_variance,
            mode_centers=mixture_centers,
            mode_variances=mixture_variances,
            mode_probabilities=mixture_probabilities,
            future_valid_mask=spatial_valid,
            selected_forecast_mask=selected_forecast,
            step_index=step_index,
        )

    def association_cost(
        self,
        prior: ForecastPriorResult,
        proposal_boxes: torch.Tensor,
        proposal_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full query/proposal costs and pre-assignment validity."""

        if proposal_boxes.ndim != 3 or proposal_boxes.shape[-1] != 7:
            raise ValueError("proposal boxes must have shape [B,P,7]")
        if proposal_logits.shape != proposal_boxes.shape[:2]:
            raise ValueError("proposal logits and boxes disagree")
        cv_distance = torch.cdist(
            prior.mode_centers[..., 0, :].float(), proposal_boxes[..., :2].float()
        ).to(proposal_boxes.dtype)
        cv_valid = cv_distance <= self.cv_match_distance_m
        if (
            self.mode in ("none", "propagation_detached", "propagation_bptt")
            or not bool(prior.future_valid_mask.any())
        ):
            return cv_distance, cv_valid

        difference = (
            proposal_boxes[:, None, :, None, :2]
            - prior.mode_centers[:, :, None, :, :]
        )
        distance_squared = difference.square().sum(dim=-1)
        variance = prior.mode_variances[:, :, None, :].clamp(
            self.min_variance, self.max_variance
        )
        log_probability = torch.log(
            prior.mode_probabilities[:, :, None, :].clamp_min(1e-12)
        )
        mode_distance = torch.sqrt(
            (distance_squared / variance).clamp_min(0.0)
        ) - 0.25 * log_probability
        fallback_cost = self.cv_match_distance_m + mode_distance.amin(dim=-1)
        score_cost = self.proposal_score_weight * F.softplus(
            -proposal_logits[:, None, :]
        )
        forecast_cost = fallback_cost + score_cost
        minimum_mahalanobis = torch.sqrt(
            (distance_squared / variance).clamp_min(0.0)
        ).amin(dim=-1)
        minimum_absolute = torch.sqrt(distance_squared.clamp_min(0.0)).amin(dim=-1)
        forecast_valid = (
            (minimum_mahalanobis <= self.mahalanobis_gate)
            & (minimum_absolute <= self.absolute_gate_m)
        )
        # A reliable local CV explanation always wins.  The forecast mixture is
        # a secondary model only for rows with no CV-gated candidate, so it
        # cannot replace easy current-measurement associations.
        query_needs_fallback = (
            prior.future_valid_mask[..., None] & ~cv_valid.any(dim=-1, keepdim=True)
        )
        cost = torch.where(query_needs_fallback, forecast_cost, cv_distance)
        valid = torch.where(query_needs_fallback, forecast_valid, cv_valid)
        return cost, valid

    def compact_forecast(
        self,
        mode_probabilities: torch.Tensor,
        trajectories: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Keep only the causal short-horizon spatial GMM needed next frame."""

        if mode_probabilities.ndim != 3:
            raise ValueError("Forecast probabilities must have shape [B,Q,K]")
        if (
            trajectories.shape[:3] != mode_probabilities.shape
            or trajectories.ndim != 5
            or trajectories.shape[-1] < 4
        ):
            raise ValueError("Forecast trajectories must have shape [B,Q,K,T,>=4]")
        if valid_mask.shape != mode_probabilities.shape[:2]:
            raise ValueError("Forecast valid mask must have shape [B,Q]")
        if trajectories.shape[-2] < self.horizon_steps:
            raise ValueError("Forecast output is shorter than the stored prior horizon")
        probabilities = mode_probabilities.clamp_min(0.0)
        probability_sum = probabilities.sum(dim=-1, keepdim=True)
        probabilities = probabilities / probability_sum.clamp_min(1e-8)
        means = trajectories[..., : self.horizon_steps, :2]
        log_std = trajectories[..., : self.horizon_steps, 2:4]
        variances = torch.exp(2.0 * log_std).mean(dim=-1)
        finite = (
            torch.isfinite(probabilities).all(dim=-1)
            & torch.isfinite(means).flatten(-3).all(dim=-1)
            & torch.isfinite(variances).flatten(-2).all(dim=-1)
        )
        spatial_valid = (
            valid_mask
            & (probability_sum[..., 0] > 0.0)
            & finite
            & (self.mode != "none")
        )
        if self.mode not in ("propagation_bptt", "recurrent_bptt"):
            probabilities = probabilities.detach()
            means = means.detach()
            variances = variances.detach()
        return probabilities, means, variances, spatial_valid

    @staticmethod
    def _validate_shapes(
        cv_box,
        cv_velocity,
        previous_box,
        probabilities,
        means,
        variances,
        valid_mask,
    ):
        shape = cv_box.shape[:2]
        if cv_box.shape != (*shape, 7) or previous_box.shape != cv_box.shape:
            raise ValueError("CV and previous boxes must have shape [B,Q,7]")
        if cv_velocity.shape != (*shape, 2):
            raise ValueError("CV velocity must have shape [B,Q,2]")
        if probabilities.shape[:2] != shape or probabilities.ndim != 3:
            raise ValueError("Forecast probabilities must have shape [B,Q,K]")
        modes = probabilities.shape[-1]
        if means.shape[:3] != (*shape, modes) or means.shape[-1] != 2:
            raise ValueError("Forecast means must have shape [B,Q,K,S,2]")
        if variances.shape != means.shape[:-1]:
            raise ValueError("Forecast variances must have shape [B,Q,K,S]")
        if valid_mask.shape != shape:
            raise ValueError("Forecast valid mask must have shape [B,Q]")
