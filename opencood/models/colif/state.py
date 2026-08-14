"""Tensor-only runtime state for fused-BEV persistent forecast queries."""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class GlobalForecastState:
    """One physical query bank per receiver, with no source-specific identity.

    The state deliberately stores only a compact future cue.  Full predicted
    trajectories are frame outputs and are never copied into recurrence.
    """

    query_feat: torch.Tensor
    ref_box: torch.Tensor
    # Filter state used by runtime association.  ``ref_box`` may contain a
    # forecast-imputed output during a measurement gap, but that imputation
    # must never become the next CV association origin.
    association_ref_box: torch.Tensor
    velocity: torch.Tensor
    exist_logit: torch.Tensor
    active_mask: torch.Tensor
    age: torch.Tensor
    miss_count: torch.Tensor
    runtime_id: torch.Tensor
    generation: torch.Tensor
    next_runtime_id: torch.Tensor
    history_feat: torch.Tensor
    # Tracking memory is continuously re-expressed in the latest receiver
    # frame.  Forecast memory keeps each observation in its capture-time
    # receiver frame, matching CMP/MTR's actual trajectory-cache contract.
    history_box: torch.Tensor
    forecast_history_box: torch.Tensor
    history_mask: torch.Tensor
    future_cue: torch.Tensor
    future_uncertainty: torch.Tensor
    future_valid_mask: torch.Tensor
    future_mode_probabilities: torch.Tensor
    future_relative_means: torch.Tensor
    future_mode_variances: torch.Tensor
    future_spatial_valid_mask: torch.Tensor
    receiver_pose: torch.Tensor
    receiver_time_s: torch.Tensor

    @classmethod
    def empty(
        cls,
        query_feat: torch.Tensor,
        ref_box: torch.Tensor,
        memory_len: int,
        receiver_pose: torch.Tensor,
        receiver_time_s: torch.Tensor,
        future_modes: int = 0,
        future_horizon_steps: int = 0,
    ) -> "GlobalForecastState":
        batch_size, max_queries, query_dim = query_feat.shape
        device = query_feat.device
        dtype = query_feat.dtype
        return cls(
            query_feat=query_feat,
            ref_box=ref_box,
            association_ref_box=ref_box.clone(),
            velocity=torch.zeros(batch_size, max_queries, 2, device=device, dtype=dtype),
            exist_logit=torch.full(
                (batch_size, max_queries), -10.0, device=device, dtype=dtype
            ),
            active_mask=torch.zeros(
                batch_size, max_queries, dtype=torch.bool, device=device
            ),
            age=torch.zeros(batch_size, max_queries, dtype=torch.long, device=device),
            miss_count=torch.zeros(
                batch_size, max_queries, dtype=torch.long, device=device
            ),
            runtime_id=torch.full(
                (batch_size, max_queries), -1, dtype=torch.long, device=device
            ),
            generation=torch.zeros(
                batch_size, max_queries, dtype=torch.long, device=device
            ),
            next_runtime_id=torch.zeros(batch_size, dtype=torch.long, device=device),
            history_feat=torch.zeros(
                batch_size,
                max_queries,
                memory_len,
                query_dim,
                device=device,
                dtype=dtype,
            ),
            history_box=torch.zeros(
                batch_size, max_queries, memory_len, 7, device=device, dtype=dtype
            ),
            forecast_history_box=torch.zeros(
                batch_size, max_queries, memory_len, 7, device=device, dtype=dtype
            ),
            history_mask=torch.zeros(
                batch_size,
                max_queries,
                memory_len,
                dtype=torch.bool,
                device=device,
            ),
            future_cue=torch.zeros_like(query_feat),
            future_uncertainty=torch.zeros(
                batch_size, max_queries, 1, device=device, dtype=dtype
            ),
            future_valid_mask=torch.zeros(
                batch_size, max_queries, dtype=torch.bool, device=device
            ),
            future_mode_probabilities=torch.zeros(
                batch_size,
                max_queries,
                future_modes,
                device=device,
                dtype=dtype,
            ),
            future_relative_means=torch.zeros(
                batch_size,
                max_queries,
                future_modes,
                future_horizon_steps,
                2,
                device=device,
                dtype=dtype,
            ),
            future_mode_variances=torch.zeros(
                batch_size,
                max_queries,
                future_modes,
                future_horizon_steps,
                device=device,
                dtype=dtype,
            ),
            future_spatial_valid_mask=torch.zeros(
                batch_size, max_queries, dtype=torch.bool, device=device
            ),
            receiver_pose=receiver_pose,
            receiver_time_s=receiver_time_s,
        )

    def validate(self) -> None:
        shape = self.active_mask.shape
        if self.query_feat.shape[:2] != shape:
            raise ValueError("query_feat and active_mask disagree")
        if self.ref_box.shape != (*shape, 7):
            raise ValueError("ref_box and active_mask disagree")
        if self.association_ref_box.shape != (*shape, 7):
            raise ValueError("association_ref_box and active_mask disagree")
        if self.velocity.shape != (*shape, 2):
            raise ValueError("velocity and active_mask disagree")
        if self.exist_logit.shape != shape:
            raise ValueError("exist_logit and active_mask disagree")
        if self.history_feat.shape[:2] != shape:
            raise ValueError("history and active_mask disagree")
        if self.forecast_history_box.shape != self.history_box.shape:
            raise ValueError("tracking and forecast histories disagree")
        if self.future_cue.shape != self.query_feat.shape:
            raise ValueError("future_cue and query_feat disagree")
        if self.future_valid_mask.shape != shape:
            raise ValueError("future_valid_mask and active_mask disagree")
        if self.future_mode_probabilities.shape[:2] != shape:
            raise ValueError("future mode probabilities and active_mask disagree")
        modes = self.future_mode_probabilities.shape[-1]
        if (
            self.future_relative_means.shape[:3] != (*shape, modes)
            or self.future_relative_means.shape[-1] != 2
        ):
            raise ValueError("future relative means have an invalid shape")
        if self.future_mode_variances.shape != self.future_relative_means.shape[:-1]:
            raise ValueError("future mode variances and means disagree")
        if self.future_spatial_valid_mask.shape != shape:
            raise ValueError("future spatial valid mask and active_mask disagree")

    def detach_differentiable_tensors(self) -> "GlobalForecastState":
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.detach() if value.is_floating_point() else value
        return GlobalForecastState(**values)

    def replace_rows(
        self, reset_mask: torch.Tensor, replacement: "GlobalForecastState"
    ) -> "GlobalForecastState":
        """Replace complete batch rows at scene/receiver boundaries."""

        reset_mask = reset_mask.to(self.active_mask.device, dtype=torch.bool)
        if reset_mask.shape != (self.active_mask.shape[0],):
            raise ValueError("reset_mask must have shape [batch_size]")
        values = {}
        for item in fields(self):
            current = getattr(self, item.name)
            fresh = getattr(replacement, item.name)
            view = reset_mask.view(-1, *([1] * (current.ndim - 1)))
            values[item.name] = torch.where(view, fresh, current)
        return GlobalForecastState(**values)

    # Backward-compatible spelling used by the sequence trainer.  A caller
    # must now provide a correctly initialized replacement so reset rows do
    # not become zero-valued pseudo queries.
    def reset_rows(self, reset_mask, replacement=None):
        if replacement is None:
            raise ValueError("reset_rows requires a freshly initialized replacement")
        return self.replace_rows(reset_mask, replacement)
