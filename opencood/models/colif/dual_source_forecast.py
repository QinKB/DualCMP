"""Legacy-exact motion base with trajectory-aligned source refinement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from opencood.models.colif.motion_anchored_forecast import (
    MotionAnchoredForecastHead,
)
from opencood.models.colif.trajectory_corridor import (
    SourceCorridorRefiner,
    TrajectoryCorridorSampler,
)


class DualSourceMotionAnchoredForecastHead(nn.Module):
    """Preserve the released actor-source base and add a source corridor layer."""

    uses_dual_source_context = True

    def __init__(self, query_dim: int, args: dict) -> None:
        super().__init__()
        version = int(args.get("dual_source_version", 2))
        if version != 2:
            raise ValueError("dual_source_motion_anchored requires version 2")
        self.dual_source_version = version
        self.query_dim = int(query_dim)
        self.source_refinement_enabled = bool(
            args.get("source_refinement_enabled", True)
        )
        self.source_availability_bias_enabled = bool(
            args.get("source_availability_bias_enabled", True)
        )
        self.force_null_route = bool(args.get("force_null_route", False))
        self.source_dropout = float(args.get("source_dropout", 0.1))
        if not 0.0 <= self.source_dropout <= 1.0:
            raise ValueError("source_dropout must lie in [0,1]")

        # This must be the first randomly initialized v2 module.  It preserves
        # the mature A0 parameter order and its two-layer training contract.
        base_args = dict(args)
        base_args["inference_layer"] = str(
            args.get("base_inference_layer", "proposal")
        )
        base_args["refinement_type"] = "cross_mode"
        base_args["proposal_refinement"] = True
        self.motion_base = MotionAnchoredForecastHead(self.query_dim, base_args)
        self.modes = self.motion_base.modes
        self.future_frames = self.motion_base.future_frames
        self.history_frames = self.motion_base.history_frames
        self.min_forecast_history_steps = (
            self.motion_base.min_forecast_history_steps
        )
        self.temporal_history_enabled = self.motion_base.temporal_history_enabled
        self.enabled = self.motion_base.enabled
        self.inference_layer = str(
            args.get(
                "dual_source_inference_layer",
                args.get("inference_layer", "proposal"),
            )
        )
        if self.inference_layer == "final":
            self.inference_layer = "refined"
        if self.inference_layer not in ("proposal", "refined"):
            raise ValueError("dual source inference layer must be proposal or refined")
        self.loss_layers = self.motion_base.loss_layers + int(
            self.source_refinement_enabled
        )

        lidar_range = args.get("lidar_range")
        if lidar_range is None or len(lidar_range) != 6:
            raise ValueError("dual source v2 requires lidar_range")
        self.x_min = float(lidar_range[0])
        self.y_min = float(lidar_range[1])
        self.x_max = float(lidar_range[3])
        self.y_max = float(lidar_range[4])
        self.source_bev_dim = int(args.get("source_bev_dim", self.query_dim))
        self.corridor_sampler = TrajectoryCorridorSampler(
            lidar_range=lidar_range,
            lateral_offset_m=float(
                args.get("corridor_lateral_offset_m", 1.5)
            ),
            longitudinal_offset_m=float(
                args.get("corridor_longitudinal_offset_m", 1.5)
            ),
        )
        self.source_refiner = SourceCorridorRefiner(
            query_dim=self.query_dim,
            source_bev_dim=self.source_bev_dim,
            modes=self.modes,
            future_frames=self.future_frames,
            args=args,
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(7, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, self.query_dim),
        )
        self.age_projection = nn.Sequential(
            nn.Linear(3, self.query_dim),
            nn.GELU(),
            nn.Linear(self.query_dim, self.query_dim),
        )
        self.availability_projection = nn.Linear(
            self.source_bev_dim, self.query_dim
        )
        self.availability_head = nn.Sequential(
            nn.LayerNorm(self.query_dim * 4),
            nn.Linear(self.query_dim * 4, self.query_dim),
            nn.GELU(),
            nn.Dropout(float(args.get("dropout", 0.1))),
            nn.Linear(self.query_dim, 1),
        )
        self.availability_query_norm = nn.LayerNorm(self.query_dim)

    def _source_geometry(
        self, boxes: torch.Tensor, source_to_receiver: torch.Tensor
    ) -> torch.Tensor:
        source_origin = torch.nan_to_num(source_to_receiver[..., :2, 3])
        relative = boxes[..., None, :2] - source_origin[:, None, :, :]
        distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True)
        bearing = torch.atan2(relative[..., 1], relative[..., 0])
        actor_yaw = boxes[..., 6, None]
        source_yaw = torch.atan2(
            source_to_receiver[..., 1, 0], source_to_receiver[..., 0, 0]
        )[:, None, :]
        bearing_delta = bearing - actor_yaw
        heading_delta = source_yaw - actor_yaw
        diagonal = math.hypot(
            self.x_max - self.x_min, self.y_max - self.y_min
        )
        normalizer = boxes.new_tensor(
            (self.x_max - self.x_min, self.y_max - self.y_min)
        )
        raw = torch.cat(
            (
                relative / normalizer,
                distance / diagonal,
                torch.sin(bearing_delta)[..., None],
                torch.cos(bearing_delta)[..., None],
                torch.sin(heading_delta)[..., None],
                torch.cos(heading_delta)[..., None],
            ),
            dim=-1,
        )
        return self.geometry_projection(torch.nan_to_num(raw))

    def _source_age(self, delta_t_s: torch.Tensor) -> torch.Tensor:
        finite = torch.nan_to_num(
            delta_t_s, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp(-5.0, 5.0)
        raw = torch.stack(
            (
                finite / 5.0,
                finite.abs() / 5.0,
                torch.exp(-finite.abs()),
            ),
            dim=-1,
        )
        return self.age_projection(raw)

    @staticmethod
    def _restore_one_source(
        original: torch.Tensor, dropped: torch.Tensor
    ) -> torch.Tensor:
        lost = original.any(dim=-1) & ~dropped.any(dim=-1)
        if not lost.any():
            return dropped
        first = original.to(torch.long).argmax(dim=-1)
        repaired = dropped.clone()
        rows = torch.where(lost)[0]
        repaired[rows, first[rows]] = True
        return repaired

    def _availability_logits(
        self,
        samples: torch.Tensor,
        sample_valid: torch.Tensor,
        geometry: torch.Tensor,
        age: torch.Tensor,
        query_feat: torch.Tensor,
        cav_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = sample_valid[..., None].to(samples.dtype)
        # Pool K,T while preserving B,Q,L,C; spatial offsets were pooled
        # immediately after sampling to keep the recurrent clip memory bounded.
        summary = (samples * valid).sum(dim=(2, 3)) / valid.sum(
            dim=(2, 3)
        ).clamp_min(1.0)
        projected = self.availability_projection(summary)
        query = self.availability_query_norm(query_feat)[..., None, :].expand_as(
            projected
        )
        logits = self.availability_head(
            torch.cat((projected, geometry, age, query), dim=-1)
        ).squeeze(-1)
        return torch.where(
            cav_mask[:, None, :].bool(), logits, torch.zeros_like(logits)
        )

    def _source_rng_devices(self, reference: torch.Tensor):
        return [reference.device] if reference.is_cuda else []

    def empty_outputs(
        self,
        query_feat: torch.Tensor,
        valid_mask: torch.Tensor,
        source_count: int = 0,
    ) -> dict:
        base = self.motion_base.empty_outputs(query_feat, valid_mask)
        batch, queries = query_feat.shape[:2]
        steps = self.future_frames
        if self.source_refinement_enabled:
            base["loss_pred_scores_layers"] = torch.cat(
                (
                    base["loss_pred_scores_layers"],
                    base["loss_pred_scores"].unsqueeze(2),
                ),
                dim=2,
            )
            base["loss_pred_trajs_local_layers"] = torch.cat(
                (
                    base["loss_pred_trajs_local_layers"],
                    base["loss_pred_trajs_local"].unsqueeze(2),
                ),
                dim=2,
            )
            base["loss_intention_points_layers"] = torch.cat(
                (
                    base["loss_intention_points_layers"],
                    base["loss_intention_points"].unsqueeze(2),
                ),
                dim=2,
            )
        base.update(
            {
                "motion_proposal_pred_scores": base["pred_scores"],
                "motion_proposal_pred_trajs": base["pred_trajs"],
                "base_pred_scores": base["pred_scores"],
                "base_pred_trajs": base["pred_trajs"],
                "source_availability_logits": query_feat.new_zeros(
                    batch, queries, source_count
                ),
                "source_gate": query_feat.new_zeros(
                    batch, queries, self.modes
                ),
                "source_selection_weights": query_feat.new_zeros(
                    batch, queries, self.modes, steps, source_count
                ),
                "source_router_weights": query_feat.new_zeros(
                    batch, queries, self.modes, steps, source_count + 1
                ),
                "source_velocity_delta": query_feat.new_zeros(
                    batch, queries, self.modes, steps, 2
                ),
                "source_mean_delta": query_feat.new_zeros(
                    batch, queries, self.modes, steps, 2
                ),
                "source_refinement_active": self.source_refinement_enabled,
                "forecast_loss_layer_names": (
                    ("proposal", "cross_mode", "source_final")
                    if self.source_refinement_enabled
                    else ("proposal", "cross_mode")
                ),
                "forecast_loss_layer_weights": query_feat.new_tensor(
                    [0.5, 0.5, 1.0]
                    if self.source_refinement_enabled
                    else [0.5, 0.5]
                ),
            }
        )
        return base

    def forward(
        self,
        query_feat: torch.Tensor,
        boxes: torch.Tensor,
        valid_mask: torch.Tensor,
        history_box: torch.Tensor,
        history_mask: torch.Tensor,
        receiver_time_s: torch.Tensor,
        cooperative_context: torch.Tensor,
        source_bev: torch.Tensor | None,
        cav_mask: torch.Tensor | None,
        source_to_receiver: torch.Tensor | None,
        delta_t_s: torch.Tensor | None,
    ) -> dict:
        base = self.motion_base(
            query_feat,
            boxes,
            valid_mask,
            history_box,
            history_mask,
            receiver_time_s,
            cooperative_context=cooperative_context,
        )
        source_count = 0 if source_bev is None else int(source_bev.shape[1])
        if not self.enabled:
            return self.empty_outputs(query_feat, valid_mask, source_count)
        output = dict(base)
        output.update(
            {
                "motion_proposal_pred_scores": base["pred_scores"],
                "motion_proposal_pred_trajs": base["pred_trajs"],
                "base_pred_scores": base["pred_scores"],
                "base_pred_trajs": base["pred_trajs"],
                "source_refinement_active": self.source_refinement_enabled,
            }
        )
        if source_bev is None:
            empty = self.empty_outputs(query_feat, valid_mask, 0)
            for key in (
                "source_availability_logits",
                "source_gate",
                "source_selection_weights",
                "source_router_weights",
                "source_velocity_delta",
                "source_mean_delta",
            ):
                output[key] = empty[key]
            output["forecast_loss_layer_names"] = ("proposal", "cross_mode")
            output["forecast_loss_layer_weights"] = query_feat.new_tensor(
                [0.5, 0.5]
            )
            return output
        if cav_mask is None or source_to_receiver is None or delta_t_s is None:
            raise ValueError("dual source v2 requires source mask, transform, and age")
        batch, sources = source_bev.shape[:2]
        if cav_mask.shape != (batch, sources):
            raise ValueError("cav_mask must have shape [B,L]")
        if source_to_receiver.shape != (batch, sources, 4, 4):
            raise ValueError("source_to_receiver must have shape [B,L,4,4]")
        if delta_t_s.shape != (batch, sources):
            raise ValueError("delta_t_s must have shape [B,L]")

        base_logits = base["loss_pred_scores_layers"][:, :, 0]
        base_trajs = base["pred_trajs"]
        source_valid = cav_mask.bool()
        with torch.random.fork_rng(
            devices=self._source_rng_devices(query_feat), enabled=True
        ):
            if self.training and self.source_dropout > 0.0:
                kept = torch.rand(
                    source_valid.shape,
                    device=source_valid.device,
                    dtype=query_feat.dtype,
                ) >= self.source_dropout
                source_valid = self._restore_one_source(
                    source_valid, source_valid & kept
                )
            corridor = self.corridor_sampler(
                source_bev,
                boxes,
                base_trajs.detach()[..., :2],
                source_valid,
            )
            geometry = self._source_geometry(boxes, source_to_receiver)
            age_source = self._source_age(delta_t_s.to(query_feat))
            age = age_source[:, None].expand(
                batch, boxes.shape[1], sources, self.query_dim
            )
            availability = self._availability_logits(
                corridor.samples,
                corridor.valid,
                geometry,
                age,
                query_feat,
                cav_mask,
            )
            refined = self.source_refiner(
                corridor.samples,
                corridor.valid,
                base_trajs,
                base_logits,
                geometry,
                age,
                availability,
                base["forecast_valid_mask"].bool(),
                force_null=(
                    self.force_null_route or not self.source_refinement_enabled
                ),
                availability_bias_enabled=self.source_availability_bias_enabled,
            )

        motion_valid = base["forecast_valid_mask"].bool()
        float_trajectory_mask = motion_valid[..., None, None, None].to(
            query_feat.dtype
        )
        final_trajs = refined.final_trajs * float_trajectory_mask
        final_logits = torch.where(
            motion_valid[..., None],
            refined.final_logits,
            torch.zeros_like(refined.final_logits),
        )
        final_scores = torch.softmax(final_logits, dim=-1) * motion_valid[
            ..., None
        ].to(query_feat.dtype)
        gate = refined.source_gate * motion_valid[..., None].to(query_feat.dtype)
        selection = refined.selection_weights * motion_valid[
            ..., None, None, None
        ].to(query_feat.dtype)
        router = refined.router_weights * motion_valid[
            ..., None, None, None
        ].to(query_feat.dtype)
        velocity_delta = refined.velocity_delta * float_trajectory_mask
        mean_delta = refined.mean_delta * float_trajectory_mask

        if self.source_refinement_enabled:
            loss_scores_layers = torch.cat(
                (base["loss_pred_scores_layers"], final_logits.unsqueeze(2)),
                dim=2,
            )
            loss_trajs_layers = torch.cat(
                (
                    base["loss_pred_trajs_local_layers"],
                    final_trajs.unsqueeze(2),
                ),
                dim=2,
            )
            final_intention = final_trajs[..., -1, :2]
            intention_layers = torch.cat(
                (
                    base["loss_intention_points_layers"],
                    final_intention.unsqueeze(2),
                ),
                dim=2,
            )
            if self.inference_layer == "refined":
                output["pred_scores"] = final_scores
                output["pred_trajs"] = final_trajs
                output["pred_trajs_local"] = final_trajs
            output.update(
                {
                    "loss_pred_scores": final_logits,
                    "loss_pred_trajs_local": final_trajs,
                    "loss_pred_scores_layers": loss_scores_layers,
                    "loss_pred_trajs_local_layers": loss_trajs_layers,
                    "loss_intention_points": final_intention,
                    "loss_intention_points_layers": intention_layers,
                    "forecast_loss_layer_names": (
                        "proposal",
                        "cross_mode",
                        "source_final",
                    ),
                    "forecast_loss_layer_weights": query_feat.new_tensor(
                        [0.5, 0.5, 1.0]
                    ),
                }
            )
        else:
            output["forecast_loss_layer_names"] = ("proposal", "cross_mode")
            output["forecast_loss_layer_weights"] = query_feat.new_tensor(
                [0.5, 0.5]
            )
        probabilities = output["pred_scores"]
        output["future_uncertainty"] = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1, keepdim=True) / math.log(self.modes)
        output.update(
            {
                "source_availability_logits": availability
                * valid_mask[..., None].to(availability.dtype),
                "source_gate": gate,
                "source_selection_weights": selection,
                "source_router_weights": router,
                "source_velocity_delta": velocity_delta,
                "source_mean_delta": mean_delta,
            }
        )
        return output

    def freeze_unused_map_parameters(self) -> None:
        self.motion_base.freeze_unused_map_parameters()

    def freeze_inactive_parameters(
        self, actor_cooperative_fusion_enabled: bool = True
    ) -> None:
        self.motion_base.freeze_inactive_parameters(
            actor_cooperative_fusion_enabled=actor_cooperative_fusion_enabled
        )
        if not self.source_refinement_enabled:
            for module in (
                self.source_refiner,
                self.geometry_projection,
                self.age_projection,
                self.availability_projection,
                self.availability_head,
                self.availability_query_norm,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        if not self.source_availability_bias_enabled:
            self.source_refiner.availability_beta_raw.requires_grad = False

    def gradient_parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        base_ids = {id(parameter) for parameter in self.motion_base.parameters()}
        base = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) in base_ids
        ]
        source = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in base_ids
        ]
        return {"base": base, "source": source}
