"""Masked nearest-mode bivariate GMM loss copied from CoLIF 6b8788e."""
import torch
import torch.nn.functional as F

from opencood.models.dualcmp.mtr_vendor import nll_loss_gmm_direct

PARTIAL_FORECAST_SUPERVISION = "partial_future_v1"
COMPLETE_FORECAST_SUPERVISION = "cmp_complete_50_v1"
FORECAST_SUPERVISIONS = (PARTIAL_FORECAST_SUPERVISION, COMPLETE_FORECAST_SUPERVISION)


def forecast_supervision_mask(future_mask, contract):
    if contract not in FORECAST_SUPERVISIONS:
        raise ValueError("unknown forecast supervision contract")
    if future_mask.ndim < 2 or future_mask.dtype != torch.bool:
        raise ValueError("forecast supervision requires a boolean actor/time mask")
    if contract == COMPLETE_FORECAST_SUPERVISION:
        if future_mask.shape[-1] != 50:
            raise ValueError("CMP complete supervision requires fifty 10Hz future frames")
        return future_mask.all(-1)
    return future_mask.any(-1)


def nearest_mode_gmm_loss(mode_logits, trajectories, target_xy, future_mask,
                          intention_points=None):
    mode_logits_float, trajectories_float = mode_logits.float(), trajectories.float()
    target_xy_float, valid_actor = target_xy.float(), future_mask.any(-1)
    if intention_points is None:
        distance = torch.norm(trajectories_float[..., :2] - target_xy_float.unsqueeze(-3), dim=-1)
        selected_mode = (distance * future_mask.unsqueeze(-2).to(distance.dtype)).sum(-1).div(
            future_mask.sum(-1).clamp_min(1).unsqueeze(-1)).argmin(-1)
    else:
        if intention_points.shape != tuple(mode_logits.shape) + (2,):
            raise ValueError("intention_points must have shape [actors,modes,2]")
        indices = torch.arange(future_mask.shape[-1], device=future_mask.device).expand_as(future_mask)
        final = torch.where(future_mask, indices, torch.full_like(indices, -1)).max(-1).values.clamp_min(0)
        selected_mode = torch.norm(intention_points.float() - target_xy_float[torch.arange(target_xy.shape[0], device=target_xy.device), final, None], dim=-1).argmin(-1)
    if valid_actor.any():
        regression, _ = nll_loss_gmm_direct(mode_logits_float[valid_actor], trajectories_float[valid_actor], target_xy_float[valid_actor], future_mask[valid_actor], selected_mode[valid_actor], None, False)
        gmm, mode = regression.sum(), F.cross_entropy(mode_logits_float[valid_actor], selected_mode[valid_actor], reduction="sum")
    else:
        gmm, mode = trajectories_float.sum() * 0.0, mode_logits_float.sum() * 0.0
    row = torch.arange(target_xy.shape[0], device=target_xy.device)
    selected_xy = trajectories_float[row, selected_mode, :, :2]
    point_reg = F.smooth_l1_loss(
        selected_xy, target_xy_float, reduction="none").sum(dim=-1)
    mean_reg = (point_reg * future_mask.to(point_reg.dtype)).sum()
    mean_reg_count = future_mask.sum().to(trajectories_float.dtype)
    count = valid_actor.sum().to(trajectories_float.dtype)
    return {"gmm": gmm, "mode": mode, "gmm_count": count, "mode_count": count,
            "mean_reg": mean_reg, "mean_reg_count": mean_reg_count,
            "selected_mode": selected_mode}
