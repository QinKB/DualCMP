"""Masked nearest-mode bivariate GMM loss used by CoLIF."""

import torch
import torch.nn.functional as F


def nearest_mode_gmm_loss(
    mode_logits,
    trajectories,
    target_xy,
    future_mask,
    intention_points=None,
):
    """Return regression/mode sums, counts and selected mode indices."""

    # Long-horizon squared errors overflow float16 easily during early
    # training.  GMM parameterization and reduction are always float32 even
    # when the surrounding model runs under autocast.
    mode_logits_float = mode_logits.float()
    trajectories_float = trajectories.float()
    target_xy_float = target_xy.float()
    valid_actor = future_mask.any(-1)
    if intention_points is None:
        distance = torch.norm(
            trajectories_float[..., :2] - target_xy_float.unsqueeze(-3), dim=-1
        )
        masked_distance = distance * future_mask.unsqueeze(-2).to(distance.dtype)
        mode_cost = (
            masked_distance.sum(-1)
            / future_mask.sum(-1).clamp_min(1).unsqueeze(-1)
        )
        selected_mode = mode_cost.argmin(-1)
    else:
        intention_points = intention_points.float()
        if intention_points.shape != (*mode_logits.shape, 2):
            raise ValueError("intention_points must have shape [actors,modes,2]")
        time_indices = torch.arange(
            future_mask.shape[-1], device=future_mask.device
        ).expand_as(future_mask)
        final_valid = torch.where(
            future_mask,
            time_indices,
            torch.full_like(time_indices, -1),
        ).max(dim=-1).values.clamp_min(0)
        actor_indices = torch.arange(
            target_xy.shape[0], device=target_xy.device
        )
        goals = target_xy_float[actor_indices, final_valid]
        selected_mode = torch.norm(
            intention_points - goals[:, None], dim=-1
        ).argmin(dim=-1)
    gather_index = selected_mode[..., None, None, None].expand(
        *selected_mode.shape, 1,
        trajectories_float.shape[-2], trajectories_float.shape[-1]
    )
    selected = trajectories_float.gather(-3, gather_index).squeeze(-3)
    mu = selected[..., :2]
    # Preserve CMP/MTR's published V2V4Real GMM parameter bounds.  Tighter
    # variances here make early tracker mismatches produce explosive gradients
    # and are not semantically equivalent to nll_loss_gmm_direct.
    log_std_x = selected[..., 2].clamp(-1.609, 5.0)
    log_std_y = selected[..., 3].clamp(-1.609, 5.0)
    rho = selected[..., 4].clamp(-0.5, 0.5)
    dx = target_xy_float[..., 0] - mu[..., 0]
    dy = target_xy_float[..., 1] - mu[..., 1]
    std_x = torch.exp(log_std_x)
    std_y = torch.exp(log_std_y)
    one_minus_rho2 = (1.0 - rho.square()).clamp_min(1e-4)
    z = (
        (dx / std_x).square() + (dy / std_y).square()
        - 2.0 * rho * dx * dy / (std_x * std_y)
    )
    nll = (
        log_std_x + log_std_y + 0.5 * torch.log(one_minus_rho2)
        + 0.5 * z / one_minus_rho2
    )
    step_mask = future_mask & valid_actor.unsqueeze(-1)
    regression_sum = (nll * step_mask.to(nll.dtype)).sum()
    mode_sum = F.cross_entropy(
        mode_logits_float[valid_actor], selected_mode[valid_actor], reduction="sum"
    ) if valid_actor.any() else mode_logits_float.sum() * 0.0
    # CMP/MTR first sums the per-timestamp GMM NLL for every actor and then
    # averages those actor losses.  Returning a timestamp count here used to
    # divide the regression term by roughly 50 once ``ColifLoss`` normalized
    # distributed raw sums, while the classification term was still divided
    # only by actor count.  That silently changed the published MTR objective
    # from ``sum_time(NLL) + CE`` to ``mean_time(NLL) + CE`` and made endpoint
    # geometry far too weak relative to mode classification.
    actor_count = valid_actor.sum().to(trajectories_float.dtype)
    return {
        "gmm": regression_sum,
        "mode": mode_sum,
        "gmm_count": actor_count,
        "mode_count": actor_count,
        "selected_mode": selected_mode,
    }
