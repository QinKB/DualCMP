"""Strictly bounded future-conditioned query-content recurrence."""

from __future__ import annotations

import torch
import torch.nn as nn


class FutureAdapter(nn.Module):
    """Inject a compact future cue without moving spatial references.

    For every actor, the update obeys
    ``||q_prior - LN(q)||_2 <= gate_max * clip_norm``.  Spatial propagation is
    intentionally implemented elsewhere from current state and velocity.
    """

    def __init__(
        self,
        query_dim: int,
        mode: str = "none",
        gate_max: float = 0.25,
        clip_norm: float = 4.0,
        dropout_probability: float = 0.3,
    ):
        super().__init__()
        if mode not in ("none", "detached", "recurrent_bptt"):
            raise ValueError("Unknown future adapter mode: %s" % mode)
        if not 0.0 < gate_max < 1.0:
            raise ValueError("gate_max must lie strictly between zero and one")
        if clip_norm <= 0.0 or not 0.0 <= dropout_probability < 1.0:
            raise ValueError("Invalid future adapter bound/dropout")
        self.mode = mode
        self.gate_max = float(gate_max)
        self.clip_norm = float(clip_norm)
        self.dropout_probability = float(dropout_probability)
        self.query_norm = nn.LayerNorm(query_dim)
        self.cue_projection = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.cue_norm = nn.LayerNorm(query_dim)
        self.gate = nn.Sequential(
            nn.Linear(query_dim * 2 + 1, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 1),
        )
        # Phase 2 starts from a pretrained tracker.  Begin as an almost exact
        # no-feedback model and let the recurrent tracking loss open the gate
        # only when the future cue is useful; a random half-open gate causes a
        # distribution jump at the phase boundary despite the hard norm cap.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -4.0)

    def forward(
        self,
        query: torch.Tensor,
        future_cue: torch.Tensor,
        uncertainty: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        # A training clip executes its causal burn-in under ``no_grad`` and
        # the supervised suffix with gradients enabled.  If both regions sit
        # inside one outer AMP context, autocast can cache the Linear layers'
        # half-precision weights during burn-in and reuse those detached casts
        # later.  LayerNorm parameters would then train while cue_projection
        # and gate silently receive no gradients.  Keep this small recurrent
        # adapter in float32, as we already do for the MOTR CUDA islands.
        with torch.cuda.amp.autocast(enabled=False):
            return self._forward_float(
                query.float(),
                future_cue.float(),
                uncertainty.float(),
                valid_mask,
            )

    def _forward_float(
        self,
        query: torch.Tensor,
        future_cue: torch.Tensor,
        uncertainty: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        base = self.query_norm(query)
        if self.mode == "none":
            zeros = torch.zeros_like(uncertainty)
            return base, {
                "feedback_gate": zeros,
                "feedback_delta_norm": zeros,
                "feedback_kept": torch.zeros_like(valid_mask),
            }

        cue = future_cue.detach() if self.mode == "detached" else future_cue
        cue_value = self.cue_norm(self.cue_projection(cue))
        delta = cue_value - base
        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        scale = (self.clip_norm / norm.clamp_min(1e-6)).clamp(max=1.0)
        delta = delta * scale
        gate = self.gate(torch.cat([base, cue, uncertainty], dim=-1)).sigmoid()
        gate = gate * self.gate_max

        keep = valid_mask
        if self.training and self.dropout_probability > 0.0:
            keep = keep & (
                torch.rand_like(gate[..., 0]) >= self.dropout_probability
            )
        gate = gate * keep[..., None].to(gate.dtype)
        updated = base + gate * delta
        return updated, {
            "feedback_gate": gate,
            "feedback_delta_norm": torch.linalg.vector_norm(
                updated - base, dim=-1, keepdim=True
            ),
            "feedback_kept": keep,
        }
