"""Explicit, opt-in aligned-protocol feature/ablation contracts."""
import torch

EVIDENCE_POLICIES = ("source_temporal", "source_current", "receiver_temporal",
                     "geometry_temporal", "fused_temporal", "history")
FEATURE_ENCODERS = ("legacy_additive_v1", "factorized_additive_v1",
                    "temporal_qk_v1", "factorized_qk_v1", "motion_correlation_v1")
MOTION_EVIDENCE_CONTRACT = "same_source_cost_full_width_convgru_actor_scene_memory_norm_v5"


def apply_evidence_policy(inputs, policy):
    """Change observed input, not labels, tracks, losses or matching."""
    if policy not in EVIDENCE_POLICIES:
        raise ValueError("unknown aligned evidence policy: %s" % policy)
    tokens, positions, mask, age, pose = inputs
    if policy == "source_current":
        mask = mask.clone()
        mask[:, :, :, :-1] = False
    elif policy == "receiver_temporal":
        mask = mask.clone()
        mask[:, :, 1:] = False
    elif policy == "geometry_temporal":
        tokens = torch.zeros_like(tokens)
    elif policy == "history":
        mask = torch.zeros_like(mask)
    return tokens, positions, mask, age, pose
