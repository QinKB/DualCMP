"""Focused CPU preflight only; does not establish trajectory quality."""
import subprocess

import torch
import pytest

from opencood.models.dualcmp.source_temporal_encoder import SourceTemporalEncoder
from opencood.models.dualcmp.source_evidence_policy import apply_evidence_policy


def inputs():
    torch.manual_seed(1)
    tokens = torch.randn(1, 2, 2, 3, 25, 16)
    positions = torch.randn(1, 2, 2, 3, 25, 3)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    mask[:, 1] = False
    age = torch.ones(1, 2, 2, 3) * .2
    pose = torch.zeros(1, 2, 2, 3, 3)
    return tokens, positions, mask, age, pose


def test_legacy_aligned_math_is_unchanged():
    old_source = subprocess.check_output(["git", "show", "0f988f7:opencood/models/dualcmp/source_temporal_encoder.py"]).decode()
    namespace = {"__name__": "old_source_encoder", "__package__": "opencood.models.dualcmp"}
    exec(compile(old_source, "historical_source_encoder", "exec"), namespace)
    old = namespace["SourceTemporalEncoder"](16, pooled_grid_size=5, heads=4, dropout=0., standard_position_encoding=True).eval()
    current = SourceTemporalEncoder(16, pooled_grid_size=5, heads=4, dropout=0., standard_position_encoding=True).eval()
    current.load_state_dict(old.state_dict(), strict=True)
    data = inputs()
    assert torch.equal(old(*data)["tokens"], current(*data)["tokens"])


def test_factorized_attention_mixes_space_not_sources_and_has_gradients():
    module = SourceTemporalEncoder(16, pooled_grid_size=5, heads=4, dropout=0.,
        standard_position_encoding=True, feature_encoder="factorized_qk_v1").eval()
    data = inputs()
    first = module(*data)["tokens"]
    assert torch.isfinite(first).all() and not first[:, 1].count_nonzero()
    changed = list(data)
    changed[0] = data[0].clone()
    changed[0][0, 0, 0, 0, 0] += torch.arange(16)
    second = module(*changed)["tokens"]
    assert not torch.equal(first[0, 0, 0, 0, 1], second[0, 0, 0, 0, 1])
    assert torch.equal(first[:, :, 1], second[:, :, 1])
    first[0, 0, 0, 0, 0, 0].backward()
    for name in ("spatial_encoder", "temporal_encoder"):
        gradients = [p.grad for p in getattr(module, name).parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.norm()) for g in gradients) > 0


def test_policies_only_change_observed_features_and_masks():
    data = inputs()
    for name in ("source_current", "receiver_temporal", "geometry_temporal", "history"):
        changed = apply_evidence_policy(data, name)
        assert changed[1] is data[1] and changed[3] is data[3] and changed[4] is data[4]
    assert not apply_evidence_policy(data, "source_current")[2][:, :, :, :-1].any()
    assert not apply_evidence_policy(data, "receiver_temporal")[2][:, :, 1:].any()
    assert not apply_evidence_policy(data, "geometry_temporal")[0].any()


@pytest.mark.parametrize("key", ["frame_period_s", "pose_scale_m"])
def test_source_encoder_rejects_nonfinite_scales(key):
    with pytest.raises(ValueError, match="configuration"):
        SourceTemporalEncoder(16, heads=4, **{key: float("nan")})
