"""Contracts for compact causal per-source temporal evidence."""

import torch

from opencood.models.dualcmp.source_temporal_encoder import (
    SourceTemporalEncoder,
)
from opencood.models.dualcmp.intention_query_forecast import (
    IntentionQueryForecastHead,
)


def _encoder():
    encoder = SourceTemporalEncoder(
        feature_dim=8, roi_grid_size=5, pooled_grid_size=2,
        heads=2, layers=1, dropout=0.0, frame_period_s=0.1,
    )
    encoder.eval()
    return encoder


def test_pool_frame_is_mask_aware_and_compact():
    encoder = _encoder()
    tokens = torch.arange(1 * 2 * 2 * 25 * 8, dtype=torch.float32).reshape(
        1, 2, 2, 25, 8)
    positions = torch.zeros(1, 2, 2, 25, 3)
    mask = torch.ones(1, 2, 2, 25, dtype=torch.bool)
    mask[:, 1, 1] = False
    pooled, pooled_positions, pooled_mask = encoder.pool_frame(
        tokens, positions, mask)
    assert pooled.shape == (1, 2, 2, 4, 8)
    assert pooled_positions.shape == (1, 2, 2, 4, 3)
    assert pooled_mask.shape == (1, 2, 2, 4)
    assert not pooled_mask[0, 1, 1].any()
    assert not pooled[0, 1, 1].any()


def test_runtime_identity_mask_rejects_reused_slots():
    runtime_ids = torch.tensor([[[3, 3, 7], [5, -1, -1]]])
    final_ids = torch.tensor([[7, 8]])
    assert SourceTemporalEncoder.same_runtime_mask(
        runtime_ids, final_ids).tolist() == [
            [[False, False, True], [False, False, False]]]


def test_temporal_encoder_uses_time_metadata_and_backpropagates():
    torch.manual_seed(9)
    encoder = _encoder()
    tokens = torch.randn(1, 2, 2, 3, 4, 8, requires_grad=True)
    positions = torch.randn(1, 2, 2, 3, 4, 3)
    mask = torch.ones(1, 2, 2, 3, 4, dtype=torch.bool)
    age = torch.tensor([0.2, 0.1, 0.0]).view(1, 1, 1, 3).expand(1, 2, 2, 3)
    pose = torch.zeros(1, 2, 3, 3)
    pose[:, 1, :, 0] = 5.0
    direct = encoder(tokens, positions, mask, age, pose)["tokens"]
    shuffled = encoder(
        tokens.detach().flip(3), positions, mask, age, pose)["tokens"]
    assert direct.shape == tokens.shape
    assert not torch.allclose(direct, shuffled)
    direct.square().mean().backward()
    assert tokens.grad is not None and float(tokens.grad.norm()) > 0
    parameter_grad = sum(
        float(parameter.grad.norm())
        for parameter in encoder.parameters()
        if parameter.grad is not None)
    assert parameter_grad > 0


def test_parallel_source_fusion_preserves_mtr_and_backpropagates():
    baseline = torch.randn(2, 3, 4)
    layer_input = torch.randn_like(baseline)
    source = torch.randn_like(baseline, requires_grad=True)
    assert torch.equal(
        IntentionQueryForecastHead._fuse_parallel_source(
            baseline, source, layer_input, torch.tensor(0.0)),
        baseline,
    )
    gate = torch.tensor(0.1, requires_grad=True)
    fused = IntentionQueryForecastHead._fuse_parallel_source(
        baseline, source, layer_input, gate)
    fused.sum().backward()
    assert source.grad.abs().sum() > 0
    assert gate.grad.abs() > 0
