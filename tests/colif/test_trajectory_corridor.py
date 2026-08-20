"""Contracts for trajectory-aligned per-source BEV refinement."""

import math

import torch

from opencood.models.colif.trajectory_corridor import (
    SourceCorridorRefiner,
    TrajectoryCorridorSampler,
)


def _sampler():
    return TrajectoryCorridorSampler(
        lidar_range=[-4.0, -4.0, -3.0, 4.0, 4.0, 3.0],
        lateral_offset_m=1.0,
        longitudinal_offset_m=1.0,
    )


def test_corridor_sampler_preserves_mode_time_source_and_offset_axes():
    source = torch.randn(1, 2, 4, 8, 8)
    boxes = torch.zeros(1, 1, 7)
    boxes[..., 6] = math.pi / 2.0
    proposal = torch.tensor(
        [[[[[0.5, 0.0], [1.0, 0.0], [1.5, 0.5]]]]]
    ).reshape(1, 1, 1, 3, 2)
    cav_mask = torch.tensor([[True, False]])

    result = _sampler()(source, boxes, proposal, cav_mask)

    assert result.samples.shape == (1, 1, 1, 3, 2, 4)
    assert result.valid.shape == (1, 1, 1, 3, 2)
    assert result.grid.shape == (1, 1, 1, 3, 5, 2)
    assert result.valid[..., 0].all()
    assert not result.valid[..., 1].any()
    assert torch.count_nonzero(result.samples[..., 1, :]) == 0
    assert not torch.equal(result.grid[..., 0, :], result.grid[..., 1, :])


def test_corridor_sampler_uses_current_yaw_when_first_motion_is_zero():
    source = torch.randn(1, 1, 2, 8, 8)
    boxes = torch.zeros(1, 1, 7)
    boxes[..., 6] = math.pi / 2.0
    proposal = torch.zeros(1, 1, 1, 2, 2)

    result = _sampler()(source, boxes, proposal, torch.ones(1, 1, dtype=torch.bool))

    center = result.grid[0, 0, 0, 0, 0]
    forward = result.grid[0, 0, 0, 0, 3]
    # A +90 degree heading turns the longitudinal offset toward +y.
    assert torch.isclose(center[0], forward[0], atol=1e-6)
    assert forward[1] > center[1]


def test_corridor_sampler_hard_masks_out_of_range_states():
    source = torch.randn(1, 1, 2, 8, 8)
    boxes = torch.zeros(1, 1, 7)
    proposal = torch.full((1, 1, 1, 2, 2), 20.0)

    result = _sampler()(source, boxes, proposal, torch.ones(1, 1, dtype=torch.bool))

    assert not result.valid.any()
    assert torch.count_nonzero(result.samples) == 0


def _refiner():
    return SourceCorridorRefiner(
        query_dim=8,
        source_bev_dim=4,
        modes=2,
        future_frames=3,
        args={
            "frame_period_s": 0.1,
            "source_velocity_limit_mps": 3.0,
            "source_residual_limit_m": 4.0,
            "source_gate_prior": 0.1,
            "source_heads": 2,
            "dropout": 0.0,
        },
    )


def _refiner_inputs():
    torch.manual_seed(7)
    samples = torch.randn(1, 2, 2, 3, 2, 4)
    valid = torch.ones(1, 2, 2, 3, 2, dtype=torch.bool)
    base_trajs = torch.zeros(1, 2, 2, 3, 5)
    base_trajs[..., 2:4] = math.log(5.0)
    base_logits = torch.zeros(1, 2, 2)
    geometry = torch.randn(1, 2, 2, 8)
    age = torch.randn(1, 2, 2, 8)
    availability = torch.zeros(1, 2, 2)
    actor_valid = torch.ones(1, 2, dtype=torch.bool)
    return samples, valid, base_trajs, base_logits, geometry, age, availability, actor_valid


def test_zero_source_context_cannot_use_base_as_residual_bypass():
    refiner = _refiner().eval()
    with torch.no_grad():
        refiner.velocity_head.weight.fill_(0.25)
        refiner.log_std_head.weight.fill_(0.25)
        refiner.raw_rho_head.weight.fill_(0.25)
        refiner.mode_score_head.weight.fill_(0.25)
    inputs = list(_refiner_inputs())
    inputs[0] = torch.zeros_like(inputs[0])

    output = refiner(*inputs)

    torch.testing.assert_close(output.mean_delta, torch.zeros_like(output.mean_delta), atol=0, rtol=0)
    torch.testing.assert_close(output.velocity_delta, torch.zeros_like(output.velocity_delta), atol=0, rtol=0)
    torch.testing.assert_close(output.final_trajs, inputs[2], atol=0, rtol=0)
    torch.testing.assert_close(output.final_logits, inputs[3], atol=0, rtol=0)


def test_all_invalid_sources_select_exact_null():
    refiner = _refiner().eval()
    inputs = list(_refiner_inputs())
    inputs[1] = torch.zeros_like(inputs[1])

    output = refiner(*inputs)

    assert torch.count_nonzero(output.source_gate) == 0
    assert torch.count_nonzero(output.selection_weights) == 0
    expected = torch.zeros_like(output.router_weights)
    expected[..., 0] = 1.0
    torch.testing.assert_close(output.router_weights, expected, atol=0, rtol=0)
    torch.testing.assert_close(output.final_trajs, inputs[2], atol=0, rtol=0)


def test_joint_source_permutation_preserves_final_and_permutes_selection():
    refiner = _refiner().eval()
    inputs = _refiner_inputs()
    original = refiner(*inputs)
    permutation = torch.tensor([1, 0])
    permuted = refiner(
        inputs[0][..., permutation, :],
        inputs[1][..., permutation],
        inputs[2],
        inputs[3],
        inputs[4][..., permutation, :],
        inputs[5][..., permutation, :],
        inputs[6][..., permutation],
        inputs[7],
    )

    torch.testing.assert_close(original.final_trajs, permuted.final_trajs)
    torch.testing.assert_close(original.final_logits, permuted.final_logits)
    torch.testing.assert_close(
        original.selection_weights,
        permuted.selection_weights[..., permutation],
    )


def test_velocity_integration_is_norm_bounded_and_cumulative():
    raw = torch.tensor([[[[[30.0, 40.0], [0.0, 30.0], [0.0, 0.0]]]]])
    bounded = SourceCorridorRefiner.bound_vector_norm(raw, 3.0)
    integrated = SourceCorridorRefiner.integrate_velocity(
        bounded, frame_period_s=0.1, position_limit_m=4.0
    )

    assert torch.linalg.vector_norm(bounded, dim=-1).max() <= 3.0 + 1e-6
    assert torch.linalg.vector_norm(integrated[..., 0, :], dim=-1).max() <= 0.3 + 1e-6
    expected_second = bounded[..., :2, :].sum(dim=-2) * 0.1
    torch.testing.assert_close(integrated[..., 1, :], expected_second, atol=1e-5, rtol=1e-5)
