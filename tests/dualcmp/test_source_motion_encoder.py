"""Focused proofs for correspondence, physical coordinates and source identity."""
import torch

from opencood.models.dualcmp.source_motion_encoder import SourceMotionEncoder


def test_spatial_correspondence_preserves_displacement_and_excludes_repeated_capture():
    features = torch.zeros(1, 3, 4, 4)
    features[0, 0] = torch.eye(4)
    features[0, 1] = torch.eye(4).roll(1, 0)
    features[0, 2] = features[0, 1]
    positions = torch.zeros(1, 3, 4, 3)
    positions[0, :, :, 0] = torch.arange(4)[None]
    positions[0, 1:, :, 0] += 2
    mask = torch.ones(1, 3, 4, dtype=torch.bool)
    ages = torch.tensor([[.1, 0., 0.]])
    corr, offsets, pairs, valid = SourceMotionEncoder.correspondence(features, positions, mask, ages, .1)
    assert corr[0, 1].argmax(-1).tolist() == [3, 0, 1, 2]
    assert offsets[0, 1, 1, 0, 0] == 3  # following ROI has NOT erased physical motion
    assert pairs[0, 1].all() and not pairs[0, 0].any() and not pairs[0, 2].any()
    assert not valid[0, 2].any()
    _, _, gap_pairs, _ = SourceMotionEncoder.correspondence(features, positions, mask, ages * 3, .1)
    assert not gap_pairs.any()


def test_motion_encoder_masks_unused_slots_and_is_source_permutation_equivariant():
    torch.manual_seed(11)
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3).eval()
    tokens = torch.randn(1, 2, 2, 4, 9, 32, requires_grad=True)
    xyz = torch.randn(1, 2, 2, 4, 9, 3)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    mask[:, 1] = False
    ages = torch.tensor([.3, .2, .1, 0.])[None, None, None].expand(1, 2, 2, -1)
    pose = torch.zeros(1, 2, 2, 4, 3)
    before = model(tokens, xyz, mask, ages, pose)
    after = model(tokens.flip(2), xyz.flip(2), mask.flip(2), ages.flip(2), pose.flip(2))
    torch.testing.assert_close(before["tokens"].flip(2), after["tokens"])
    assert before["token_mask"].any(-1).any(-1).any(-1).tolist() == [[True, False]]
    assert before["tokens"][:, 1].count_nonzero() == 0
    assert torch.isfinite(before["tokens"]).all()
    before["tokens"].square().sum().backward()
    assert tokens.grad[:, 0, :, :-1].abs().sum() > 0
    assert tokens.grad[:, 1].count_nonzero() == 0
    assert model.correlation_encoder[0].weight.grad.abs().sum() > 0
    assert model.temporal_encoder[0].convq.weight.grad[:, :32].abs().sum() > 0


def test_no_valid_evidence_is_zero_not_nan():
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3)
    tokens = torch.zeros(1, 1, 2, 2, 9, 32)
    output = model(tokens, torch.zeros(1, 1, 2, 2, 9, 3),
                   torch.zeros(tokens.shape[:-1], dtype=torch.bool),
                   torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 3))
    assert output["tokens"].count_nonzero() == 0
    assert not output["token_mask"].any()


def test_zero_score_candidate_validity_is_not_collapsed():
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3).eval()
    tokens = torch.zeros(1, 1, 1, 2, 9, 32)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    mask[..., 0, 4] = False
    captured = []
    hook = model.correlation_encoder[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach()))
    try:
        out = model(tokens, torch.zeros(1, 1, 1, 2, 9, 3), mask,
                    torch.tensor([[[[.1, 0.]]]]), torch.zeros(1, 1, 2, 3))
    finally:
        hook.remove()
    candidates = captured[0].permute(0, 2, 3, 1).reshape(2, 9, -1)
    assert candidates.shape[-1] == 4 * 9 + 4
    assert not candidates[..., :9].any()  # all scores zero, both valid and invalid
    flags = candidates[1, :, 3 * 9:4 * 9]
    assert torch.equal(flags.bool(), mask[0, 0, 0, 0].expand(9, -1))
    assert torch.equal(out["token_mask"], mask)


def test_no_age_ablation_keeps_physical_time_and_temporal_observations():
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3).eval()
    tokens = torch.randn(1, 1, 2, 3, 9, 32, requires_grad=True)
    xyz = torch.randn(1, 1, 2, 3, 9, 3)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    age = torch.tensor([.2, .1, 0.])[None, None, None].expand(1, 1, 2, -1)
    pose = torch.zeros(1, 2, 3, 3)
    normal = model(tokens, xyz, mask, age, pose)
    ablated = model(tokens, xyz, mask, age, pose, encode_age=False)
    assert torch.equal(normal["token_mask"], ablated["token_mask"])
    assert ablated["token_mask"].all()
    ablated["tokens"].square().sum().backward()
    assert tokens.grad[..., :-1, :, :].abs().sum() > 0


def test_later_observations_cannot_change_earlier_encoded_tokens():
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3).eval()
    tokens = torch.randn(1, 1, 2, 4, 9, 32)
    xyz = torch.randn(1, 1, 2, 4, 9, 3)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    age = torch.tensor([.3, .2, .1, 0.])[None, None, None].expand(1, 1, 2, -1)
    pose = torch.zeros(1, 2, 4, 3)
    before = model(tokens, xyz, mask, age, pose)["tokens"]
    changed = tokens.clone()
    changed[:, :, :, 2:] += torch.randn_like(changed[:, :, :, 2:]) * 10
    after = model(changed, xyz, mask, age, pose)["tokens"]
    torch.testing.assert_close(before[:, :, :, :2], after[:, :, :, :2], rtol=0, atol=0)


def test_last_motion_state_uses_oldest_observation_of_full_history():
    torch.manual_seed(21)
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3).eval()
    tokens = torch.randn(1, 1, 1, 11, 9, 32, requires_grad=True)
    xyz = torch.zeros(1, 1, 1, 11, 9, 3)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    age = torch.arange(10, -1, -1).float()[None, None, None] * .1
    output = model(tokens, xyz, mask, age, torch.zeros(1, 1, 11, 3))
    # Loss uses only final learned state, not the verbatim observation branch.
    output["tokens"][..., -1, :, 32:].square().sum().backward()
    assert torch.isfinite(tokens.grad).all()
    assert tokens.grad[..., 0, :, :].abs().sum() > 0
    assert model.temporal_encoder[0].convq.weight.grad[:, :32].abs().sum() > 0


def test_invalid_capture_cannot_update_recurrent_state():
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3).eval()
    tokens = torch.randn(1, 1, 1, 4, 9, 32)
    xyz = torch.zeros(1, 1, 1, 4, 9, 3)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    mask[..., 1, :] = False
    age = torch.tensor([[[[.3, .2, .1, 0.]]]])
    pose = torch.zeros(1, 1, 4, 3)
    before = model(tokens, xyz, mask, age, pose)["tokens"]
    changed = tokens.clone()
    changed[..., 1, :, :] = float("nan")
    after = model(changed, xyz, mask, age, pose)["tokens"]
    assert torch.equal(before, after)


def test_valid_observation_after_invalid_prefix_is_not_a_duplicate():
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3).eval()
    tokens = torch.randn(1, 1, 2, 4, 9, 32)
    xyz = torch.zeros(1, 1, 2, 4, 9, 3)
    age = torch.tensor([.3, .2, .1, 0.])[None, None, None].expand(1, 1, 2, -1)
    pose = torch.zeros(1, 2, 4, 3)
    for first_observed in (1, 3):
        mask = torch.zeros(tokens.shape[:-1], dtype=torch.bool)
        mask[:, :, :, first_observed:] = True
        output = model(tokens, xyz, mask, age, pose)
        assert torch.equal(output["token_mask"], mask)
        assert torch.isfinite(output["tokens"]).all()


def test_original_observations_are_preserved_without_axis_or_channel_reduction():
    model = SourceMotionEncoder(32, roi_grid_size=3, pooled_grid_size=3).eval()
    tokens = torch.randn(1, 1, 2, 4, 9, 32)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    ages = torch.tensor([.3, .2, .1, 0.])[None, None, None].expand(1, 1, 2, -1)
    out = model(tokens, torch.zeros(1, 1, 2, 4, 9, 3), mask, ages, torch.zeros(1, 2, 4, 3))
    assert out["tokens"].shape == (1, 1, 2, 4, 9, 64)
    assert torch.equal(out["tokens"][..., :32], tokens)
    assert model.feature_projection[0].in_channels == model.feature_projection[0].out_channels == 32
    assert model.spatial_encoder[0].in_channels == model.spatial_encoder[0].out_channels == 32
    try:
        SourceMotionEncoder(32, roi_grid_size=5, pooled_grid_size=3)
    except ValueError:
        pass
    else:
        raise AssertionError("motion observations must not be spatially downsampled")
