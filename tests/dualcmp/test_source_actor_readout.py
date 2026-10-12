"""Full-memory ownership and neighbor readout before MTR scene attention."""
import torch

from opencood.models.dualcmp.source_motion_encoder import SourceActorReadout


def _inputs():
    torch.manual_seed(4)
    reader = SourceActorReadout(32, 64, dropout=0).eval()
    features = torch.randn(2, 3, 32, requires_grad=True)
    entries = ((0, 0), (0, 1))
    slots = torch.tensor([[0, 1, 2], [0, 1, -1]])
    tokens = torch.randn(1, 3, 2, 9, 64, requires_grad=True)
    positions = torch.randn(1, 3, 2, 9, 3)
    mask = torch.ones(tokens.shape[:-1], dtype=torch.bool)
    mask[:, 2] = False
    return reader, features, entries, slots, tokens, positions, mask


def test_neighbor_reads_own_full_memory_without_changing_other_actors():
    reader, features, entries, slots, tokens, xyz, mask = _inputs()
    before_tokens = tokens.detach().clone()
    before = reader(features, entries, slots, tokens, xyz, mask)
    assert reader.last_counts == {"contexts": 4, "neighbors": 2, "memory_rows": 2}
    assert torch.equal(tokens, before_tokens)
    assert torch.equal(before[:, 2], features[:, 2])
    changed = tokens.detach().clone()
    changed[:, 1] += torch.randn_like(changed[:, 1]) * 3
    after = reader(features, entries, slots, changed, xyz, mask)
    assert torch.equal(before[:, 0], after[:, 0])
    assert not torch.equal(before[:, 1], after[:, 1])
    # Center0's neighbor1 readout reaches all valid memory, not center0's bank.
    before[0, 1].square().sum().backward()
    assert tokens.grad[:, 1].abs().sum() > 0
    assert tokens.grad[:, 0].count_nonzero() == 0
    assert tokens.grad[:, 2].count_nonzero() == 0
    assert reader.attention.in_proj_weight.grad.abs().sum() > 0


def test_memory_permutation_and_invalid_values_do_not_change_readout():
    reader, features, entries, slots, tokens, xyz, mask = _inputs()
    before = reader(features, entries, slots, tokens, xyz, mask)
    after = reader(features, entries, slots, tokens.flip(2), xyz.flip(2), mask.flip(2))
    torch.testing.assert_close(before, after)
    dirty_tokens, dirty_xyz = tokens.detach().clone(), xyz.clone()
    dirty_tokens[~mask] = float("nan")
    dirty_xyz[~mask] = float("nan")
    assert torch.equal(before, reader(features, entries, slots, dirty_tokens, dirty_xyz, mask))
    empty = reader(features, entries, slots, dirty_tokens, dirty_xyz, torch.zeros_like(mask))
    assert empty is features


def test_attention_message_is_normalized_without_modifying_full_memory():
    reader, features, entries, slots, tokens, xyz, mask = _inputs()
    original = tokens.detach().clone()
    with torch.no_grad():
        # Output magnitude must not scale with an oversized learned projection.
        reader.output_projection.weight.mul_(100)
        reader.output_projection.bias.mul_(100)
    output = reader(features, entries, slots, tokens, xyz, mask)
    message = (output - features)[:, :2]
    torch.testing.assert_close(message.mean(-1), torch.zeros_like(message[..., 0]), atol=1e-6, rtol=0)
    torch.testing.assert_close(message.var(-1, unbiased=False), torch.ones_like(message[..., 0]), atol=1e-4, rtol=0)
    assert torch.equal(tokens, original) and torch.equal(output[:, 2], features[:, 2])
    output[0, 1].square().sum().backward()
    assert reader.output_normalization.weight.grad.abs().sum() > 0
    assert reader.output_normalization.bias.grad.abs().sum() > 0
    assert reader.attention.in_proj_weight.grad.abs().sum() > 0
    assert tokens.grad[:, 1].abs().sum() > 0
