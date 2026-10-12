"""Same MTR indexed dot-product/value sums, using deterministic PyTorch autograd."""
import torch


def _selected_features(key_batch_cnt, index_pair_batch, index_pair, features):
    offsets = torch.cat((key_batch_cnt.new_zeros(1), key_batch_cnt.cumsum(0)[:-1]))
    valid = index_pair >= 0
    indices = index_pair.long() + offsets[index_pair_batch.long(), None]
    # MTR -1 neighbors contribute zero; never index another scene for padding.
    indices = torch.where(valid, indices, torch.zeros_like(indices))
    selected = features.index_select(0, indices.reshape(-1)).reshape(
        *indices.shape, *features.shape[1:])
    return selected, valid


def attention_weight_computation(query_batch_cnt, key_batch_cnt, index_pair_batch,
                                 index_pair, query_features, key_features):
    selected, valid = _selected_features(key_batch_cnt, index_pair_batch, index_pair, key_features)
    values = (query_features[:, None] * selected).sum(-1)
    return torch.where(valid[..., None], values, torch.zeros_like(values))


def attention_value_computation(query_batch_cnt, key_batch_cnt, index_pair_batch,
                                index_pair, attn_weight, value_features):
    selected, valid = _selected_features(key_batch_cnt, index_pair_batch, index_pair, value_features)
    values = attn_weight[..., None] * selected
    return torch.where(valid[..., None, None], values, torch.zeros_like(values)).sum(1)


def deterministic_or_native(reference, native):
    def compute(*args):
        if torch.are_deterministic_algorithms_enabled():
            reference_output = reference(*args)
            # Preserve the pinned bitwise forward; differentiate the same dot/
            # weighted-sum formulas without atomicAdd in their backward.
            with torch.no_grad():
                native_output = native(*args)
            return native_output + (reference_output - reference_output.detach())
        return native(*args)
    return compute
