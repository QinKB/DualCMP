import unittest

import torch
from opencood.models.dualcmp import mtr_attention


class IndexedAttentionTest(unittest.TestCase):
    def test_sparse_math_gradients_masks_and_scene_offsets(self):
        torch.manual_seed(4)
        query = torch.randn(3, 2, 4, dtype=torch.float64, requires_grad=True)
        key = torch.randn(5, 2, 4, dtype=torch.float64, requires_grad=True)
        value = torch.randn(5, 2, 3, dtype=torch.float64, requires_grad=True)
        counts = torch.tensor([2, 3], dtype=torch.int32)
        owners = torch.tensor([0, 1, 1], dtype=torch.int32)
        pairs = torch.tensor([[0, 1, -1], [1, 2, -1], [0, -1, -1]], dtype=torch.int32)
        query_counts = torch.tensor([1, 2], dtype=torch.int32)
        weights = mtr_attention.attention_weight_computation(query_counts, counts, owners, pairs, query, key)
        expected = torch.stack((torch.stack(((query[0] * key[0]).sum(-1), (query[0] * key[1]).sum(-1), query[0].sum(-1) * 0)),
                                torch.stack(((query[1] * key[3]).sum(-1), (query[1] * key[4]).sum(-1), query[1].sum(-1) * 0)),
                                torch.stack(((query[2] * key[2]).sum(-1), query[2].sum(-1) * 0, query[2].sum(-1) * 0))))
        torch.testing.assert_close(weights, expected)
        probabilities = weights.masked_fill((pairs == -1)[..., None], float('-inf')).softmax(1)
        reference_probabilities = expected.masked_fill((pairs == -1)[..., None], float('-inf')).softmax(1)
        output = mtr_attention.attention_value_computation(query_counts, counts, owners, pairs, probabilities, value)
        reference = torch.stack((reference_probabilities[0, 0, :, None] * value[0] + reference_probabilities[0, 1, :, None] * value[1],
                                 reference_probabilities[1, 0, :, None] * value[3] + reference_probabilities[1, 1, :, None] * value[4], value[2]))
        torch.testing.assert_close(output, reference)
        actual_grad = torch.autograd.grad(output.square().sum(), (query, key, value), retain_graph=True)
        reference_grad = torch.autograd.grad(reference.square().sum(), (query, key, value))
        for actual, expected_grad in zip(actual_grad, reference_grad):
            torch.testing.assert_close(actual, expected_grad)

    def test_native_default_is_not_changed(self):
        original = torch.are_deterministic_algorithms_enabled()
        try:
            calls = []
            def reference(x):
                calls.append('reference')
                return x * 2
            def native(x):
                calls.append('native')
                return x * 2 + 1e-6
            function = mtr_attention.deterministic_or_native(reference, native)
            x = torch.tensor(1., requires_grad=True)
            torch.use_deterministic_algorithms(False)
            output = function(x)
            self.assertEqual(calls, ['native'])
            self.assertTrue(torch.equal(output, native(x)))
            calls.clear()
            torch.use_deterministic_algorithms(True)
            output = function(x)
            self.assertEqual(calls, ['reference', 'native'])
            self.assertTrue(torch.equal(output, native(x)))
            self.assertEqual(torch.autograd.grad(output, x)[0], 2)
        finally:
            torch.use_deterministic_algorithms(original)


if __name__ == '__main__':
    unittest.main()
