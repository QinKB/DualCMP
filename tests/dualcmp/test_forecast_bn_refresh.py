import unittest

import torch
from torch import nn

from opencood.tools.dualcmp.forecast_bn_refresh import refresh_forecast_bn


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.detector = nn.BatchNorm1d(2)
        self.forecast_head = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(.9), nn.BatchNorm1d(2))
        self.forward_modes = []

    def forward(self, runtime, run_forecast):
        self.forward_modes.append((self.detector.training, self.forecast_head[1].training))
        if runtime is None:
            return
        return self.forecast_head(runtime)


class RefreshTest(unittest.TestCase):
    def test_observed_moments_skip_empty_restore_and_preserve_weights(self):
        model = ToyModel().train()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        first = torch.tensor([[1., 2.], [3., 6.]])
        second = first + 4
        report = refresh_forecast_bn(model, [None, first, first[:1], second])
        torch.testing.assert_close(model.forecast_head[0].running_mean, (first.mean(0) + second.mean(0)) / 2)
        torch.testing.assert_close(model.forecast_head[0].running_var, first.var(0))
        self.assertEqual(model.forecast_head[0].num_batches_tracked, 2)
        self.assertEqual(report["single_or_empty_row_calls_skipped"], 2)
        self.assertTrue(model.forecast_head[1].training and model.detector.training)
        self.assertEqual(model.forward_modes, [(False, False)] * 4)
        for key, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter, before[key]))
        for key, buffer in model.detector.named_buffers():
            self.assertTrue(torch.equal(buffer, before["detector." + key]))

    def test_no_observations_fails_without_mutating_buffers(self):
        model = ToyModel().eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with self.assertRaises(RuntimeError):
            refresh_forecast_bn(model, [None, torch.ones(1, 2)])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        self.assertFalse(model.forecast_head[0].training)


if __name__ == "__main__":
    unittest.main()
