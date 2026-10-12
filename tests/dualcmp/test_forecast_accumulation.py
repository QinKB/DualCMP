"""Whole-window supervision normalization, not microbatch-mean averaging."""
from contextlib import nullcontext

import torch
from torch import nn

from opencood.tools.dualcmp.train_forecast_ddp import (
    _forecast_batch_windows, _forecast_window_loss, _ensure_forecast_ddp_backward)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.))

    def no_sync(self):
        return nullcontext()

    def forward(self, frames, run_forecast=True):
        output = {"value": self.weight * frames, "forecast": {}}
        self.last_output = output
        return [output], None


class ToyCriterion:
    def __call__(self, outputs, mask):
        error = (outputs[-1]["value"].square() * mask).sum()
        self.last_loss = {
            prefix + field: (error * scale if prefix == "S_" else mask.sum().float())
            for prefix in ("S_", "C_")
            for field, scale in zip(("gmm", "mode", "mean_reg", "dense_gmm"), (1., 2., 3., 4.))}
        return self.last_loss


def batch(values, mask):
    return {"runtime_frames": torch.tensor(values), "target_frames": torch.tensor(mask),
            "scene_receiver_keys": list(range(len(values)))}


def test_window_counts_and_gradient_match_concatenation_not_microbatch_means():
    model, criterion = ToyModel(), ToyCriterion()
    window = [batch([1., 2.], [True, False]), batch([3., 4.], [True, True])]
    loss, output, samples = _forecast_window_loss(model, criterion, window, torch.device("cpu"))
    assert samples == 4 and all(loss["C_" + field] == 3 for field in ("gmm", "mode", "mean_reg", "dense_gmm"))
    objective = sum(loss["S_" + field] * scale / loss["C_" + field]
                    for field, scale in zip(("gmm", "mode", "mean_reg", "dense_gmm"), (1., 1., 5., 1.)))
    assert _ensure_forecast_ddp_backward(model, objective, loss, output) is objective
    objective.backward()
    reference = ToyModel()
    concatenated = batch([1., 2., 3., 4.], [True, False, True, True])
    expected = (reference.weight * concatenated["runtime_frames"]).square()[concatenated["target_frames"]].mean() * 22.
    expected.backward()
    torch.testing.assert_close(model.weight.grad, reference.weight.grad)


def test_accumulation_one_preserves_loss_and_output_objects():
    model, criterion = ToyModel(), ToyCriterion()
    loss, output, _ = _forecast_window_loss(model, criterion, [batch([1.], [True])], torch.device("cpu"))
    assert loss is criterion.last_loss and output is model.last_output


def test_incomplete_epoch_tail_is_not_a_smaller_global_batch():
    assert list(_forecast_batch_windows(range(5), 2)) == [[0, 1], [2, 3]]


def test_wholly_empty_window_has_zero_backward():
    model, criterion = ToyModel(), ToyCriterion()
    loss, output, _ = _forecast_window_loss(model, criterion,
        [batch([1.], [False]), batch([2.], [False])], torch.device("cpu"))
    objective = _ensure_forecast_ddp_backward(model, sum(loss["S_" + field]
        for field in ("gmm", "mode", "mean_reg", "dense_gmm")), loss, output)
    objective.backward()
    assert model.weight.grad.item() == 0.
