"""A single zero-gradient attention layer is not a disconnected query branch."""
from types import SimpleNamespace

import torch
from torch import nn

from opencood.tools.dualcmp.train_forecast_ddp import (
    _persistent_query_position_gradients, _validate_source_motion_gradients)


def test_active_branch_gradient():
    layers = [SimpleNamespace(ca_qpos_proj=nn.Linear(2, 2)) for _ in range(3)]
    for index, layer in enumerate(layers):
        for parameter in layer.ca_qpos_proj.parameters():
            parameter.grad = torch.full_like(parameter, float(index))
    layers[2].ca_qpos_proj.requires_grad_(False)
    head = SimpleNamespace(decoder=SimpleNamespace(obj_decoder_layers=layers),
                           evidence_decoder_layers=[])
    gradients = _persistent_query_position_gradients(head, torch.device("cpu"))
    assert gradients.shape == (2,) and gradients[0] == 0 and gradients.sum() > 0
    for parameter in layers[1].ca_qpos_proj.parameters():
        parameter.grad.zero_()
    assert _persistent_query_position_gradients(head, torch.device("cpu")).sum() == 0
    layers[1].ca_qpos_proj.weight.grad[0, 0] = float("nan")
    assert not torch.isfinite(_persistent_query_position_gradients(head, torch.device("cpu")).sum())


def test_deliberate_ablation_zero_gradients_do_not_hide_active_branch_failures():
    import pytest

    _validate_source_motion_gradients(torch.tensor([0., 0.]), "source_current", False)
    _validate_source_motion_gradients(torch.tensor([0., 1.]), "geometry_temporal", False)
    _validate_source_motion_gradients(torch.tensor([1., 1.]), "source_temporal", False)
    with pytest.raises(RuntimeError):
        _validate_source_motion_gradients(torch.tensor([0., 1.]), "source_temporal", False)
    with pytest.raises(RuntimeError):
        _validate_source_motion_gradients(torch.tensor([0., 0.]), "geometry_temporal", False)
    with pytest.raises(RuntimeError):
        _validate_source_motion_gradients(torch.tensor([float("nan"), 0.]), "source_current", True)
