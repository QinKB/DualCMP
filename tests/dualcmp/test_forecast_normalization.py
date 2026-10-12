import copy

import torch
import torch.nn as nn

from opencood.models.dualcmp.forecast_normalization import configure_forecast_normalization


def test_old_contract_is_exact_noop():
    module = nn.Sequential(nn.Linear(8, 8, bias=False), nn.BatchNorm1d(8), nn.ReLU())
    original = copy.deepcopy(module).eval()
    child = module[1]
    configure_forecast_normalization(module, "batch_norm_v1")
    assert module[1] is child
    values = torch.randn(4, 8)
    assert torch.equal(module.eval()(values), original(values))
    assert module.state_dict().keys() == original.state_dict().keys()


def test_row_normalization_train_eval_and_batch_independent():
    module = nn.Sequential(nn.Linear(8, 8, bias=False), nn.BatchNorm1d(8), nn.ReLU())
    configure_forecast_normalization(module, "layer_norm_v1")
    assert isinstance(module[1], nn.LayerNorm)
    values = torch.randn(4, 8)
    expected = module.train()(values)
    assert torch.equal(module.eval()(values), expected)
    assert torch.allclose(module(values[:1]), expected[:1], atol=1e-6, rtol=1e-6)
    module(values).sum().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in module.parameters())
