"""Opt-in row-wise normalization for packed MTR forecasting features.

Only the forecasting head's two-dimensional Linear/BatchNorm1d/ReLU MLPs
are adapted. Perception/tracking and vendored MTR sources remain unchanged.
The old contract does not visit or modify any module.
"""
import torch
import torch.nn as nn


FORECAST_NORMALIZATIONS = ("batch_norm_v1", "layer_norm_v1")


def configure_forecast_normalization(module, contract):
    if contract not in FORECAST_NORMALIZATIONS:
        raise ValueError("unknown forecast normalization contract")
    if contract == "batch_norm_v1":
        return
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm1d):
            # MTR only applies these modules to [valid rows, channels].
            # Normalize a row's channels, never different actors/scenes.
            replacement = nn.LayerNorm(child.num_features, eps=child.eps,
                                       elementwise_affine=child.affine)
            if child.affine:
                replacement = replacement.to(device=child.weight.device,
                                             dtype=child.weight.dtype)
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    replacement.bias.copy_(child.bias)
            replacement.train(child.training)
            setattr(module, name, replacement)
        elif isinstance(child, nn.modules.batchnorm._BatchNorm):
            raise ValueError("forecast normalization only supports MTR BatchNorm1d")
        else:
            configure_forecast_normalization(child, contract)
