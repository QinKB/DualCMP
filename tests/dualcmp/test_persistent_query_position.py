"""A focused query-identity check, not training/metric acceptance."""
import torch
import opencood.models.dualcmp.mtr_vendor  # installs the bundled MTR import root
from mtr.models_v2v4real.utils.transformer.transformer_decoder_layer import TransformerDecoderLayer


def test_persistent_static_position_distinguishes_collapsed_dynamic_queries():
    torch.manual_seed(42)
    layer = TransformerDecoderLayer(d_model=16, nhead=2, dropout=0.)
    inputs = dict(tgt=torch.zeros(4, 1, 16), memory=torch.randn(3, 1, 16),
                  query_pos=torch.randn(4, 1, 16), query_sine_embed=torch.zeros(4, 1, 16),
                  pos=torch.randn(3, 1, 16), is_first=False)
    original = layer(**inputs)
    assert original.std(dim=0, unbiased=False).max() < 1e-6
    assert not layer.keep_query_pos
    layer.keep_query_pos = True
    persistent = layer(**inputs)
    assert persistent.std(dim=0, unbiased=False).mean() > 1e-3
    persistent.square().sum().backward()
    assert layer.ca_qpos_proj.weight.grad.norm() > 0


def test_one_object_key_has_legitimate_zero_query_projection_gradient():
    torch.manual_seed(42)
    layer = TransformerDecoderLayer(d_model=16, nhead=2, dropout=0., keep_query_pos=True)
    output = layer(tgt=torch.randn(4, 1, 16), memory=torch.randn(1, 1, 16),
                   query_pos=torch.randn(4, 1, 16), query_sine_embed=torch.randn(4, 1, 16),
                   pos=torch.randn(1, 1, 16), is_first=False)
    assert torch.isfinite(output).all()
    output.square().sum().backward()
    assert torch.count_nonzero(layer.ca_qpos_proj.weight.grad) == 0
