"""Legacy graph identity and recoverable positive-scale likelihood gradients."""
import torch

from opencood.models.dualcmp.intention_query_forecast import parameterize_gmm_variance
from opencood.loss.dualcmp_gmm_loss import nearest_mode_gmm_loss


def test_positive_scale_preserves_geometry_and_avoids_exponential_dead_upper_clamp():
    raw = torch.zeros(1, 1, 50, 5, requires_grad=True)
    with torch.no_grad():
        raw[..., 2:4] = 6.
        raw[..., 4] = .1
    assert parameterize_gmm_variance(raw, 'log_std_clamp_v1') is raw
    mask = torch.ones(1, 50, dtype=torch.bool)
    logits, target = torch.zeros(1, 1), torch.zeros(1, 50, 2)
    legacy = nearest_mode_gmm_loss(logits, raw, target, mask)['gmm']
    legacy.backward()
    assert raw.grad[..., 2:4].abs().sum() == 0
    raw.grad.zero_()
    positive = parameterize_gmm_variance(raw, 'softplus_std_v1')
    assert torch.equal(raw[..., :2], positive[..., :2])
    assert torch.equal(raw[..., 4], positive[..., 4])
    assert torch.isfinite(positive).all() and (positive[..., 2:4].exp() > .2).all()
    nearest_mode_gmm_loss(logits, positive, target, mask)['gmm'].backward()
    assert (raw.grad[..., 2:4] > 0).all()  # gradient descent reduces excess uncertainty
