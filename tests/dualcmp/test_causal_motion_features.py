"""Focused adapter checks; real two-GPU training remains the acceptance gate."""
import torch

from opencood.models.dualcmp.mtr_actor_adapter import causal_motion_features
from opencood.models.dualcmp.mtr_vendor import MTRDecoder
from easydict import EasyDict


def test_causal_constant_velocity_and_acceleration():
    time = torch.arange(11).float()[None] * .1
    xy = torch.stack((3 * time + time.square(), -2 * time), dim=-1)
    mask = torch.ones_like(time, dtype=torch.bool)
    velocity, acceleration = causal_motion_features(xy, mask, time, .1)
    assert torch.allclose(velocity[0, 1:, 0], 3 + 2 * time[0, 1:] - .1, atol=1e-5)
    assert torch.allclose(velocity[0, 1:, 1], torch.full((10,), -2.), atol=1e-5)
    assert torch.allclose(acceleration[0, 2:, 0], torch.full((9,), 2.), atol=1e-4)
    assert torch.count_nonzero(velocity[:, 0]) == 0
    assert torch.count_nonzero(acceleration[:, :2]) == 0


def test_missing_observation_and_time_gap_do_not_create_motion():
    time = torch.tensor([[0., .1, .2, .4, .5]])
    xy = torch.stack((time, time), dim=-1)
    mask = torch.tensor([[True, False, True, True, True]])
    velocity, acceleration = causal_motion_features(xy, mask, time, .1)
    assert torch.count_nonzero(velocity[:, :4]) == 0
    assert torch.allclose(velocity[:, 4], torch.ones((1, 2)))
    assert torch.count_nonzero(acceleration) == 0


def test_dense_motion_features_use_xy_velocity_not_gmm_uncertainty():
    class Dense(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("values", torch.tensor([[1., 2., 3., 4., .2], [2., 4., 5., 6., .3]]))

        def forward(self, features):
            return self.values.flatten()[None].expand(features.shape[0], -1)

    class Capture(torch.nn.Module):
        def forward(self, value):
            self.input = value.clone()
            return value[:, :4] * 0.

    decoder = MTRDecoder.__new__(MTRDecoder)
    torch.nn.Module.__init__(decoder)
    decoder.model_cfg = EasyDict(PHYSICAL_DENSE_MOTION=True, FRAME_PERIOD_S=.1)
    decoder.num_future_frames = 2
    decoder.obj_pos_encoding_layer = torch.nn.Identity()
    decoder.dense_future_head = Dense()
    decoder.future_traj_mlps = Capture()
    decoder.traj_fusion_mlps = torch.nn.Linear(8, 4)
    decoder.forward_ret_dict = {}
    args = (torch.zeros(1, 1, 4), torch.ones(1, 1, dtype=torch.bool), torch.tensor([[[10., 20., 0.]]]))
    decoder.apply_dense_future_prediction(*args)
    expected = torch.tensor([[11., 22., 10., 20., 12., 24., 10., 20.]])
    assert torch.equal(decoder.future_traj_mlps.input, expected)
    decoder.dense_future_head.values[:, 2:] += 17.
    decoder.apply_dense_future_prediction(*args)
    assert torch.equal(decoder.future_traj_mlps.input, expected)
