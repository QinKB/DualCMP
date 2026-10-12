"""Tensor Kalman core adapted from DMSTrack's differentiable DKF.

Source: ``DMSTrack/differentiable_kalman_filter.py`` (DKF default
covariances and DifferentiableKalmanFilter.predict/update).  DMSTrack and the
bundled AB3DMOT license are academic/non-profit noncommercial research only.
"""
from __future__ import annotations

import torch


def matrices(dtype, device):
    """DMSTrack DKF's unit-frame F/H and AB3DMOT default P/Q/R."""
    f = torch.eye(10, dtype=dtype, device=device)
    f[0, 7], f[1, 8], f[2, 9] = 1.0, 1.0, 1.0
    h = torch.zeros(7, 10, dtype=dtype, device=device)
    h[:, :7] = torch.eye(7, dtype=dtype, device=device)
    p = torch.eye(10, dtype=dtype, device=device)
    q = torch.eye(10, dtype=dtype, device=device)
    q[7:, 7:] *= 0.01
    r = torch.eye(7, dtype=dtype, device=device)
    return f, h, p, q, r


def predict(mean, covariance):
    """Exact DMSTrack DifferentiableKalmanFilter.predict tensor equations."""
    f, _, _, q, _ = matrices(mean.dtype, mean.device)
    return (torch.matmul(mean, f.t()),
            torch.matmul(torch.matmul(f, covariance), f.t()) + q)


def update(mean, covariance, measurement):
    """Exact DMSTrack DifferentiableKalmanFilter.update tensor equations."""
    _, h, _, _, r = matrices(mean.dtype, mean.device)
    identity = torch.eye(10, dtype=mean.dtype, device=mean.device)
    innovation_covariance = torch.matmul(torch.matmul(h, covariance), h.t()) + r
    gain = torch.matmul(torch.matmul(covariance, h.t()), torch.linalg.inv(innovation_covariance))
    innovation = measurement - torch.matmul(mean, h.t())
    mean = mean + torch.matmul(gain, innovation.unsqueeze(-1)).squeeze(-1)
    covariance = torch.matmul(identity - torch.matmul(gain, h), covariance)
    return mean, covariance


def initialize(measurement):
    """DKF newborn: observation first seven dimensions and zero velocity."""
    _, _, p, _, _ = matrices(measurement.dtype, measurement.device)
    mean = torch.cat((measurement, torch.zeros(3, dtype=measurement.dtype,
                                                device=measurement.device)))
    return mean, p


def orientation_correction(theta_pre, theta_obs):
    """DMSTrack model.orientation_correction_torch, scalar tensor adaptation."""
    theta_pre = torch.atan2(torch.sin(theta_pre), torch.cos(theta_pre))
    theta_obs = torch.atan2(torch.sin(theta_obs), torch.cos(theta_obs))
    difference = torch.abs(theta_obs - theta_pre)
    theta_pre = torch.where((difference > torch.pi / 2) & (difference < torch.pi * 3 / 2),
                            theta_pre + torch.pi, theta_pre)
    theta_pre = torch.atan2(torch.sin(theta_pre), torch.cos(theta_pre))
    theta_pre = torch.where(torch.abs(theta_obs - theta_pre) >= torch.pi * 3 / 2,
                            theta_pre + torch.where(theta_obs > 0, torch.pi * 2, -torch.pi * 2), theta_pre)
    return theta_pre, theta_obs


def box_to_measurement(box):
    """DualCMP [x,y,z,h,w,l,yaw] to DKF [x,y,z,yaw,l,w,h]."""
    return torch.stack((box[..., 0], box[..., 1], box[..., 2], box[..., 6],
                        box[..., 5], box[..., 4], box[..., 3]), dim=-1)


def mean_to_box(mean):
    """DKF [x,y,z,yaw,l,w,h,...] to DualCMP [x,y,z,h,w,l,yaw]."""
    return torch.stack((mean[..., 0], mean[..., 1], mean[..., 2], mean[..., 6],
                        mean[..., 5], mean[..., 4], mean[..., 3]), dim=-1)
