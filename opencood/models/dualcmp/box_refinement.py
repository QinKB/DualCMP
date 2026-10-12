"""Sparse4D@249ffbb box encoding and refinement, adapted to plain PyTorch.

Copyright (c) Horizon Robotics. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    input_dims = embed_dims if input_dims is None else input_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers += [nn.Linear(input_dims, embed_dims), nn.ReLU(inplace=True)]
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def encode_box(box, velocity=None):
    """DualCMP hwl boxes to Sparse4D [xyz,logw,logl,logh,sin,cos,vx,vy]."""
    if torch.any(box[..., 3:6] <= 0):
        raise ValueError("valid Sparse4D boxes require positive hwl")
    velocity = box.new_zeros(*box.shape[:-1], 2) if velocity is None else velocity
    return torch.cat((box[..., :3], box[..., [4, 5, 3]].log(),
                      box[..., 6:7].sin(), box[..., 6:7].cos(), velocity), dim=-1)


def decode_box(box):
    """Sparse4D code to DualCMP hwl boxes plus XY velocity."""
    yaw = torch.atan2(box[..., 6], box[..., 7])
    hwl = box[..., [5, 3, 4]].exp()
    return torch.cat((box[..., :3], hwl, yaw[..., None]), dim=-1), box[..., 8:10]


class SparseBox3DEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_fc = nn.Sequential(*linear_relu_ln(128, 1, 4, 3))
        self.size_fc = nn.Sequential(*linear_relu_ln(32, 1, 4, 3))
        self.yaw_fc = nn.Sequential(*linear_relu_ln(32, 1, 4, 2))
        self.vel_fc = nn.Sequential(*linear_relu_ln(64, 1, 4, 2))

    def forward(self, box):
        return torch.cat((self.pos_fc(box[..., :3]), self.size_fc(box[..., 3:6]),
                          self.yaw_fc(box[..., 6:8]), self.vel_fc(box[..., 8:10])), dim=-1)


class SparseBox3DRefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(*linear_relu_ln(256, 2, 2), nn.Linear(256, 10))
        self.scale = nn.Parameter(torch.ones(10))
        self.cls_layers = nn.Sequential(*linear_relu_ln(256, 1, 2), nn.Linear(256, 1))
        nn.init.constant_(self.cls_layers[-1].bias, -4.59511985013459)

    def forward(self, feature, anchor, anchor_embed, time_interval):
        output = self.layers(feature + anchor_embed) * self.scale
        output = torch.cat((output[..., :8] + anchor[..., :8],
                            output[..., 8:10] / time_interval[:, None, None] + anchor[..., 8:10]), dim=-1)
        return output, self.cls_layers(feature).squeeze(-1)
