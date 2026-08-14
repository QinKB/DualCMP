"""Shared PointPillar trunk applied independently to each source."""

from __future__ import annotations

import torch.nn as nn

from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter


class SharedLocalPillarTrunk(nn.Module):
    """Encode flattened source-native voxels without regroup or BEV fusion."""

    def __init__(self, args):
        super().__init__()
        self.pillar_vfe = PillarVFE(
            args["pillar_vfe"],
            num_point_features=4,
            voxel_size=args["voxel_size"],
            point_cloud_range=args["lidar_range"],
        )
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        self.shrink_conv = None
        if "shrink_header" in args:
            self.shrink_conv = DownsampleConv(args["shrink_header"])

        compression_ratio = int(args.get("compression", 0))
        self.naive_compressor = None
        if compression_ratio > 0:
            self.naive_compressor = NaiveCompressor(256, compression_ratio)

        local_trunk_args = args.get("local_trunk", {})
        self.feature_tap = local_trunk_args.get("feature_tap", "after_shrink")
        if self.feature_tap not in ("after_shrink", "after_compressor"):
            raise ValueError("Unknown local_trunk.feature_tap: %s" % self.feature_tap)
        if self.feature_tap == "after_compressor" and self.naive_compressor is None:
            raise ValueError("after_compressor requires model.args.compression > 0")

        if bool(local_trunk_args.get("freeze", False)):
            self.set_trainable(False)

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(trainable)

    def forward(self, runtime_frame):
        processed_lidar = runtime_frame["processed_lidar"]
        batch_dict = {
            "voxel_features": processed_lidar["voxel_features"],
            "voxel_coords": processed_lidar["voxel_coords"],
            "voxel_num_points": processed_lidar["voxel_num_points"],
            "record_len": runtime_frame["record_len"],
        }
        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        expected_sources = int(runtime_frame["source_to_batch"].numel())
        actual_sources = int(batch_dict["spatial_features"].shape[0])
        if actual_sources != expected_sources:
            raise RuntimeError(
                "Scatter/source mapping mismatch: %d BEVs for %d sources"
                % (actual_sources, expected_sources)
            )
        batch_dict = self.backbone(batch_dict)
        feature = batch_dict["spatial_features_2d"]
        if self.shrink_conv is not None:
            feature = self.shrink_conv(feature)
        if self.feature_tap == "after_compressor":
            feature = self.naive_compressor(feature)
        return feature
