"""Stable imports for CMP's bundled V2V4Real MTR implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MTR_ROOT = PROJECT_ROOT / "MTR"
if str(MTR_ROOT) not in sys.path:
    sys.path.insert(0, str(MTR_ROOT))

from mtr.models_v2v4real.context_encoder.mtr_encoder import MTREncoder
from mtr.models_v2v4real.motion_decoder.mtr_decoder import (
    MTRDecoder as OfficialMTRDecoder,
)
from mtr.models_v2v4real.utils.transformer import position_encoding_utils
from mtr.utils.motion_utils import batch_nms
from mtr.utils.loss_utils import nll_loss_gmm_direct


class MTRDecoder(OfficialMTRDecoder):
    """Official CMP decoder with only the DualCMP evidence insertion point."""

    def forward(
        self, batch_dict, query_feature_hook=None, store_training_targets=True
    ):
        self._dualcmp_query_feature_hook = query_feature_hook
        training = self.training
        if not store_training_targets:
            # Skip only the official target-copy branch. Child dropout modules
            # retain their own training state.
            self.training = False
        try:
            return super().forward(batch_dict)
        finally:
            self.training = training
            self._dualcmp_query_feature_hook = None

    def apply_transformer_decoder(
        self,
        center_objects_feature,
        center_objects_type,
        obj_feature,
        obj_mask,
        obj_pos,
    ):
        """Copy CMP MTRDecoder.apply_transformer_decoder and add one hook."""
        intention_query, intention_points = self.get_motion_query(
            center_objects_type
        )
        query_content = torch.zeros_like(intention_query)
        self.forward_ret_dict["intention_points"] = intention_points.permute(
            1, 0, 2
        )

        num_center_objects = query_content.shape[1]
        num_query = query_content.shape[0]
        center_objects_feature = center_objects_feature[None, :, :].repeat(
            num_query, 1, 1
        )
        base_map_idxs = None
        pred_waypoints = intention_points.permute(1, 0, 2)[:, :, None, :]
        dynamic_query_center = intention_points
        pred_list = []
        for layer_idx in range(self.num_decoder_layers):
            obj_query_feature = self.apply_cross_attention(
                kv_feature=obj_feature,
                kv_mask=obj_mask,
                kv_pos=obj_pos,
                query_content=query_content,
                query_embed=intention_query,
                attention_layer=self.obj_decoder_layers[layer_idx],
                dynamic_query_center=dynamic_query_center,
                layer_idx=layer_idx,
            )
            query_feature = torch.cat(
                [center_objects_feature, obj_query_feature], dim=-1
            )
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
            ).view(num_query, num_center_objects, -1)
            hook = getattr(self, "_dualcmp_query_feature_hook", None)
            if hook is not None:
                query_content = hook(
                    query_content=query_content,
                    intention_query=intention_query,
                    dynamic_query_center=dynamic_query_center,
                    layer_idx=layer_idx,
                )

            query_content_t = query_content.permute(1, 0, 2).contiguous().view(
                num_center_objects * num_query, -1
            )
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(
                num_center_objects, num_query
            )
            if self.motion_vel_heads is not None:
                pred_trajs = self.motion_reg_heads[layer_idx](
                    query_content_t
                ).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    5,
                )
                pred_vel = self.motion_vel_heads[layer_idx](
                    query_content_t
                ).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    2,
                )
                pred_trajs = torch.cat((pred_trajs, pred_vel), dim=-1)
            else:
                pred_trajs = self.motion_reg_heads[layer_idx](
                    query_content_t
                ).view(
                    num_center_objects,
                    num_query,
                    self.num_future_frames,
                    5,
                )
            pred_list.append([pred_scores, pred_trajs])
            pred_waypoints = pred_trajs[:, :, :, 0:2]
            dynamic_query_center = (
                pred_trajs[:, :, -1, 0:2].contiguous().permute(1, 0, 2)
            )

        if self.use_place_holder:
            raise NotImplementedError
        assert len(pred_list) == self.num_decoder_layers
        return pred_list


__all__ = [
    "MTREncoder",
    "MTRDecoder",
    "batch_nms",
    "nll_loss_gmm_direct",
    "position_encoding_utils",
]
