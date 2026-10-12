"""Stable imports for CMP's bundled V2V4Real MTR implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MTR_ROOT = PROJECT_ROOT / "MTR"
if str(MTR_ROOT) not in sys.path:
    sys.path.insert(0, str(MTR_ROOT))

from mtr.models_v2v4real.context_encoder.mtr_encoder import MTREncoder as OfficialMTREncoder
from mtr.models_v2v4real.motion_decoder.mtr_decoder import (
    MTRDecoder as OfficialMTRDecoder,
)
from mtr.models_v2v4real.utils.transformer import position_encoding_utils
from mtr.utils.motion_utils import batch_nms
from mtr.utils.loss_utils import nll_loss_gmm_direct
from mtr.ops.attention import attention_utils_v2
from opencood.models.dualcmp import mtr_attention

# Native v2 backward atomicAdd bypasses PyTorch's deterministic switch.
# Change execution only; keep the local neighbor graph and attention math.
for _name in ("attention_weight_computation", "attention_value_computation"):
    setattr(attention_utils_v2, _name, mtr_attention.deterministic_or_native(
        getattr(mtr_attention, _name), getattr(attention_utils_v2, _name)))


class MTREncoder(OfficialMTREncoder):
    """Keep released MTR intact; optionally enrich actors before scene attention."""

    def forward(self, batch_dict, actor_feature_hook=None):
        if actor_feature_hook is None:
            return super().forward(batch_dict)
        handle = self.agent_polyline_encoder.register_forward_hook(
            lambda module, inputs, output: actor_feature_hook(output))
        try:
            return super().forward(batch_dict)
        finally:
            handle.remove()


class MTRDecoder(OfficialMTRDecoder):
    """Official CMP decoder with only the DualCMP evidence insertion point."""

    def apply_dense_future_prediction(self, obj_feature, obj_mask, obj_pos):
        if not self.model_cfg.get("PHYSICAL_DENSE_MOTION", False):
            return super().apply_dense_future_prediction(obj_feature, obj_mask, obj_pos)
        # Copy CMP's existing dense head/fusion path. Its 5D GMM output has no
        # velocity channels: indices [-2,-1] are log_std_y and correlation.
        # Restore original MTR future feature semantics [x,y,vx,vy] using the
        # predicted means only. Do not add future GT, heads or loss terms.
        num_centers, num_objects, _ = obj_feature.shape
        positions = obj_pos[obj_mask][..., :2]
        features = obj_feature[obj_mask]
        fused = torch.cat((self.obj_pos_encoding_layer(positions), features), dim=-1)
        if fused.shape[0] == 1:
            trajectories = self.dense_future_head(fused.repeat(2, 1))[:1]
        else:
            trajectories = self.dense_future_head(fused)
        trajectories = trajectories.view(-1, self.num_future_frames, 5)
        xy = trajectories[..., :2] + positions[:, None]
        trajectories = torch.cat((xy, trajectories[..., 2:]), dim=-1)
        previous_xy = torch.cat((positions[:, None], xy[:, :-1]), dim=1)
        velocity = (xy - previous_xy) / float(self.model_cfg.FRAME_PERIOD_S)
        future_input = torch.cat((xy, velocity), dim=-1).flatten(start_dim=1)
        future_feature = self.future_traj_mlps(future_input)
        full = self.traj_fusion_mlps(torch.cat((features, future_feature), dim=-1))
        output_feature = torch.zeros_like(obj_feature)
        output_feature[obj_mask] = full
        output_trajectories = obj_feature.new_zeros(num_centers, num_objects, self.num_future_frames, 5)
        output_trajectories[obj_mask] = trajectories
        self.forward_ret_dict["pred_dense_trajs"] = output_trajectories
        return output_feature, output_trajectories

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
            layer_input_query = query_content
            obj_query_feature = self.apply_cross_attention(
                kv_feature=obj_feature,
                kv_mask=obj_mask,
                kv_pos=obj_pos,
                query_content=layer_input_query,
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
                    layer_input_query=layer_input_query,
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
