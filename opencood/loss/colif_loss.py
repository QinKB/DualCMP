"""Sequential tracking and anytime forecasting losses for fused-BEV CoLIF."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from opencood.loss.colif_gmm_loss import nearest_mode_gmm_loss


class ColifLoss(nn.Module):
    """MOTR-style persistent assignment with newborn-only Hungarian matching."""

    def __init__(self, args):
        super().__init__()
        weights = args.get("weights", {})
        self.aux_detection_weight = float(weights.get("aux_detection", 0.2))
        self.existence_weight = float(weights.get("existence", 1.0))
        self.box_weight = float(weights.get("box", 2.0))
        self.velocity_weight = float(weights.get("velocity", 0.25))
        self.decoder_aux_weight = float(weights.get("decoder_aux", 0.5))
        self.gmm_weight = float(weights.get("gmm", 1.0))
        self.mode_weight = float(weights.get("mode", 1.0))
        self.dense_gmm_weight = float(weights.get("dense_gmm", 1.0))
        matcher = args.get("matcher", {})
        self.cost_class = float(matcher.get("cost_class", 2.0))
        self.cost_center = float(matcher.get("cost_center", 1.0))
        self.cost_box = float(matcher.get("cost_box", 0.25))
        self.match_distance = float(matcher.get("max_center_distance", 12.0))
        self.supervision_ttl = int(matcher.get("supervision_ttl", 5))
        self.focal_alpha = float(args.get("focal_alpha", 0.25))
        self.focal_gamma = float(args.get("focal_gamma", 2.0))

    @staticmethod
    def _yaw_aware_box_loss(prediction, target, reduction="sum"):
        linear = F.smooth_l1_loss(
            prediction[..., :6], target[..., :6], reduction=reduction
        )
        yaw = 1.0 - torch.cos(prediction[..., 6] - target[..., 6])
        return linear + (yaw.sum() if reduction == "sum" else yaw)

    def _focal_sum(self, logits: torch.Tensor, target: torch.Tensor):
        probability = logits.sigmoid()
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = probability * target + (1.0 - probability) * (1.0 - target)
        alpha = self.focal_alpha * target + (1.0 - self.focal_alpha) * (1.0 - target)
        return (alpha * (1.0 - pt).pow(self.focal_gamma) * ce).sum()

    @staticmethod
    def _anchor_sin_difference(prediction: torch.Tensor, target: torch.Tensor):
        pred_sin = torch.sin(prediction[..., 6:7]) * torch.cos(target[..., 6:7])
        target_sin = torch.cos(prediction[..., 6:7]) * torch.sin(target[..., 6:7])
        return (
            torch.cat((prediction[..., :6], pred_sin), dim=-1),
            torch.cat((target[..., :6], target_sin), dim=-1),
        )

    @staticmethod
    def _receiver_displacement_to_actor(displacement, heading):
        """Match CMP/MTR's center-actor heading-normalized target frame."""

        # CMP's released V2V4Real MTR consumes
        # numeric degree headings as trigonometric angles.  This compatibility
        # conversion is deliberately local; tracking/state geometry remains
        # in radians.
        cmp_heading = torch.rad2deg(heading)
        cosine = torch.cos(cmp_heading)
        sine = torch.sin(cmp_heading)
        x, y = displacement[..., 0], displacement[..., 1]
        return torch.stack(
            (cosine * x + sine * y, -sine * x + cosine * y), dim=-1
        )

    @staticmethod
    def _future_relative_to_track(
        future_from_gt_current: torch.Tensor,
        gt_current_xy: torch.Tensor,
        track_current_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Express absolute GT future positions around the predicted track.

        The dataset stores future displacement relative to the GT current
        center because that representation is independent of model outputs.
        CMP/MTR, however, centers both its history input and future target on
        the predicted tracking box at the receiver time.  Re-centering here
        keeps the predictor input and supervision in the same frame.
        """

        future_absolute = gt_current_xy + future_from_gt_current
        return future_absolute - track_current_xy

    def _dense_context_future_loss(self, outputs, targets, pairs, zero):
        """Reproduce CMP's per-center dense context-future objective.

        The model exposes only anonymous packed query slots and predictions.
        Current-frame query-to-target pairs are criterion state, so GT IDs and
        future trajectories never enter model forward or recurrent state.
        CMP first averages valid context-actor losses within each center row,
        then averages center rows; return a row sum/count for DDP-equivalent
        normalization in ``weighted_sum``.
        """

        dense = outputs.get("packed_dense_trajs_local")
        entries = outputs.get("packed_entries", ())
        context_slots = outputs.get("packed_context_slots")
        headings = outputs.get("packed_center_heading")
        if (
            dense is None
            or context_slots is None
            or headings is None
            or dense.shape[0] == 0
        ):
            reference = dense if dense is not None else outputs["boxes"]
            return reference.sum() * 0.0, zero.detach().clone()

        pair_by_query = {
            (int(batch_index), int(query_index)): int(target_index)
            for batch_index, query_index, target_index in pairs
        }
        center_row_sum = dense.sum() * 0.0
        center_row_count = dense.new_zeros(())
        for row, (batch_index, center_slot) in enumerate(entries):
            center_xy = outputs["track_boxes"][
                batch_index, center_slot, :2
            ]
            row_prediction, row_target, row_mask = [], [], []
            for context_index, query_slot in enumerate(
                context_slots[row].tolist()
            ):
                if query_slot < 0:
                    continue
                target_index = pair_by_query.get((batch_index, query_slot))
                if target_index is None:
                    continue
                future_mask = targets["future_mask"][batch_index, target_index]
                if not bool(future_mask.any()):
                    continue
                future_receiver = self._future_relative_to_track(
                    targets["future_xy"][batch_index, target_index],
                    targets["current_boxes"][batch_index, target_index, :2],
                    center_xy,
                )
                row_prediction.append(dense[row, context_index])
                row_target.append(
                    self._receiver_displacement_to_actor(
                        future_receiver, headings[row]
                    )
                )
                row_mask.append(future_mask)
            if not row_prediction:
                continue
            prediction = torch.stack(row_prediction)
            target = torch.stack(row_target)
            mask = torch.stack(row_mask)
            fake_scores = prediction.new_zeros(prediction.shape[0], 1)
            row_loss = nearest_mode_gmm_loss(
                fake_scores,
                prediction[:, None],
                target,
                mask,
            )
            valid_context_count = row_loss["gmm_count"].clamp_min(1.0)
            center_row_sum = center_row_sum + row_loss["gmm"] / valid_context_count
            center_row_count = center_row_count + 1.0
        return center_row_sum, center_row_count

    def _auxiliary_fused_detection(self, outputs, targets):
        psm = outputs["auxiliary_psm"]
        rm = outputs["auxiliary_rm"]
        labels = targets["fused_label_dict"]
        positive = labels["pos_equal_one"].reshape(psm.shape[0], -1) > 0
        negative = labels["neg_equal_one"].reshape(psm.shape[0], -1) > 0
        valid = positive | negative
        logits = psm.permute(0, 2, 3, 1).reshape(psm.shape[0], -1)
        cls_target = positive.to(logits.dtype)
        cls_sum = self._focal_sum(logits[valid], cls_target[valid])

        prediction = rm.permute(0, 2, 3, 1).reshape(rm.shape[0], -1, 7)
        target = labels["targets"].reshape(rm.shape[0], -1, 7).to(rm.dtype)
        prediction, target = self._anchor_sin_difference(prediction, target)
        difference = prediction - target
        absolute = difference.abs()
        beta = 1.0 / 9.0
        smooth = torch.where(
            absolute < beta,
            0.5 * difference.square() / beta,
            absolute - 0.5 * beta,
        )
        reg_sum = (smooth * positive[..., None].to(smooth.dtype)).sum()
        count = positive.sum().to(rm.dtype).clamp_min(1.0)
        return cls_sum, reg_sum, count

    def _newborn_match(self, outputs, targets, previous_state):
        batch_size, query_count = outputs["exist_logits"].shape
        if previous_state is None:
            previous_ids = torch.full(
                (batch_size, query_count),
                -1,
                dtype=torch.long,
                device=outputs["exist_logits"].device,
            )
            previous_miss = torch.zeros_like(previous_ids)
        else:
            previous_ids = previous_state["actor_ids"].clone()
            previous_miss = previous_state["miss_count"].clone()
            # Criterion identity may persist only while the corresponding
            # runtime query slot persists.  The old implementation retained a
            # GT actor ID after the model had killed/reset that slot, then
            # forced the reset newborn content to the old actor on the next
            # frame.  This trained a different state machine from inference.
            runtime_active = outputs.get("active_mask")
            if runtime_active is not None:
                reset_slots = (previous_ids >= 0) & ~runtime_active.bool()
                previous_ids[reset_slots] = -1
                previous_miss[reset_slots] = 0

        pairs = []
        next_ids = previous_ids.clone()
        next_miss = previous_miss.clone()
        for batch_index in range(batch_size):
            valid_targets = torch.where(targets["current_mask"][batch_index])[0]
            target_by_id = {
                int(targets["actor_ids"][batch_index, index]): int(index)
                for index in valid_targets
            }
            matched_target_ids = set()
            # Persistent slots retain their identity.  Only their matching
            # target index is looked up; no global Hungarian rematch occurs.
            for slot in torch.where(previous_ids[batch_index] >= 0)[0].tolist():
                actor_id = int(previous_ids[batch_index, slot])
                if actor_id in target_by_id:
                    target_index = target_by_id[actor_id]
                    pairs.append((batch_index, slot, target_index))
                    matched_target_ids.add(actor_id)
                    next_miss[batch_index, slot] = 0
                else:
                    next_miss[batch_index, slot] += 1
                    if next_miss[batch_index, slot] > self.supervision_ttl:
                        next_ids[batch_index, slot] = -1
                        next_miss[batch_index, slot] = 0

            remaining_targets = torch.tensor(
                [
                    int(index)
                    for index in valid_targets
                    if int(targets["actor_ids"][batch_index, index])
                    not in matched_target_ids
                ],
                device=valid_targets.device,
                dtype=torch.long,
            )
            available = next_ids[batch_index] < 0
            birth_candidates = outputs.get("birth_candidate_mask")
            runtime_active = outputs.get("active_mask")
            if birth_candidates is not None and runtime_active is not None:
                # A clip may contain a supervision-free causal burn-in.  Its
                # mature runtime queries have no criterion identity yet, but
                # must be eligible for the first supervised Hungarian match.
                # Inactive slots remain eligible only when seeded by a current
                # dense measurement.
                available = available & (
                    runtime_active[batch_index].bool()
                    | birth_candidates[batch_index].bool()
                )
            elif birth_candidates is not None:
                available = available & birth_candidates[batch_index].bool()
            available_slots = torch.where(available)[0]
            if not remaining_targets.numel() or not available_slots.numel():
                continue
            predicted_box = outputs["boxes"][batch_index, available_slots]
            target_box = targets["current_boxes"][batch_index, remaining_targets]
            center_distance = torch.cdist(
                predicted_box[:, :2], target_box[:, :2], p=1
            )
            dimension_distance = torch.cdist(
                predicted_box[:, 3:6], target_box[:, 3:6], p=1
            )
            class_cost = -outputs["exist_logits"][
                batch_index, available_slots
            ].sigmoid()[:, None]
            cost = (
                self.cost_center * center_distance
                + self.cost_box * dimension_distance
                + self.cost_class * class_cost
            )
            rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
            for row, column in zip(rows.tolist(), columns.tolist()):
                if float(center_distance[row, column].detach()) > self.match_distance:
                    continue
                slot = int(available_slots[row])
                target_index = int(remaining_targets[column])
                actor_id = int(targets["actor_ids"][batch_index, target_index])
                pairs.append((batch_index, slot, target_index))
                next_ids[batch_index, slot] = actor_id
                next_miss[batch_index, slot] = 0
        return pairs, {
            "actor_ids": next_ids.detach(),
            "miss_count": next_miss.detach(),
        }

    def forward_frame(self, outputs, aux, targets, prev_supervision_state=None):
        pairs, next_supervision_state = self._newborn_match(
            outputs, targets, prev_supervision_state
        )
        device = outputs["boxes"].device
        dtype = outputs["boxes"].dtype
        zero = outputs["boxes"].sum() * 0.0
        existence_target = torch.zeros_like(outputs["exist_logits"])
        box_sum = zero
        velocity_sum = zero
        matched_scores, matched_trajs, matched_future, matched_mask = [], [], [], []
        matched_intentions = []
        state_boxes = outputs.get("track_boxes", outputs["boxes"])
        state_velocity = outputs.get("track_velocity", outputs["velocity"])
        for batch_index, query_index, target_index in pairs:
            existence_target[batch_index, query_index] = 1.0
            box_sum = box_sum + self._yaw_aware_box_loss(
                state_boxes[batch_index, query_index],
                targets["current_boxes"][batch_index, target_index],
            )
            velocity_sum = velocity_sum + F.smooth_l1_loss(
                state_velocity[batch_index, query_index],
                targets["current_velocity"][batch_index, target_index],
                reduction="sum",
            )
            if (
                "pred_scores" in outputs
                and bool(outputs["forecast_valid_mask"][batch_index, query_index])
                and bool(targets["future_mask"][batch_index, target_index].any())
            ):
                matched_scores.append(
                    outputs.get(
                        "loss_pred_scores_layers",
                        outputs.get("loss_pred_scores", outputs["pred_scores"])[
                            :, :, None
                        ],
                    )[batch_index, query_index]
                )
                # The reused CMP/MTR decoder and its pretrained GMM parameters
                # operate in the center actor's heading-normalized frame.
                # Public trajectories are rotated back to receiver axes for
                # evaluation, while the loss stays in MTR's native frame.
                matched_trajs.append(
                    outputs.get(
                        "loss_pred_trajs_local_layers",
                        outputs.get(
                            "loss_pred_trajs_local",
                            outputs.get(
                                "pred_trajs_local", outputs["pred_trajs"]
                            ),
                        )[:, :, None],
                    )[batch_index, query_index]
                )
                future_receiver = self._future_relative_to_track(
                    targets["future_xy"][batch_index, target_index],
                    targets["current_boxes"][batch_index, target_index, :2],
                    outputs["track_boxes"][batch_index, query_index, :2],
                )
                matched_future.append(
                    future_receiver
                    if bool(outputs.get("forecast_loss_in_receiver_frame", False))
                    else self._receiver_displacement_to_actor(
                        future_receiver,
                        outputs.get(
                            "forecast_frame_heading",
                            outputs["track_boxes"][..., 6],
                        )[batch_index, query_index],
                    )
                )
                matched_mask.append(targets["future_mask"][batch_index, target_index])
                matched_intentions.append(
                    outputs.get(
                        "loss_intention_points_layers",
                        outputs.get("loss_intention_points", None),
                    )[batch_index, query_index]
                    if (
                        "loss_intention_points_layers" in outputs
                        or "loss_intention_points" in outputs
                    ) else None
                )

        existence_sum = self._focal_sum(outputs["exist_logits"], existence_target)
        aux_exist_sum = zero
        aux_box_sum = zero
        for layer_logits, layer_boxes in zip(
            outputs["aux_exist_logits"], outputs["aux_boxes"]
        ):
            aux_exist_sum = aux_exist_sum + self._focal_sum(
                layer_logits, existence_target
            )
            for batch_index, query_index, target_index in pairs:
                aux_box_sum = aux_box_sum + self._yaw_aware_box_loss(
                    layer_boxes[batch_index, query_index],
                    targets["current_boxes"][batch_index, target_index],
                )

        fused_cls, fused_reg, anchor_count = self._auxiliary_fused_detection(
            outputs, targets
        )
        match_count = torch.tensor(float(len(pairs)), device=device, dtype=dtype)
        query_count = torch.tensor(
            float(outputs["exist_logits"].numel()), device=device, dtype=dtype
        )
        decoder_layer_count = max(1, int(outputs["aux_exist_logits"].shape[0]))
        loss_dict = {
            "fused_cls": fused_cls,
            "fused_reg": fused_reg,
            "existence": existence_sum,
            "box": box_sum,
            "velocity": velocity_sum,
            "aux_existence": aux_exist_sum,
            "aux_box": aux_box_sum,
        }
        count_dict = {
            "fused_cls": anchor_count,
            "fused_reg": anchor_count,
            "existence": query_count,
            "box": match_count.clamp_min(1.0),
            "velocity": match_count.clamp_min(1.0),
            "aux_existence": query_count * decoder_layer_count,
            "aux_box": match_count.clamp_min(1.0) * decoder_layer_count,
            "matched": match_count,
        }
        if matched_scores:
            scores_by_actor = torch.stack(matched_scores)
            trajs_by_actor = torch.stack(matched_trajs)
            future_by_actor = torch.stack(matched_future)
            mask_by_actor = torch.stack(matched_mask)
            intentions = (
                torch.stack(matched_intentions)
                if matched_intentions and matched_intentions[0] is not None
                else None
            )
            layer_count = scores_by_actor.shape[1]
            layer_losses = [
                nearest_mode_gmm_loss(
                    scores_by_actor[:, layer_index],
                    trajs_by_actor[:, layer_index],
                    future_by_actor,
                    mask_by_actor,
                    intention_points=(
                        intentions[:, layer_index]
                        if intentions is not None and intentions.ndim == 4
                        else intentions
                    ),
                )
                for layer_index in range(layer_count)
            ]
            # Deep-supervise both the motion proposal and auxiliary cross-mode
            # refinement, then average layers. Runtime may intentionally serve
            # the proposal layer when the refinement hurts endpoint accuracy.
            loss_dict["gmm"] = torch.stack(
                [item["gmm"] for item in layer_losses]
            ).mean()
            loss_dict["mode"] = torch.stack(
                [item["mode"] for item in layer_losses]
            ).mean()
            count_dict["gmm"] = layer_losses[-1]["gmm_count"]
            count_dict["mode"] = layer_losses[-1]["mode_count"]
        else:
            forecast_reference = outputs.get("pred_trajs", outputs["boxes"])
            score_reference = outputs.get("pred_scores", outputs["exist_logits"])
            loss_dict["gmm"] = forecast_reference.sum() * 0.0
            loss_dict["mode"] = score_reference.sum() * 0.0
            # A rank with no forecast target contributes zero to both the
            # numerator and denominator.  The global denominator is clamped
            # only after all-reduce; inserting a fake count would bias DDP.
            count_dict["gmm"] = torch.zeros((), device=device, dtype=dtype)
            count_dict["mode"] = torch.zeros((), device=device, dtype=dtype)
        dense_gmm, dense_center_count = self._dense_context_future_loss(
            outputs, targets, pairs, zero
        )
        loss_dict["dense_gmm"] = dense_gmm
        count_dict["dense_gmm"] = dense_center_count
        aux["matched_query_target_pairs"] = pairs
        aux["matched_count"] = match_count
        return loss_dict, count_dict, next_supervision_state

    @staticmethod
    def _normalize(value, count):
        if dist.is_available() and dist.is_initialized():
            global_count = count.detach().clone()
            dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
            return value * dist.get_world_size() / global_count.clamp_min(1.0)
        return value / count.clamp_min(1.0)

    def weighted_sum(self, loss_dict, count_dict, phase_cfg=None):
        phase = int((phase_cfg or {}).get("phase", 2))
        if phase not in (1, 2):
            raise ValueError("fused CoLIF uses two training phases")
        names = {
            1: (
                "fused_cls", "fused_reg", "existence", "box", "velocity",
                "aux_existence", "aux_box",
            ),
            2: (
                "existence", "box", "velocity", "aux_existence", "aux_box",
                "gmm", "mode", "dense_gmm",
            ),
        }[phase]
        global_counts = torch.stack(
            [count_dict[name].detach().float() for name in names]
        )
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
        count_by_name = {
            name: global_counts[index].clamp_min(1.0)
            for index, name in enumerate(names)
        }
        norm = lambda name: (
            loss_dict[name] * world_size / count_by_name[name]
        )
        tracking = (
            self.existence_weight * norm("existence")
            + self.box_weight * norm("box")
            + self.velocity_weight * norm("velocity")
            + self.decoder_aux_weight
            * (norm("aux_existence") + self.box_weight * norm("aux_box"))
        )
        if phase == 1:
            auxiliary_detection = self.aux_detection_weight * (
                norm("fused_cls") + 2.0 * norm("fused_reg")
            )
            return auxiliary_detection + tracking
        forecast = (
            self.gmm_weight * norm("gmm")
            + self.mode_weight * norm("mode")
            + self.dense_gmm_weight * norm("dense_gmm")
        )
        recurrence_mode = str(
            (phase_cfg or {}).get(
                "future_prior_mode",
                (phase_cfg or {}).get("feedback_mode", "none"),
            )
        )
        if recurrence_mode not in ("propagation_bptt", "recurrent_bptt"):
            return forecast
        return tracking + forecast

    def forward(self, outputs, targets):
        loss_dict, count_dict, _ = self.forward_frame(outputs, {}, targets, None)
        return self.weighted_sum(loss_dict, count_dict)
