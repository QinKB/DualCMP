"""Strict-online sequence-preserving evaluation for fused-BEV CoLIF."""

from __future__ import annotations

import copy
from collections import defaultdict

import torch
import torch.distributed as dist
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Sampler

from opencood.data_utils.v2v4real.datasets import build_v2v4real_dataset
from opencood.tools import train_utils


class SceneReceiverDistributedSampler(Sampler):
    """Shard complete receiver timelines and retain chronological order."""

    def __init__(self, dataset, rank=0, world_size=1):
        groups = defaultdict(list)
        for index, record in enumerate(dataset.clip_records):
            groups[(record.scene_id, record.receiver_id)].append(
                (record.start_position, index)
            )
        ordered = [
            [index for _, index in sorted(records)]
            for _, records in sorted(groups.items())
        ]
        rank_groups = [[] for _ in range(world_size)]
        rank_sizes = [0] * world_size
        for group in sorted(ordered, key=len, reverse=True):
            destination = min(range(world_size), key=lambda value: rank_sizes[value])
            rank_groups[destination].append(group)
            rank_sizes[destination] += len(group)
        self.indices = [index for group in rank_groups[rank] for index in group]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def build_colif_evaluation_loader(
    hypes, split="test", rank=0, world_size=1, workers=0
):
    if split != "test":
        raise ValueError("CoLIF follows CMP and evaluates the official test split only")
    evaluation_hypes = copy.deepcopy(hypes)
    fusion = evaluation_hypes["fusion"]["args"]
    fusion["runtime_only"] = False
    fusion.pop("local_targets_only", None)
    fusion["warmup_length"] = 0
    fusion["unroll_length"] = 1
    fusion["clip_stride"] = 1
    fusion["split_name"] = split
    dataset = build_v2v4real_dataset(evaluation_hypes, train=False)
    sampler = SceneReceiverDistributedSampler(dataset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=workers,
        collate_fn=dataset.collate_batch_test,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    return dataset, loader


def _hungarian_matches(prediction_xy, target_xy, prediction_indices, target_indices, limit):
    if not prediction_indices.numel() or not target_indices.numel():
        return []
    cost = torch.cdist(
        prediction_xy[prediction_indices].float(),
        target_xy[target_indices].float(),
        p=2,
    )
    rows, columns = linear_sum_assignment(cost.cpu().numpy())
    return [
        (int(prediction_indices[row]), int(target_indices[column]))
        for row, column in zip(rows.tolist(), columns.tolist())
        if float(cost[row, column]) <= limit
    ]


def _absolute_forecast_error(
    predicted_displacement: torch.Tensor,
    track_current_xy: torch.Tensor,
    target_displacement: torch.Tensor,
    target_current_xy: torch.Tensor,
) -> torch.Tensor:
    """Return mode-wise errors in the receiver's absolute coordinate frame."""

    predicted_absolute = predicted_displacement + track_current_xy
    target_absolute = target_displacement + target_current_xy
    return torch.norm(
        predicted_absolute - target_absolute.unsqueeze(0), dim=-1
    )


def _history_velocity_estimates(
    history_xy: torch.Tensor,
    history_mask: torch.Tensor,
    frame_period_s: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return OLS and robust pairwise-median velocities from online history.

    This is an evaluation-only diagnostic.  It consumes exactly the same
    anonymous runtime history as MTR and never reads target positions.  The
    comparison identifies whether a poor forecast is caused by the state
    history itself or by its learned decoder.
    """

    valid = torch.where(history_mask)[0]
    zero = history_xy.new_zeros(2)
    if valid.numel() < 2:
        return zero, zero
    times = valid.to(history_xy.dtype) * float(frame_period_s)
    positions = history_xy[valid]
    centered_time = times - times.mean()
    denominator = centered_time.square().sum().clamp_min(1e-6)
    ols = (
        centered_time[:, None] * (positions - positions.mean(dim=0))
    ).sum(dim=0) / denominator

    pair_velocity = []
    for earlier in range(valid.numel() - 1):
        elapsed = times[earlier + 1 :] - times[earlier]
        pair_velocity.append(
            (positions[earlier + 1 :] - positions[earlier])
            / elapsed[:, None].clamp_min(1e-6)
        )
    median = torch.cat(pair_velocity, dim=0).median(dim=0).values
    return ols, median


def _history_linear_predictions(
    history_xy: torch.Tensor,
    history_mask: torch.Tensor,
    future_times_s: torch.Tensor,
    frame_period_s: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extrapolate fitted online tracks, including a denoised intercept.

    The returned values are displacements from the latest runtime state, so
    they use the same public forecast contract as CoLIF.  This remains an
    evaluation-only diagnosis and consumes no target/future values.
    """

    valid = torch.where(history_mask)[0]
    zeros = history_xy.new_zeros(future_times_s.shape[0], 2)
    if valid.numel() < 2:
        return zeros, zeros
    times = valid.to(history_xy.dtype) * float(frame_period_s)
    positions = history_xy[valid]
    ols_velocity, median_velocity = _history_velocity_estimates(
        history_xy, history_mask, frame_period_s=frame_period_s
    )
    ols_intercept = (
        positions - times[:, None] * ols_velocity[None]
    ).mean(dim=0)
    median_intercept = (
        positions - times[:, None] * median_velocity[None]
    ).median(dim=0).values
    current_time = (history_xy.shape[0] - 1) * float(frame_period_s)
    prediction_times = current_time + future_times_s.to(history_xy.dtype)
    current_xy = history_xy[-1]
    return (
        ols_intercept[None]
        + prediction_times[:, None] * ols_velocity[None]
        - current_xy[None],
        median_intercept[None]
        + prediction_times[:, None] * median_velocity[None]
        - current_xy[None],
    )


def _kinematic_lateral_mode_bank(
    velocity: torch.Tensor,
    heading: torch.Tensor,
    future_times_s: torch.Tensor,
    lateral_endpoint_m: float,
) -> torch.Tensor:
    """Build six causal motion hypotheses with symmetric lateral coverage."""

    speed = torch.linalg.vector_norm(velocity)
    heading_direction = torch.stack((torch.cos(heading), torch.sin(heading)))
    direction = torch.where(
        speed > 0.5,
        velocity / speed.clamp_min(1e-6),
        heading_direction,
    )
    perpendicular = torch.stack((-direction[1], direction[0]))
    longitudinal = torch.stack(
        [
            future_times_s[:, None] * velocity[None] * scale
            for scale in (0.0, 0.5, 1.0, 1.5, 1.0, 1.0)
        ],
        dim=0,
    )
    normalized_time = future_times_s / future_times_s[-1].clamp_min(1e-6)
    lateral_profile = normalized_time.square()[:, None] * perpendicular[None]
    longitudinal[4] = longitudinal[4] + float(lateral_endpoint_m) * lateral_profile
    longitudinal[5] = longitudinal[5] - float(lateral_endpoint_m) * lateral_profile
    return longitudinal


def _residual_prototype_mode_bank(
    velocity: torch.Tensor,
    heading: torch.Tensor,
    future_times_s: torch.Tensor,
    prototypes_longitudinal_lateral_m: torch.Tensor,
) -> torch.Tensor:
    """Apply train-only local endpoint residual prototypes to causal CV."""

    prototypes = prototypes_longitudinal_lateral_m.to(
        device=velocity.device, dtype=velocity.dtype
    )
    if prototypes.ndim != 2 or prototypes.shape[1] != 2:
        raise ValueError("residual prototypes must have shape [K,2]")
    speed = torch.linalg.vector_norm(velocity)
    heading_direction = torch.stack((torch.cos(heading), torch.sin(heading)))
    direction = torch.where(
        speed > 0.5,
        velocity / speed.clamp_min(1e-6),
        heading_direction,
    )
    perpendicular = torch.stack((-direction[1], direction[0]))
    residual_receiver = (
        prototypes[:, :1] * direction[None]
        + prototypes[:, 1:] * perpendicular[None]
    )
    normalized_time = future_times_s / future_times_s[-1].clamp_min(1e-6)
    return (
        future_times_s[None, :, None] * velocity[None, None]
        + normalized_time[None, :, None].square()
        * residual_receiver[:, None]
    )


def evaluate_colif_strict_online(
    model,
    loader,
    device,
    match_distance=12.0,
    horizons=(10, 30, 50),
    miss_thresholds=(2.0, 3.6),
    max_frames=None,
    progress_interval=100,
    diagnostic_residual_prototypes=None,
):
    """Evaluate only after a supervision-free forward at every receiver time."""

    was_training = model.training
    model.eval()
    fields_per_horizon = 3 + len(miss_thresholds)
    # Tracking/forecast/lifecycle raw sums follow the horizon fields.
    age_bucket_names = ("age_1", "age_2_3", "age_4_plus")
    diagnostic_names = (
        "online_cv",
        "history_ols_cv",
        "history_median_cv",
        "history_ols_line",
        "history_median_line",
        "stationary",
        "history_multiscale_cv",
        "history_lateral3_cv",
        "history_lateral6_cv",
        "history_lateral9_cv",
        "history_residual_proto_cv",
    )
    support_bucket_names = ("receiver_only", "partner_only", "shared")
    diagnostic_fields_per_horizon = len(diagnostic_names) * 2 + 1
    support_fields = 3 * len(support_bucket_names) + 3
    statistics = torch.zeros(
        len(horizons) * fields_per_horizon
        + 22
        + 3 * len(age_bucket_names)
        + diagnostic_fields_per_horizon * len(horizons)
        + support_fields,
        dtype=torch.float64,
        device=device,
    )
    base = len(horizons) * fields_per_horizon
    age_base = base + 22
    diagnostic_base = age_base + 3 * len(age_bucket_names)
    support_base = (
        diagnostic_base + diagnostic_fields_per_horizon * len(horizons)
    )
    previous_key = None
    state = None
    lifecycle = None

    def fresh_lifecycle():
        return {
            "first_evidence": {},
            "first_ego": {},
            "last_evidence": {},
            "first_forecast": {},
            "forecast_frames": defaultdict(int),
            "previous_runtime": {},
            "was_matched": {},
            "ever_matched": set(),
        }

    def finalize(record):
        if record is None:
            return
        for actor_id, first_evidence in record["first_evidence"].items():
            statistics[base + 20] += 1
            if actor_id not in record["first_forecast"]:
                continue
            first_forecast = record["first_forecast"][actor_id]
            statistics[base + 21] += 1
            statistics[base + 12] += max(first_forecast - first_evidence, 0) / 10.0
            statistics[base + 13] += 1
            if actor_id in record["first_ego"]:
                lead = (record["first_ego"][actor_id] - first_forecast) / 10.0
                statistics[base + 14] += lead
                statistics[base + 15] += 1
                statistics[base + 16] += float(lead > 0.0)
            else:
                statistics[base + 17] += 1
            span = max(record["last_evidence"][actor_id] - first_forecast + 1, 1)
            statistics[base + 18] += min(
                record["forecast_frames"][actor_id] / span, 1.0
            )
            statistics[base + 19] += 1

    with torch.no_grad():
        for sequence_index, batch in enumerate(loader):
            if max_frames is not None and sequence_index >= max_frames:
                break
            if progress_interval and sequence_index % progress_interval == 0:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                print(
                    "CoLIF strict-online test rank=%d frame=%d/%d"
                    % (rank, sequence_index, len(loader)),
                    flush=True,
                )
            scene_receiver_key = batch["scene_receiver_keys"][0]
            if scene_receiver_key != previous_key:
                finalize(lifecycle)
                state = None
                lifecycle = fresh_lifecycle()
                previous_key = scene_receiver_key

            runtime = train_utils.to_device(batch["runtime_frames"][0], device)
            targets = train_utils.to_device(batch["target_frames"][0], device)
            outputs, state, _ = model(runtime, state)
            frame_number = int(runtime["receiver_frame_idx"][0])
            target_current = torch.where(targets["current_mask"][0])[0]
            for target_index in target_current.tolist():
                actor_id = int(targets["actor_ids"][0, target_index])
                lifecycle["first_evidence"].setdefault(actor_id, frame_number)
                lifecycle["last_evidence"][actor_id] = frame_number
                if int(targets["evidence_source_bits"][0, target_index]) & 1:
                    lifecycle["first_ego"].setdefault(actor_id, frame_number)

            tracking_queries = torch.where(outputs["active_mask"][0])[0]
            tracking_matches = _hungarian_matches(
                outputs["track_boxes"][0, :, :2],
                targets["current_boxes"][0, :, :2],
                tracking_queries,
                target_current,
                match_distance,
            )
            statistics[base] += target_current.numel()
            statistics[base + 1] += len(tracking_matches)
            statistics[base + 2] += max(
                tracking_queries.numel() - len(tracking_matches), 0
            )
            statistics[base + 3] += max(
                target_current.numel() - len(tracking_matches), 0
            )
            matched_actor_ids = set()
            for query_index, target_index in tracking_matches:
                actor_id = int(targets["actor_ids"][0, target_index])
                runtime_id = int(outputs["runtime_ids"][0, query_index])
                matched_actor_ids.add(actor_id)
                previous_runtime = lifecycle["previous_runtime"].get(actor_id)
                if previous_runtime is not None and previous_runtime != runtime_id:
                    statistics[base + 4] += 1
                if actor_id in lifecycle["ever_matched"] and not lifecycle["was_matched"].get(actor_id, False):
                    statistics[base + 6] += 1
                lifecycle["previous_runtime"][actor_id] = runtime_id
                lifecycle["ever_matched"].add(actor_id)
                statistics[base + 5] += torch.norm(
                    outputs["track_boxes"][0, query_index, :2].float()
                    - targets["current_boxes"][0, target_index, :2].float()
                ).double()
                mask_full = targets["future_mask"][0, target_index]
                target_future = targets["future_xy"][0, target_index].float()
                has_runtime_history = (
                    hasattr(state, "history_mask")
                    and hasattr(state, "forecast_history_box")
                )
                valid_history = (
                    torch.where(state.history_mask[0, query_index])[0]
                    if has_runtime_history
                    else torch.zeros(
                        0, dtype=torch.long, device=target_future.device
                    )
                )
                if valid_history.numel() >= 2:
                    previous_index = int(valid_history[-2])
                    elapsed_s = max(
                        (state.history_mask.shape[-1] - 1 - previous_index)
                        * 0.1,
                        0.1,
                    )
                    native_velocity = (
                        state.forecast_history_box[0, query_index, -1, :2]
                        - state.forecast_history_box[
                            0, query_index, previous_index, :2
                        ]
                    ) / elapsed_s
                else:
                    native_velocity = target_future.new_zeros(2)
                if has_runtime_history:
                    history_ols_velocity, history_median_velocity = (
                        _history_velocity_estimates(
                            state.forecast_history_box[
                                0, query_index, :, :2
                            ].float(),
                            state.history_mask[0, query_index],
                        )
                    )
                else:
                    history_ols_velocity = target_future.new_zeros(2)
                    history_median_velocity = target_future.new_zeros(2)
                future_times = torch.arange(
                    1,
                    target_future.shape[0] + 1,
                    device=target_future.device,
                    dtype=target_future.dtype,
                ) * 0.1
                if has_runtime_history:
                    history_ols_line, history_median_line = (
                        _history_linear_predictions(
                            state.forecast_history_box[
                                0, query_index, :, :2
                            ].float(),
                            state.history_mask[0, query_index],
                            future_times,
                        )
                    )
                else:
                    history_ols_line = target_future.new_zeros(
                        target_future.shape[0], 2
                    )
                    history_median_line = torch.zeros_like(history_ols_line)
                diagnostic_predictions = (
                    future_times[:, None] * native_velocity[None],
                    future_times[:, None] * history_ols_velocity[None],
                    future_times[:, None] * history_median_velocity[None],
                    history_ols_line,
                    history_median_line,
                    torch.zeros(
                        target_future.shape[0],
                        2,
                        device=target_future.device,
                        dtype=target_future.dtype,
                    ),
                    torch.stack(
                        [
                            future_times[:, None]
                            * history_median_velocity[None]
                            * scale
                            for scale in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5)
                        ],
                        dim=0,
                    ),
                    _kinematic_lateral_mode_bank(
                        history_median_velocity,
                        outputs["track_boxes"][0, query_index, 6].float(),
                        future_times,
                        3.0,
                    ),
                    _kinematic_lateral_mode_bank(
                        history_median_velocity,
                        outputs["track_boxes"][0, query_index, 6].float(),
                        future_times,
                        6.0,
                    ),
                    _kinematic_lateral_mode_bank(
                        history_median_velocity,
                        outputs["track_boxes"][0, query_index, 6].float(),
                        future_times,
                        9.0,
                    ),
                    (
                        _residual_prototype_mode_bank(
                            history_median_velocity,
                            outputs["track_boxes"][0, query_index, 6].float(),
                            future_times,
                            diagnostic_residual_prototypes,
                        )
                        if diagnostic_residual_prototypes is not None
                        else torch.stack(
                            [
                                future_times[:, None]
                                * history_median_velocity[None]
                            ]
                            * 6,
                            dim=0,
                        )
                    ),
                )
                for horizon_index, horizon in enumerate(horizons):
                    mask = mask_full[:horizon]
                    if not mask.any():
                        continue
                    last_valid = int(torch.where(mask)[0][-1])
                    diagnostic_offset = (
                        diagnostic_base
                        + horizon_index * diagnostic_fields_per_horizon
                    )
                    for prediction_index, baseline_prediction in enumerate(
                        diagnostic_predictions
                    ):
                        prediction_modes = (
                            baseline_prediction[:, :horizon]
                            if baseline_prediction.ndim == 3
                            else baseline_prediction[:horizon].unsqueeze(0)
                        )
                        baseline_errors = _absolute_forecast_error(
                            prediction_modes,
                            outputs["track_boxes"][
                                0, query_index, :2
                            ].float(),
                            target_future[:horizon],
                            targets["current_boxes"][
                                0, target_index, :2
                            ].float(),
                        )
                        mode_ade = (
                            baseline_errors
                            * mask[None].to(baseline_errors.dtype)
                        ).sum(dim=-1) / mask.sum()
                        statistics[
                            diagnostic_offset + prediction_index * 2
                        ] += (
                            mode_ade.min()
                        )
                        statistics[
                            diagnostic_offset + prediction_index * 2 + 1
                        ] += baseline_errors[:, last_valid].min()
                    statistics[
                        diagnostic_offset + len(diagnostic_names) * 2
                    ] += 1
            for target_index in target_current.tolist():
                actor_id = int(targets["actor_ids"][0, target_index])
                lifecycle["was_matched"][actor_id] = actor_id in matched_actor_ids
            statistics[base + 7] += 1

            forecast_queries = torch.where(
                outputs["active_mask"][0] & outputs["forecast_valid_mask"][0]
            )[0]
            forecast_target_mask = (
                targets["current_mask"][0]
                & targets["future_mask"][0].any(dim=-1)
            )
            forecast_targets = torch.where(forecast_target_mask)[0]
            forecast_matches = _hungarian_matches(
                outputs["track_boxes"][0, :, :2],
                targets["current_boxes"][0, :, :2],
                forecast_queries,
                forecast_targets,
                match_distance,
            )
            statistics[base + 8] += forecast_targets.numel()
            statistics[base + 9] += len(forecast_matches)
            statistics[base + 10] += forecast_queries.numel()
            statistics[base + 11] += max(
                forecast_queries.numel() - len(forecast_matches), 0
            )

            for query_index, target_index in forecast_matches:
                actor_id = int(targets["actor_ids"][0, target_index])
                lifecycle["first_forecast"].setdefault(actor_id, frame_number)
                lifecycle["forecast_frames"][actor_id] += 1
                prediction = outputs["pred_trajs"][0, query_index, :, :, :2].float()
                target = targets["future_xy"][0, target_index].float()
                mask_full = targets["future_mask"][0, target_index]
                for horizon_index, horizon in enumerate(horizons):
                    mask = mask_full[:horizon]
                    if not mask.any():
                        continue
                    errors = _absolute_forecast_error(
                        prediction[:, :horizon],
                        outputs["track_boxes"][0, query_index, :2].float(),
                        target[:horizon],
                        targets["current_boxes"][0, target_index, :2].float(),
                    )
                    mode_ade = (
                        errors * mask[None].to(errors.dtype)
                    ).sum(-1) / mask.sum()
                    last_valid = int(torch.where(mask)[0][-1])
                    min_ade = mode_ade.min()
                    min_fde = errors[:, last_valid].min()
                    offset = horizon_index * fields_per_horizon
                    statistics[offset] += min_ade.double()
                    statistics[offset + 1] += min_fde.double()
                    statistics[offset + 2] += 1
                    for threshold_index, threshold in enumerate(miss_thresholds):
                        statistics[offset + 3 + threshold_index] += (
                            min_fde > threshold
                        ).double()
                age_tensor = outputs.get("query_age")
                age = int(age_tensor[0, query_index]) if age_tensor is not None else 1
                age_bucket = 0 if age <= 1 else (1 if age <= 3 else 2)
                longest_horizon = max(horizons)
                age_mask = mask_full[:longest_horizon]
                if age_mask.any():
                    age_errors = _absolute_forecast_error(
                        prediction[:, :longest_horizon],
                        outputs["track_boxes"][0, query_index, :2].float(),
                        target[:longest_horizon],
                        targets["current_boxes"][0, target_index, :2].float(),
                    )
                    age_mode_ade = (
                        age_errors * age_mask[None].to(age_errors.dtype)
                    ).sum(-1) / age_mask.sum()
                    age_last_valid = int(torch.where(age_mask)[0][-1])
                    age_offset = age_base + age_bucket * 3
                    statistics[age_offset] += age_mode_ade.min().double()
                    statistics[age_offset + 1] += age_errors[
                        :, age_last_valid
                    ].min().double()
                    statistics[age_offset + 2] += 1
                    evidence_bits = int(
                        targets["evidence_source_bits"][0, target_index]
                    )
                    receiver_supported = bool(evidence_bits & 1)
                    partner_supported = bool(evidence_bits & ~1)
                    if receiver_supported and partner_supported:
                        support_bucket = 2
                    elif receiver_supported:
                        support_bucket = 0
                    elif partner_supported:
                        support_bucket = 1
                    else:
                        support_bucket = None
                    if support_bucket is not None:
                        support_offset = support_base + support_bucket * 3
                        statistics[support_offset] += age_mode_ade.min().double()
                        statistics[support_offset + 1] += age_errors[
                            :, age_last_valid
                        ].min().double()
                        statistics[support_offset + 2] += 1
                        if support_bucket == 1 and 2 <= age <= 3:
                            short_offset = support_base + 9
                            statistics[short_offset] += age_mode_ade.min().double()
                            statistics[short_offset + 1] += age_errors[
                                :, age_last_valid
                            ].min().double()
                            statistics[short_offset + 2] += 1
        finalize(lifecycle)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
    metrics = {}
    for horizon_index, horizon in enumerate(horizons):
        offset = horizon_index * fields_per_horizon
        raw_count = statistics[offset + 2]
        count = raw_count.clamp_min(1.0)
        seconds = horizon / 10.0
        metrics["minADE_%.0fs" % seconds] = (
            float(statistics[offset] / count) if raw_count > 0 else None
        )
        metrics["minFDE_%.0fs" % seconds] = (
            float(statistics[offset + 1] / count) if raw_count > 0 else None
        )
        metrics["actor_count_%.0fs" % seconds] = int(raw_count)
        for threshold_index, threshold in enumerate(miss_thresholds):
            metrics["MR_%.1fm_%.0fs" % (threshold, seconds)] = (
                float(statistics[offset + 3 + threshold_index] / count)
                if raw_count > 0 else None
            )

    gt = statistics[base]
    tp, fp, fn = statistics[base + 1 : base + 4]
    ids = statistics[base + 4]
    forecast_targets = statistics[base + 8]
    forecast_active = statistics[base + 10]
    metrics.update(
        {
            "tracking_gt_actor_frames": int(gt),
            "tracking_true_positive_frames": int(tp),
            "tracking_false_positive_frames": int(fp),
            "tracking_false_negative_frames": int(fn),
            "tracking_recall": float(tp / gt.clamp_min(1.0)),
            "tracking_precision": float(tp / (tp + fp).clamp_min(1.0)),
            "MOTA": float(1.0 - (fp + fn + ids) / gt.clamp_min(1.0)),
            "MOTP_center_m": float(statistics[base + 5] / tp.clamp_min(1.0)),
            "identity_switches": int(ids),
            "fragments": int(statistics[base + 6]),
            "target_actor_frames": int(forecast_targets),
            "matched_actor_frames": int(statistics[base + 9]),
            "forecast_coverage": float(
                statistics[base + 9] / forecast_targets.clamp_min(1.0)
            ),
            "active_query_frames": int(forecast_active),
            "unmatched_active_query_frames": int(statistics[base + 11]),
            "unmatched_active_rate": float(
                statistics[base + 11] / forecast_active.clamp_min(1.0)
            ),
            "frame_count": int(statistics[base + 7]),
            "time_to_first_forecast_s": (
                float(statistics[base + 12] / statistics[base + 13])
                if statistics[base + 13] > 0 else None
            ),
            "anticipation_lead_vs_ego_s": (
                float(statistics[base + 14] / statistics[base + 15])
                if statistics[base + 15] > 0 else None
            ),
            "positive_anticipation_rate": (
                float(statistics[base + 16] / statistics[base + 15])
                if statistics[base + 15] > 0 else None
            ),
            "forecast_continuity": (
                float(statistics[base + 18] / statistics[base + 19])
                if statistics[base + 19] > 0 else None
            ),
            "evidence_actor_count": int(statistics[base + 20]),
            "forecasted_actor_count": int(statistics[base + 21]),
            "ego_never_observed_forecasted_actor_count": int(statistics[base + 17]),
            "strict_online": True,
            "runtime_uses_ground_truth": False,
        }
    )
    for bucket_index, bucket_name in enumerate(age_bucket_names):
        offset = age_base + bucket_index * 3
        raw_count = statistics[offset + 2]
        denominator = raw_count.clamp_min(1.0)
        metrics["%s_actor_count_5s" % bucket_name] = int(raw_count)
        metrics["%s_minADE_5s" % bucket_name] = (
            float(statistics[offset] / denominator) if raw_count > 0 else None
        )
        metrics["%s_minFDE_5s" % bucket_name] = (
            float(statistics[offset + 1] / denominator) if raw_count > 0 else None
        )
    for horizon_index, horizon in enumerate(horizons):
        offset = diagnostic_base + horizon_index * diagnostic_fields_per_horizon
        raw_count = statistics[offset + len(diagnostic_names) * 2]
        denominator = raw_count.clamp_min(1.0)
        seconds = horizon / 10.0
        metrics["diagnostic_tracking_actor_count_%.0fs" % seconds] = int(
            raw_count
        )
        for prediction_index, name in enumerate(diagnostic_names):
            metrics["diagnostic_%s_ADE_%.0fs" % (name, seconds)] = (
                float(
                    statistics[offset + prediction_index * 2] / denominator
                )
                if raw_count > 0
                else None
            )
            metrics["diagnostic_%s_FDE_%.0fs" % (name, seconds)] = (
                float(
                    statistics[offset + prediction_index * 2 + 1]
                    / denominator
                )
                if raw_count > 0
                else None
            )
    for bucket_index, bucket_name in enumerate(support_bucket_names):
        offset = support_base + bucket_index * 3
        raw_count = statistics[offset + 2]
        denominator = raw_count.clamp_min(1.0)
        metrics["%s_actor_count_5s" % bucket_name] = int(raw_count)
        metrics["%s_minADE_5s" % bucket_name] = (
            float(statistics[offset] / denominator) if raw_count > 0 else None
        )
        metrics["%s_minFDE_5s" % bucket_name] = (
            float(statistics[offset + 1] / denominator)
            if raw_count > 0 else None
        )
    short_offset = support_base + 9
    short_count = statistics[short_offset + 2]
    short_denominator = short_count.clamp_min(1.0)
    metrics["partner_only_age_2_3_actor_count_5s"] = int(short_count)
    metrics["partner_only_age_2_3_minADE_5s"] = (
        float(statistics[short_offset] / short_denominator)
        if short_count > 0 else None
    )
    metrics["partner_only_age_2_3_minFDE_5s"] = (
        float(statistics[short_offset + 1] / short_denominator)
        if short_count > 0 else None
    )
    if was_training:
        model.train()
    return metrics
