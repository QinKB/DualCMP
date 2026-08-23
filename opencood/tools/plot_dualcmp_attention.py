"""Generate paper-ready DualCMP attention diagnostics from a trained checkpoint.

The script performs strict-online inference on complete V2V4Real
scene--receiver timelines.  It produces two figures without using model
targets during forward:

1. fused-BEV tracking evidence, per-CAV agent patches, attention weights, and
   multimodal trajectories for deterministically selected agents;
2. partner-attention statistics stratified by source support and relative
   agent--CAV geometry.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
import torch
import torch.distributed as dist

from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.tools import train_utils
from opencood.tools.colif.eval_utils import (
    _absolute_forecast_error,
    _forecast_metric_window,
    _hungarian_matches,
    build_colif_evaluation_loader,
)
from opencood.tools.colif.receiver_protocol import apply_receiver_override


SUPPORT_NAMES = ("Receiver only", "Partner only", "Shared")
SUPPORT_SLUGS = ("receiver_only", "partner_only", "shared")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hypes-yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--receiver-ids")
    parser.add_argument("--max-frames-per-rank", type=int)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--match-distance", type=float)
    parser.add_argument("--crop-radius-m", type=float, default=12.0)
    parser.add_argument("--example-min-age", type=int, default=4)
    parser.add_argument("--example-min-displacement-m", type=float, default=5.0)
    parser.add_argument("--candidate-reservoir-size", type=int, default=64)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20)
    return parser.parse_args()


def support_index(source_bits: int) -> int | None:
    """Map receiver/partner support bits to the three paper categories."""

    receiver = bool(int(source_bits) & 1)
    partner = bool(int(source_bits) & ~1)
    if receiver and partner:
        return 2
    if receiver:
        return 0
    if partner:
        return 1
    return None


def source_attention_mass(
    attention_weights: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum patch-token attention for each valid CAV.

    Parameters
    ----------
    attention_weights : torch.Tensor
        Tensor with shape ``[L,P]`` where P is the number of local patch
        tokens.  The model softmax is taken jointly over ``L*P``.
    source_mask : torch.Tensor
        Boolean tensor with shape ``[L]``.
    """

    if attention_weights.ndim != 2:
        raise ValueError("attention_weights must have shape [L,P]")
    if source_mask.shape != attention_weights.shape[:1]:
        raise ValueError("source_mask must have shape [L]")
    mass = attention_weights.sum(dim=-1)
    return torch.where(source_mask.bool(), mass, torch.zeros_like(mass))


def cluster_bootstrap_binned_mean(
    x: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    bin_edges: np.ndarray,
    repeats: int = 1000,
    seed: int = 20,
) -> dict:
    """Return bin means and scene-cluster bootstrap confidence intervals."""

    x = np.asarray(x, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    if x.shape != values.shape or x.shape != groups.shape:
        raise ValueError("x, values, and groups must have equal shapes")
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bin_edges must contain at least two edges")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    finite = np.isfinite(x) & np.isfinite(values)
    x, values, groups = x[finite], values[finite], groups[finite]
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    bin_count = bin_edges.size - 1

    def means_for_indices(indices: np.ndarray) -> np.ndarray:
        result = np.full(bin_count, np.nan, dtype=np.float64)
        sampled_x = x[indices]
        sampled_values = values[indices]
        for bin_index in range(bin_count):
            in_bin = (
                (sampled_x >= bin_edges[bin_index])
                & (sampled_x < bin_edges[bin_index + 1])
            )
            if in_bin.any():
                result[bin_index] = sampled_values[in_bin].mean()
        return result

    observed = means_for_indices(np.arange(x.size))
    counts = np.asarray(
        [
            np.count_nonzero((x >= bin_edges[index]) & (x < bin_edges[index + 1]))
            for index in range(bin_count)
        ],
        dtype=np.int64,
    )
    bootstrap = np.full((repeats, bin_count), np.nan, dtype=np.float64)
    group_indices = {group: np.where(groups == group)[0] for group in unique_groups}
    for repeat in range(repeats):
        sampled_groups = rng.choice(
            unique_groups, size=unique_groups.size, replace=True
        )
        indices = np.concatenate(
            [group_indices[group] for group in sampled_groups]
        )
        bootstrap[repeat] = means_for_indices(indices)
    lower = np.full(bin_count, np.nan, dtype=np.float64)
    upper = np.full(bin_count, np.nan, dtype=np.float64)
    for bin_index in range(bin_count):
        finite_bootstrap = bootstrap[:, bin_index]
        finite_bootstrap = finite_bootstrap[np.isfinite(finite_bootstrap)]
        if finite_bootstrap.size:
            lower[bin_index], upper[bin_index] = np.percentile(
                finite_bootstrap, (2.5, 97.5)
            )
    centers = np.full(bin_count, np.nan, dtype=np.float64)
    for bin_index in range(bin_count):
        in_bin = (
            (x >= bin_edges[bin_index])
            & (x < bin_edges[bin_index + 1])
        )
        if in_bin.any():
            centers[bin_index] = x[in_bin].mean()
    return {
        "edges": bin_edges,
        "centers": centers,
        "mean": observed,
        "lower": lower,
        "upper": upper,
        "count": counts,
    }


def _metric_crop(
    value: torch.Tensor,
    center_xy: torch.Tensor,
    radius_m: float,
    lidar_range,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    if value.ndim != 2:
        raise ValueError("value must have shape [H,W]")
    height, width = value.shape
    x_min, y_min = float(lidar_range[0]), float(lidar_range[1])
    x_max, y_max = float(lidar_range[3]), float(lidar_range[4])
    resolution_x = (x_max - x_min) / width
    resolution_y = (y_max - y_min) / height
    center_x, center_y = (float(item) for item in center_xy)
    left = max(0, int(math.floor((center_x - radius_m - x_min) / resolution_x)))
    right = min(width, int(math.ceil((center_x + radius_m - x_min) / resolution_x)))
    bottom = max(0, int(math.floor((center_y - radius_m - y_min) / resolution_y)))
    top = min(height, int(math.ceil((center_y + radius_m - y_min) / resolution_y)))
    if right <= left or top <= bottom:
        raise ValueError("actor crop lies outside the BEV feature map")
    extent = (
        x_min + left * resolution_x - center_x,
        x_min + right * resolution_x - center_x,
        y_min + bottom * resolution_y - center_y,
        y_min + top * resolution_y - center_y,
    )
    return value[bottom:top, left:right].detach().float().cpu().numpy(), extent


def _display_normalize(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.zeros_like(value)
    lower, upper = np.percentile(finite, (2.0, 98.0))
    if upper <= lower:
        return np.zeros_like(value)
    return np.clip((value - lower) / (upper - lower), 0.0, 1.0)


def _candidate_priority(key: tuple) -> int:
    """Return a stable pseudo-random priority independent of model error."""

    digest = hashlib.sha256(repr(tuple(key)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _capture_example(
    model,
    outputs: dict,
    state,
    aux: dict,
    runtime: dict,
    targets: dict,
    query_index: int,
    target_index: int,
    support: int,
    crop_radius_m: float,
    lidar_range,
) -> dict:
    track_box = outputs["track_boxes"][0, query_index].detach().float()
    track_xy = track_box[:2]
    objectness = outputs["auxiliary_psm"].sigmoid().amax(dim=1)[0]
    fused_crop, fused_extent = _metric_crop(
        objectness, track_xy, crop_radius_m, lidar_range
    )
    source_bev = aux["regrouped_source_bev"][0]
    source_magnitude = torch.linalg.vector_norm(source_bev.float(), dim=1)
    source_crops = []
    source_extents = []
    for source_index in range(source_magnitude.shape[0]):
        crop, extent = _metric_crop(
            source_magnitude[source_index],
            track_xy,
            crop_radius_m,
            lidar_range,
        )
        source_crops.append(_display_normalize(crop))
        source_extents.append(extent)
    attention = outputs["cooperative_attention_weights"][
        0, query_index
    ].detach().float()
    cav_mask = aux["cav_mask"][0].detach().bool()
    mass = source_attention_mass(attention, cav_mask)
    patch_offsets = (
        model.actor_cooperative_fusion.unit_patch_offsets.detach().float()
        * float(model.actor_cooperative_fusion.patch_radius_m)
    )
    target_current = targets["current_boxes"][0, target_index, :2].float()
    target_future = targets["future_xy"][0, target_index].float()
    target_mask = targets["future_mask"][0, target_index].bool()
    predicted = outputs["pred_trajs"][0, query_index, :, :, :2].float()
    probabilities = outputs["pred_scores"][0, query_index].float()
    history_mask = state.history_mask[0, query_index].bool()
    history = state.forecast_history_box[
        0, query_index, history_mask, :2
    ].float()
    origin = track_xy
    source_ids = runtime["source_ids"][0].detach().cpu().tolist()
    return {
        "key": (
            str(runtime["scene_id"][0]),
            int(runtime["receiver_id"][0]),
            int(runtime["receiver_frame_idx"][0]),
            int(targets["actor_ids"][0, target_index]),
        ),
        "support": int(support),
        "age": int(outputs["query_age"][0, query_index]),
        "point_count": int(targets["evidence_point_counts"][0, target_index]),
        "source_ids": source_ids,
        "fused_crop": _display_normalize(fused_crop),
        "fused_extent": fused_extent,
        "source_crops": source_crops,
        "source_extents": source_extents,
        "attention": attention.cpu().numpy(),
        "attention_mass": mass.cpu().numpy(),
        "patch_offsets": patch_offsets.cpu().numpy(),
        "track_box": track_box.cpu().numpy(),
        "target_current": (target_current - origin).cpu().numpy(),
        "history": (history - origin).cpu().numpy(),
        "predicted": predicted.cpu().numpy(),
        "probabilities": probabilities.cpu().numpy(),
        "target_future": (
            target_current[None] + target_future - origin[None]
        ).cpu().numpy(),
        "target_mask": target_mask.cpu().numpy(),
    }


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _draw_track_box(axis, track_box: np.ndarray, color: str = "#00BFC4") -> None:
    width = float(track_box[4])
    length = float(track_box[5])
    yaw = float(track_box[6])
    rectangle = Rectangle(
        (-length / 2.0, -width / 2.0),
        length,
        width,
        fill=False,
        linewidth=1.4,
        edgecolor=color,
    )
    rectangle.set_transform(
        Affine2D().rotate_around(0.0, 0.0, yaw) + axis.transData
    )
    axis.add_patch(rectangle)


def plot_attention_examples(examples: list[dict], output_stem: Path) -> None:
    if len(examples) != 2:
        raise ValueError("exactly two examples are required")
    _set_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7), constrained_layout=True)
    column_titles = (
        "Fused BEV / tracking",
        "Receiver CAV patch",
        "Partner CAV patch",
        "Six-mode forecast",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title)
    trajectory_colors = (
        "#0072B2",
        "#E69F00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#D55E00",
    )
    for row, example in enumerate(examples):
        fused_axis = axes[row, 0]
        fused_axis.imshow(
            example["fused_crop"],
            origin="lower",
            extent=example["fused_extent"],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        _draw_track_box(fused_axis, example["track_box"])
        fused_axis.scatter(
            *example["target_current"], marker="x", s=20, c="white", linewidths=1.0
        )
        for source_offset in range(2):
            axis = axes[row, 1 + source_offset]
            axis.imshow(
                example["source_crops"][source_offset],
                origin="lower",
                extent=example["source_extents"][source_offset],
                cmap="magma",
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
            weights = example["attention"][source_offset]
            offsets = example["patch_offsets"]
            axis.scatter(
                offsets[:, 0],
                offsets[:, 1],
                s=25.0 + 1400.0 * weights,
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
            )
            axis.scatter(0.0, 0.0, marker="+", s=24, c="#00BFC4", linewidths=1.1)
            source_id = example["source_ids"][source_offset]
            mass = example["attention_mass"][source_offset]
            axis.text(
                0.03,
                0.96,
                "CAV %s  mass=%.3f" % (source_id, mass),
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
            )
        forecast_axis = axes[row, 3]
        probability = example["probabilities"]
        scale = max(float(probability.max()), 1e-6)
        for mode_index, trajectory in enumerate(example["predicted"]):
            path = np.concatenate((np.zeros((1, 2)), trajectory), axis=0)
            forecast_axis.plot(
                path[:, 0],
                path[:, 1],
                color=trajectory_colors[mode_index % len(trajectory_colors)],
                alpha=0.25 + 0.75 * float(probability[mode_index]) / scale,
                linewidth=1.25,
            )
        valid_future = example["target_future"][example["target_mask"]]
        if valid_future.size:
            forecast_axis.plot(
                valid_future[:, 0],
                valid_future[:, 1],
                color="black",
                linestyle="--",
                linewidth=1.6,
                label="Ground truth",
            )
        history = example["history"]
        if history.size:
            forecast_axis.plot(
                history[:, 0],
                history[:, 1],
                color="#666666",
                marker="o",
                markersize=2.0,
                linewidth=1.0,
                label="History",
            )
        _draw_track_box(forecast_axis, example["track_box"])
        all_xy = [np.zeros((1, 2)), example["predicted"].reshape(-1, 2)]
        if valid_future.size:
            all_xy.append(valid_future)
        if history.size:
            all_xy.append(history)
        combined = np.concatenate(all_xy, axis=0)
        radius = max(12.0, float(np.nanmax(np.abs(combined))) + 2.0)
        forecast_axis.set_xlim(-radius, radius)
        forecast_axis.set_ylim(-radius, radius)
        scene, receiver, frame, actor = example["key"]
        axes[row, 0].set_ylabel(
            "%s\nreceiver=%d, frame=%d\nagent=%d, age=%d"
            % (SUPPORT_NAMES[example["support"]], receiver, frame, actor, example["age"])
        )
        for axis in axes[row]:
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("x (m)")
            axis.tick_params(length=2)
            axis.grid(False)
    handles, labels = axes[0, 3].get_legend_handles_labels()
    if handles:
        axes[0, 3].legend(handles, labels, loc="best", frameon=False)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_attention_analysis(
    records: dict[str, np.ndarray],
    output_stem: Path,
    repeats: int,
    seed: int,
) -> dict:
    _set_plot_style()
    support = records["support"].astype(np.int64)
    partner_mass = records["partner_mass"].astype(np.float64)
    relative_distance = records["receiver_distance_m"].astype(
        np.float64
    ) - records["partner_distance_m"].astype(np.float64)
    groups = records["scene_receiver"]
    finite_relative_distance = relative_distance[np.isfinite(relative_distance)]
    distance_edges = np.quantile(
        finite_relative_distance, np.linspace(0.0, 1.0, 6)
    )
    if np.unique(distance_edges).size != distance_edges.size:
        raise RuntimeError(
            "relative-distance quintile edges are not distinct"
        )
    distance_edges[0] -= 1e-6
    distance_edges[-1] += 1e-6
    mass_curve = cluster_bootstrap_binned_mean(
        relative_distance,
        partner_mass,
        groups,
        distance_edges,
        repeats,
        seed,
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.35), constrained_layout=True)
    colors = ("#0072B2", "#D55E00", "#009E73")
    values_by_support = [partner_mass[support == index] for index in range(3)]
    box = axes[0].boxplot(
        values_by_support,
        tick_labels=("Receiver\nonly", "Partner\nonly", "Shared"),
        whis=1.5,
        showfliers=False,
        patch_artist=True,
        widths=0.55,
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    for median in box["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)
    axes[0].set_ylabel("Partner attention mass")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("(a) Observation support")

    axis = axes[1]
    valid = np.isfinite(mass_curve["mean"]) & (mass_curve["count"] >= 100)
    axis.plot(
        mass_curve["centers"][valid],
        mass_curve["mean"][valid],
        color="#0072B2",
        marker="o",
        linewidth=1.5,
    )
    axis.fill_between(
        mass_curve["centers"][valid],
        mass_curve["lower"][valid],
        mass_curve["upper"][valid],
        color="#0072B2",
        alpha=0.18,
        linewidth=0.0,
    )
    axis.axvline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Receiver distance $-$ partner distance (m)")
    axis.set_ylabel("Partner attention mass")
    axis.set_title("(b) Agent--CAV geometry")
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axes[1].set_ylim(0.0, 1.0)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {"attention_mass_by_relative_distance": mass_curve}


def _collect_rank(
    model,
    loader,
    device,
    match_distance: float,
    max_frames: int | None,
    progress_interval: int,
    crop_radius_m: float,
    example_min_age: int,
    example_min_displacement_m: float,
    candidate_reservoir_size: int,
    lidar_range,
) -> dict:
    model.eval()
    previous_key = None
    state = None
    records = defaultdict(list)
    candidates = defaultdict(list)
    with torch.no_grad():
        for frame_index, batch in enumerate(loader):
            if max_frames is not None and frame_index >= max_frames:
                break
            if progress_interval and frame_index % progress_interval == 0:
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    "DualCMP figure collection rank=%d frame=%d/%d"
                    % (rank, frame_index, len(loader)),
                    flush=True,
                )
            scene_receiver_key = batch["scene_receiver_keys"][0]
            if scene_receiver_key != previous_key:
                state = None
                previous_key = scene_receiver_key
            runtime = train_utils.to_device(batch["runtime_frames"][0], device)
            targets = train_utils.to_device(batch["target_frames"][0], device)
            outputs, state, aux = model(runtime, state)
            forecast_queries = torch.where(
                outputs["active_mask"][0] & outputs["forecast_valid_mask"][0]
            )[0]
            forecast_targets = torch.where(
                targets["current_mask"][0]
                & targets["future_mask"][0].any(dim=-1)
            )[0]
            matches = _hungarian_matches(
                outputs["track_boxes"][0, :, :2],
                targets["current_boxes"][0, :, :2],
                forecast_queries,
                forecast_targets,
                match_distance,
            )
            for query_index, target_index in matches:
                bits = int(targets["evidence_source_bits"][0, target_index])
                support = support_index(bits)
                if support is None:
                    continue
                attention = outputs["cooperative_attention_weights"][0, query_index]
                cav_mask = aux["cav_mask"][0].bool()
                if attention.shape[0] < 2 or not bool(cav_mask[1]):
                    continue
                mass = source_attention_mass(attention.float(), cav_mask)
                mask_full = targets["future_mask"][0, target_index]
                window = _forecast_metric_window(
                    mask_full, horizon=50, max_horizon=50, protocol="online_masked"
                )
                if window is None:
                    continue
                mask, last_valid = window
                prediction = outputs["pred_trajs"][0, query_index, :, :50, :2].float()
                target = targets["future_xy"][0, target_index, :50].float()
                errors = _absolute_forecast_error(
                    prediction,
                    outputs["track_boxes"][0, query_index, :2].float(),
                    target,
                    targets["current_boxes"][0, target_index, :2].float(),
                )
                min_fde = float(errors[:, last_valid].min())
                mode_ade = (
                    errors * mask[None].to(errors.dtype)
                ).sum(dim=-1) / mask.sum()
                current_xy = targets["current_boxes"][0, target_index, :2].float()
                scene_id = str(runtime["scene_id"][0])
                receiver_id = int(runtime["receiver_id"][0])
                records["support"].append(support)
                records["partner_mass"].append(float(mass[1:].sum()))
                records["receiver_mass"].append(float(mass[0]))
                records["receiver_distance_m"].append(
                    float(torch.linalg.vector_norm(current_xy))
                )
                partner_origin = runtime["source_to_receiver"][0, 1, :2, 3].float()
                records["partner_distance_m"].append(
                    float(torch.linalg.vector_norm(current_xy - partner_origin))
                )
                records["minade_5s_m"].append(float(mode_ade.min()))
                records["minfde_5s_m"].append(min_fde)
                records["point_count"].append(
                    int(targets["evidence_point_counts"][0, target_index])
                )
                records["age"].append(int(outputs["query_age"][0, query_index]))
                records["scene_receiver"].append(
                    "%s|%d" % (scene_id, receiver_id)
                )
                records["frame_index"].append(
                    int(runtime["receiver_frame_idx"][0])
                )
                records["actor_id"].append(
                    int(targets["actor_ids"][0, target_index])
                )

                age = int(outputs["query_age"][0, query_index])
                full_future = bool(mask_full[:50].all())
                displacement = float(
                    torch.linalg.vector_norm(target[49] - target[0])
                ) if target.shape[0] >= 50 else 0.0
                candidate_key = (
                    scene_id,
                    receiver_id,
                    int(runtime["receiver_frame_idx"][0]),
                    int(targets["actor_ids"][0, target_index]),
                )
                priority = _candidate_priority(candidate_key)
                current_priorities = [
                    int(item["candidate_priority"])
                    for item in candidates[support]
                ]
                reservoir_accepts = (
                    len(current_priorities) < candidate_reservoir_size
                    or priority < max(current_priorities)
                )
                if (
                    support in (1, 2)
                    and age >= example_min_age
                    and full_future
                    and displacement >= example_min_displacement_m
                    and reservoir_accepts
                ):
                    example = _capture_example(
                        model,
                        outputs,
                        state,
                        aux,
                        runtime,
                        targets,
                        query_index,
                        target_index,
                        support,
                        crop_radius_m,
                        lidar_range,
                    )
                    example["candidate_priority"] = priority
                    example["minfde_5s_m"] = min_fde
                    candidates[support].append(example)
                    candidates[support].sort(
                        key=lambda item: int(item["candidate_priority"])
                    )
                    del candidates[support][candidate_reservoir_size:]
    arrays = {
        name: np.asarray(values)
        for name, values in records.items()
    }
    return {"records": arrays, "candidates": dict(candidates)}


def _combine_shards(shards: list[dict]) -> tuple[dict, list[dict]]:
    names = tuple(shards[0]["records"].keys())
    if any(tuple(shard["records"].keys()) != names for shard in shards):
        raise ValueError("rank record schemas disagree")
    records = {
        name: np.concatenate([shard["records"][name] for shard in shards])
        for name in names
    }
    selected = []
    for support in (1, 2):
        candidates = [
            example
            for shard in shards
            for example in shard["candidates"].get(support, [])
        ]
        if not candidates:
            raise RuntimeError(
                "No %s example satisfies the configured selection rule"
                % SUPPORT_NAMES[support]
            )
        support_fde = records["minfde_5s_m"][records["support"] == support]
        median_fde = float(np.median(support_fde))
        selected.append(
            min(
                candidates,
                key=lambda item: (
                    abs(float(item["minfde_5s_m"]) - median_fde),
                    item["key"],
                ),
            )
        )
    return records, selected


def _jsonable_curve(curve: dict) -> dict:
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in curve.items()
    }


def main() -> None:
    args = _arguments()
    if args.candidate_reservoir_size < 1:
        raise ValueError("--candidate-reservoir-size must be positive")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(
        "cuda:%d" % local_rank if torch.cuda.is_available() else "cpu"
    )
    requested_hypes = load_yaml(args.hypes_yaml)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    hypes = payload.get("config", requested_hypes)
    receiver_ids = apply_receiver_override(hypes, args.receiver_ids)
    dataset, loader = build_colif_evaluation_loader(
        hypes,
        split="test",
        rank=rank,
        world_size=world_size,
        workers=args.workers,
    )
    model = train_utils.create_model(hypes).to(device)
    result = model.load_state_dict(payload["model_state"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("checkpoint did not load strictly")
    match_distance = (
        float(args.match_distance)
        if args.match_distance is not None
        else float(
            hypes["loss"]["args"].get("matcher", {}).get(
                "max_center_distance", 12.0
            )
        )
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_path = output_dir / (".dualcmp_attention_rank%d.pt" % rank)
    shard = _collect_rank(
        model,
        loader,
        device,
        match_distance=match_distance,
        max_frames=args.max_frames_per_rank,
        progress_interval=args.progress_interval,
        crop_radius_m=float(args.crop_radius_m),
        example_min_age=int(args.example_min_age),
        example_min_displacement_m=float(args.example_min_displacement_m),
        candidate_reservoir_size=int(args.candidate_reservoir_size),
        lidar_range=hypes["model"]["args"]["lidar_range"],
    )
    torch.save(shard, shard_path)
    if distributed:
        dist.barrier()
    if rank == 0:
        shard_paths = [
            output_dir / (".dualcmp_attention_rank%d.pt" % index)
            for index in range(world_size)
        ]
        shards = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in shard_paths
        ]
        records, examples = _combine_shards(shards)
        plot_attention_examples(
            examples, output_dir / "dualcmp_attention_examples"
        )
        curve_summary = plot_attention_analysis(
            records,
            output_dir / "dualcmp_attention_analysis",
            repeats=int(args.bootstrap_repeats),
            seed=int(args.bootstrap_seed),
        )
        np.savez_compressed(output_dir / "dualcmp_attention_records.npz", **records)
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
        summary = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_global_step": payload.get("global_step"),
            "checkpoint_config_hash": payload.get("config_hash"),
            "git_commit": git_commit,
            "receiver_ids": list(receiver_ids),
            "world_size": world_size,
            "test_frames": len(dataset),
            "processed_frames": (
                min(len(dataset), args.max_frames_per_rank * world_size)
                if args.max_frames_per_rank is not None
                else len(dataset)
            ),
            "matched_forecast_records": int(records["support"].size),
            "support_counts": {
                SUPPORT_SLUGS[index]: int(np.count_nonzero(records["support"] == index))
                for index in range(3)
            },
            "selected_examples": [
                {
                    "scene": example["key"][0],
                    "receiver": example["key"][1],
                    "frame": example["key"][2],
                    "actor": example["key"][3],
                    "support": SUPPORT_SLUGS[example["support"]],
                    "age": example["age"],
                    "minfde_5s_m": example["minfde_5s_m"],
                    "candidate_priority": example["candidate_priority"],
                    "attention_mass": example["attention_mass"].tolist(),
                }
                for example in examples
            ],
            "bootstrap": {
                "repeats": int(args.bootstrap_repeats),
                "seed": int(args.bootstrap_seed),
                "cluster": "scene_receiver",
            },
            "example_selection": {
                "rule": "closest to support-category median minFDE within a stable hash reservoir",
                "candidate_reservoir_size_per_rank": int(
                    args.candidate_reservoir_size
                ),
                "minimum_age": int(args.example_min_age),
                "minimum_future_displacement_m": float(
                    args.example_min_displacement_m
                ),
                "requires_complete_5s_future": True,
            },
            "curves": {
                name: _jsonable_curve(value)
                for name, value in curve_summary.items()
            },
        }
        (output_dir / "dualcmp_attention_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in shard_paths:
            path.unlink()
        print(json.dumps(summary, sort_keys=True), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
