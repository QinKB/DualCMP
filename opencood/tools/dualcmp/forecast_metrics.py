"""CMP eval_one_epoch_custom minADE/minFDE/MR over complete trajectories."""

from __future__ import annotations
import numpy as np


def cmp_metrics(
    predictions,
    targets,
    future_mask,
    prediction_centers=None,
    target_centers=None,
):
    """CMP metrics after restoring prediction and target to one frame."""
    if prediction_centers is None:
        prediction_centers = [np.zeros(2, dtype=np.float32)] * len(predictions)
    if target_centers is None:
        target_centers = [np.zeros(2, dtype=np.float32)] * len(targets)
    sums = {
        key: 0.0
        for key in ("ade_1s", "ade_3s", "ade_5s", "fde_1s", "fde_3s", "fde_5s", "mr")
    }
    count = 0
    for prediction, target, mask, prediction_center, target_center in zip(
        predictions,
        targets,
        future_mask,
        prediction_centers,
        target_centers,
    ):
        valid = np.flatnonzero(mask)
        if not len(valid):
            continue
        final = int(valid[-1])
        # Copy CoLIF@6b8788e::_absolute_forecast_error and CMP's
        # generate_prediction_dicts: decoder output is track-relative, while
        # the dataset target is GT-current-relative.
        prediction_absolute = prediction[:, : final + 1, :2] + np.asarray(
            prediction_center
        )[None, None]
        target_absolute = target[: final + 1, :2] + np.asarray(target_center)[None]
        distances = np.linalg.norm(
            prediction_absolute - target_absolute[None], axis=-1
        )
        for name, index in (
            ("1s", min(10, final)),
            ("3s", min(30, final)),
            ("5s", final),
        ):
            sums["ade_" + name] += float(distances[:, :index].mean(axis=-1).min())
            sums["fde_" + name] += float(distances[:, index].min())
        sums["mr"] += float(distances[:, final].min() > 3.6)
        count += 1
    if not count:
        raise ValueError("CMP metrics require at least one complete target")
    return {
        "minADE_1s": sums["ade_1s"] / count,
        "minADE_3s": sums["ade_3s"] / count,
        "minADE_5s": sums["ade_5s"] / count,
        "minFDE_1s": sums["fde_1s"] / count,
        "minFDE_3s": sums["fde_3s"] / count,
        "minFDE_5s": sums["fde_5s"] / count,
        "MR_3.6": sums["mr"] / count,
        "count": count,
    }
