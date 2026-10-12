"""CMP eval_one_epoch_custom minADE/minFDE/MR over complete trajectories."""

from __future__ import annotations
import numpy as np

FORECAST_METRIC_KEYS = ("minADE_1s", "minADE_3s", "minADE_5s",
                        "minFDE_1s", "minFDE_3s", "minFDE_5s", "MR_3.6")


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
    if not (len(predictions) == len(targets) == len(future_mask)
            == len(prediction_centers) == len(target_centers)):
        raise ValueError("metric population lengths disagree")
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
        if not np.isfinite(distances).all() or final < 1:
            raise ValueError("CMP errors must be finite and cover at least two future frames")
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


def horizon_metrics(predictions, targets, future_mask, prediction_centers=None,
                    target_centers=None, frame_period_s=0.1):
    """Additive explicit-horizon metrics, separate from frozen CMP results.

    Future index 0 means t+dt: 1/3/5 seconds end at 9/29/49 for 10 Hz.
    ADE includes the endpoint. Only complete trajectories are accepted here.
    """
    if not np.isfinite(frame_period_s) or frame_period_s <= 0:
        raise ValueError("frame period must be finite and positive")
    if not (len(predictions) == len(targets) == len(future_mask)):
        raise ValueError("metric population lengths disagree")
    prediction_centers = prediction_centers if prediction_centers is not None else [np.zeros(2)] * len(predictions)
    target_centers = target_centers if target_centers is not None else [np.zeros(2)] * len(targets)
    if not (len(predictions) == len(prediction_centers) == len(target_centers)):
        raise ValueError("metric population lengths disagree")
    totals = {"min" + kind + "_" + str(seconds) + "s": 0.0
              for seconds in (1, 3, 5) for kind in ("ADE", "FDE")}
    misses = count = 0
    for prediction, target, mask, pred_center, target_center in zip(
            predictions, targets, future_mask, prediction_centers, target_centers):
        if not np.asarray(mask).all():
            continue
        distances = np.linalg.norm(np.asarray(prediction)[..., :2] + np.asarray(pred_center)[None, None]
                                   - np.asarray(target)[None, :, :2] - np.asarray(target_center)[None, None], axis=-1)
        if not np.isfinite(distances).all():
            raise ValueError("forecast errors must be finite")
        for seconds in (1, 3, 5):
            steps = int(round(seconds / frame_period_s))
            if distances.shape[-1] < steps or abs(steps * frame_period_s - seconds) > 1e-6:
                raise ValueError("trajectory does not cover the requested exact horizon")
            totals["minADE_%ds" % seconds] += float(distances[:, :steps].mean(-1).min())
            totals["minFDE_%ds" % seconds] += float(distances[:, steps - 1].min())
        misses += float(distances[:, int(round(5 / frame_period_s)) - 1].min() > 3.6)
        count += 1
    if not count:
        raise ValueError("horizon metrics require complete trajectories")
    return dict({key: value / count for key, value in totals.items()},
                **{"MR_3.6": misses / count, "count": count})
