"""IoU-only evaluation experiment; production sources and training stay frozen."""
import hashlib
import inspect
import pickle
import sys
from pathlib import Path

import numpy as np

from opencood.tools.dualcmp import cmp_adapted_forecast_metrics as metrics
from opencood.tools.dualcmp import eval_forecast_mainstream_ddp as native

IOU05_PROTOCOL = metrics.PROTOCOL
IOU05_CONTRACT = metrics.CONTRACT
HISTORICAL_IOU05_PROTOCOL = "cmp_gtfree_adapter_iou05_v1"
HISTORICAL_IOU05_CONTRACT = dict(metrics.LEGACY_CONTRACT,
    protocol=HISTORICAL_IOU05_PROTOCOL,
    matching="pinned CMP score-descending greedy BEV IoU >= 0.5; no future/history eligibility filtering before matching",
    matching_iou_threshold=0.5,
    scope="IoU0.5 adapted CMP forecast metrics; not official CMP end-to-end evaluation")


def report_iou05(frames, cache_path=None):
    frames = metrics._validated_frames(frames)
    pairing_source_sha = hashlib.sha256(inspect.getsource(metrics._pair_frames).encode()).hexdigest()
    control = metrics.cmp_adapted_report(frames, iou_threshold=0.7)
    result = metrics.cmp_adapted_report(frames, iou_threshold=0.5)
    pairs05, _ = metrics._pair_frames(frames, iou_threshold=0.5)
    pairs07, _ = metrics._pair_frames(frames, iou_threshold=0.7)
    def identity(pair):
        frame, pred, gt = pair
        return tuple(frame["key"]) + (int(frame["gt_ids"][gt]), int(frame["pred_ids"][pred]))
    keys07 = set(identity(pair) for pair in pairs07)
    keys05 = set(identity(pair) for pair in pairs05)
    added = [pair for pair in pairs05 if identity(pair) not in keys07]
    added_metrics, added_count = metrics._official_metrics(added)
    result["iou07_control"] = control
    result["matching_population_delta"] = {
        "same_pair_count": len(keys05 & keys07),
        "added_iou05_pairs": len(keys05 - keys07),
        "not_retained_iou07_pairs": len(keys07 - keys05),
        "added_pair_metrics": dict(added_metrics, count=added_count),
        "scope": "same inference; pair identity includes frame, GT ID and prediction runtime ID"}
    independent = {}
    if pairs05:
        distances = [np.linalg.norm(np.asarray(frame["pred_future"][pred], dtype=np.float64)
                      - np.asarray(frame["gt_future"][gt], dtype=np.float64)[None], axis=-1)
                     for frame, pred, gt in pairs05]
        for seconds, prefix, endpoint in ((1, 10, 10), (3, 30, 30), (5, 49, 49)):
            independent["minADE_%ds" % seconds] = float(np.mean([d[:, :prefix].mean(-1).min() for d in distances]))
            independent["minFDE_%ds" % seconds] = float(np.mean([d[:, endpoint].min() for d in distances]))
        for threshold in (3.6, 3.0, 2.0, 1.8):
            independent["MR_%.1f" % threshold] = float(np.mean([d[:, 49].min() > threshold for d in distances]))
    difference = max([abs(result["model"][key] - value) for key, value in independent.items()] or [0.])
    if difference > 1e-5:
        raise RuntimeError("IoU0.5 official metric and independent arithmetic disagree")
    result["iou_only_adapter"] = {
        "changed_setting": {"matching_iou_threshold": {"control": 0.7, "requested": 0.5}},
        "pairing_source_sha256": pairing_source_sha,
        "driver_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independent_arithmetic": independent, "official_max_difference": difference,
        "production_sources_mutated_at_runtime": False}
    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("xb") as stream:
            pickle.dump(frames, stream, protocol=4)
        result["prediction_cache"] = {
            "path": str(cache_path.resolve()), "frames": len(frames),
            "sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
            "scope": "trusted local post-forward GT/prediction export; no GT enters model forward"}
    return result


def main():
    cache = Path(sys.argv[sys.argv.index("--output") + 1]).with_suffix(".frames.pkl")
    if cache.exists():
        raise FileExistsError(cache)
    def evaluate(model, hypes, device, root, labels, batch_size=1, workers=2,
                 full_test=False, partner_delay_ms=100.0, train_window=False):
        return native._evaluate_native_forecast_ddp(model, hypes, device, root, labels,
            batch_size, workers, full_test, partner_delay_ms, train_window,
            frame_exporter=metrics.forecast_frame,
            report_builder=lambda frames: report_iou05(frames, cache))
    native.evaluate_cmp_adapted_forecast_ddp = evaluate
    native.main()


if __name__ == "__main__":
    main()
