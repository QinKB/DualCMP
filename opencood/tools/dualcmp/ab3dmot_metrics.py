"""Adapter from strict-online DualCMP tracks to the repository evaluator.

Adapted from ``CoLIF/opencood/tools/colif/ab3dmot_metrics.py``; DualCMP keeps
the same repository evaluator and changes only the runtime-track I/O boundary.

Metric definitions and accumulation remain in
``AB3Dmot/scripts/KITTI/evaluate.py``.  This module only converts DualCMP's
online outputs to that evaluator's KITTI text interface, maps both V2V4Real
receivers to independent sequences, and reads the evaluator's summary.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import Dict, List, Optional, Union

import torch


TRACKING_METRIC_PROTOCOL = "v2v4real_ab3dmot_3d_iou_0.25_41_recall"
TRACKING_IOU_THRESHOLD = 0.25
TRACKING_CSV_FIELDS = (
    "tracking_metric_protocol",
    "tracking_iou_threshold",
    "sAMOTA",
    "AMOTA",
    "AMOTP",
    "MOTA",
    "MOTP",
    "tracking_recall",
    "tracking_precision",
    "identity_switches",
    "fragments",
)


class _QuietMail:
    @staticmethod
    def msg(_message):
        return None


def _empty_aware_evaluation_class(evaluator):
    """Adapt the repository loader gate while retaining all metric code."""

    class EmptyAwareTrackingEvaluation(evaluator.trackingEvaluation):
        def loadTracker(self):
            try:
                loaded = self._loadData(
                    self.t_path,
                    cls=self.cls,
                    loading_groundtruth=False,
                )
            except IOError:
                return False
            if loaded:
                return True
            # The repository loader already built one empty frame sequence per
            # seqmap entry, but its original gate rejects zero trajectories.
            # Keeping that representation lets its own metric implementation
            # report the all-missed case instead of inventing a sentinel track.
            return (
                getattr(self, "n_tr_trajectories", None) == 0
                and len(getattr(self, "tracker", ())) == self.n_sequences
            )

    return EmptyAwareTrackingEvaluation


def _nonplotting_stat_class(evaluator):
    """Keep repository statistics while suppressing disposable PDF plots."""

    class NonPlottingStat(evaluator.stat):
        def plot(self):
            return None

    return NonPlottingStat


def tracking_frame_prediction(
    frame_index: int,
    runtime_ids: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
) -> dict:
    """Detach one online frame without performing metric computation."""

    if runtime_ids.ndim != 1 or boxes.ndim != 2 or scores.ndim != 1:
        raise ValueError("tracking outputs must be [N], [N,7], and [N]")
    if boxes.shape[-1] != 7:
        raise ValueError("tracking boxes must use [x,y,z,h,w,l,yaw]")
    if not (runtime_ids.shape[0] == boxes.shape[0] == scores.shape[0]):
        raise ValueError("tracking IDs, boxes, and scores disagree")
    return {
        "frame_index": int(frame_index),
        "runtime_ids": runtime_ids.detach().cpu().long().tolist(),
        "boxes": boxes.detach().cpu().float().tolist(),
        "scores": scores.detach().cpu().float().tolist(),
    }


def _load_repository_evaluator(repository_root: Path):
    evaluator_path = repository_root / "AB3Dmot/scripts/KITTI/evaluate.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError("AB3DMOT evaluator not found: %s" % evaluator_path)
    module_name = "_dualcmp_repository_ab3dmot_evaluate"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    search_paths = [
        str(evaluator_path.parent),
        str(repository_root / "AB3Dmot"),
        str(repository_root / "AB3Dmot/Xinshuo_PyToolbox"),
    ]
    original_path = list(sys.path)
    try:
        sys.path[:0] = [path for path in search_paths if path not in sys.path]
        spec = importlib.util.spec_from_file_location(module_name, evaluator_path)
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load AB3DMOT evaluator: %s" % evaluator_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path[:] = original_path
    return module


def _write_prediction_file(path: Path, frames: List[Dict], frame_map: Dict) -> int:
    detection_count = 0
    previous_frame = -1
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            raw_frame_index = int(frame["frame_index"])
            if raw_frame_index not in frame_map:
                raise ValueError("prediction frame is outside the evaluation window")
            frame_index = frame_map[raw_frame_index]
            if frame_index <= previous_frame:
                raise ValueError(
                    "tracking frames must be strictly increasing: %d after %d"
                    % (frame_index, previous_frame)
                )
            previous_frame = frame_index
            ids = frame["runtime_ids"]
            boxes = frame["boxes"]
            scores = frame["scores"]
            if not (len(ids) == len(boxes) == len(scores)):
                raise ValueError("serialized tracking IDs, boxes, and scores disagree")
            if len(set(int(value) for value in ids)) != len(ids):
                raise ValueError("runtime track IDs must be unique within a frame")
            for runtime_id, box, score in zip(ids, boxes, scores):
                if len(box) != 7 or not all(math.isfinite(float(v)) for v in box):
                    raise ValueError("tracking box contains invalid values")
                if not math.isfinite(float(score)):
                    raise ValueError("tracking score is not finite")
                x, y, z, height, width, length, yaw = map(float, box)
                # The repository's V2V4Real conversion writes lidar
                # [x,y,z] into KITTI camera fields [x,z,y].
                handle.write(
                    "%d %d Car 0 0 0 0 0 0 0 %.9g %.9g %.9g "
                    "%.9g %.9g %.9g %.9g %.9g\n"
                    % (
                        frame_index,
                        int(runtime_id),
                        height,
                        width,
                        length,
                        x,
                        z,
                        y,
                        yaw,
                        float(score),
                    )
                )
                detection_count += 1
    return detection_count


def _write_window_ground_truth(source: Path, destination: Path, frame_map: Dict) -> None:
    """Copy exactly the evaluated raw GT frames, renumbered from zero.

    The repository evaluator uses the seqmap bounds as an implicit frame
    contract.  A sparse raw V2V4Real window must therefore be rewritten with
    the same contiguous frame IDs as its predictions, rather than relying on
    the source file's global frame numbering.
    """
    lines = []
    for line in source.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        raw_frame = int(fields[0])
        if raw_frame in frame_map:
            fields[0] = str(frame_map[raw_frame])
            lines.append(" ".join(fields))
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _parse_summary(summary_path: Path) -> dict:
    text = summary_path.read_text(encoding="utf-8")
    average = re.search(
        r"sAMOTA\s+AMOTA\s+AMOTP\s*\n\s*"
        r"([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)",
        text,
    )
    if average is None:
        raise RuntimeError("AB3DMOT summary is missing sAMOTA/AMOTA/AMOTP")

    def named(label: str) -> float:
        matches = re.findall(re.escape(label) + r"\s+([-+0-9.eE]+)", text)
        if not matches:
            raise RuntimeError("AB3DMOT summary is missing %s" % label)
        return float(matches[-1])

    return {
        "tracking_metric_protocol": TRACKING_METRIC_PROTOCOL,
        "tracking_iou_threshold": TRACKING_IOU_THRESHOLD,
        "sAMOTA": float(average.group(1)),
        "AMOTA": float(average.group(2)),
        "AMOTP": float(average.group(3)),
        "MOTA": named("Multiple Object Tracking Accuracy (MOTA)"),
        "MOTP": named("Multiple Object Tracking Precision (MOTP)"),
        "tracking_recall": named("Recall"),
        "tracking_precision": named("Precision"),
        "identity_switches": int(named("ID-switches")),
        "fragments": int(named("Fragmentations")),
    }


def compute_ab3dmot_metrics(
    timeline_predictions: dict,
    *,
    repository_root: Optional[Union[str, Path]] = None,
    ground_truth_root: Optional[Union[str, Path]] = None,
) -> dict:
    """Run the existing V2V4Real AB3DMOT evaluator and return its summary."""

    repository_root = Path(repository_root or Path(__file__).resolve().parents[3])
    ground_truth_root = Path(
        ground_truth_root
        or repository_root / "AB3Dmot/scripts/KITTI/v2v4real_label/test"
    )
    evaluator = _load_repository_evaluator(repository_root)
    if not timeline_predictions:
        raise ValueError("AB3DMOT evaluation requires at least one timeline")

    with tempfile.TemporaryDirectory(prefix="dualcmp-ab3dmot-") as temp_name:
        temp_root = Path(temp_name)
        project_root = temp_root / "project"
        seqmap_dir = project_root / "AB3Dmot/scripts/KITTI"
        prediction_root = temp_root / "prediction/data_0"
        gt_root = temp_root / "ground_truth"
        work_root = temp_root / "work"
        result_root = work_root / "results/KITTI/dualcmp"
        seqmap_dir.mkdir(parents=True)
        prediction_root.mkdir(parents=True)
        gt_root.mkdir(parents=True)
        result_root.mkdir(parents=True)

        seqmap_lines = []
        for sequence_index, (timeline, frames) in enumerate(
            sorted(timeline_predictions.items(), key=lambda item: str(item[0]))
        ):
            if not isinstance(timeline, (tuple, list)) or len(timeline) != 2:
                raise ValueError("timeline key must be (scene_id, receiver_id)")
            scene_id, receiver_id = str(timeline[0]), str(timeline[1])
            source_gt = ground_truth_root / (scene_id + "-" + receiver_id + ".txt")
            if not source_gt.is_file():
                raise FileNotFoundError(
                    "AB3DMOT V2V4Real ground truth not found: %s" % source_gt
                )
            if not frames:
                raise ValueError(
                    "tracking timeline has no evaluated frames: %s" % (timeline,)
                )
            alias = "timeline_%04d" % sequence_index
            raw_frames = [int(frame["frame_index"]) for frame in frames]
            if len(set(raw_frames)) != len(raw_frames):
                raise ValueError("evaluation window has duplicate receiver frames: %s" % (timeline,))
            if raw_frames != sorted(raw_frames):
                raise ValueError("evaluation window frames must be strictly increasing: %s" % (timeline,))
            expected = list(range(raw_frames[0], raw_frames[-1] + 1))
            if raw_frames != expected:
                raise ValueError("evaluation window has missing receiver frames: %s" % (timeline,))
            frame_map = {raw_frame: local_frame for local_frame, raw_frame in enumerate(raw_frames)}
            _write_window_ground_truth(source_gt, gt_root / (alias + "-0.txt"), frame_map)
            _write_prediction_file(prediction_root / (alias + "-0.txt"), frames, frame_map)
            seqmap_lines.append("%s empty 000000 %06d\n" % (alias, len(raw_frames)))

        (seqmap_dir / evaluator.V2V4Real_SEQMAP_FILE_NAME).write_text(
            "".join(seqmap_lines), encoding="utf-8"
        )

        quiet_mail = _QuietMail()
        had_mail = hasattr(evaluator, "mail")
        old_mail = getattr(evaluator, "mail", None)
        original_evaluation_class = evaluator.trackingEvaluation
        original_stat_class = evaluator.stat
        old_cwd = Path.cwd()
        captured = io.StringIO()
        try:
            evaluator.mail = quiet_mail
            evaluator.trackingEvaluation = _empty_aware_evaluation_class(evaluator)
            evaluator.stat = _nonplotting_stat_class(evaluator)
            os.chdir(work_root)
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
                captured
            ):
                success = evaluator.evaluate(
                    "dualcmp",
                    quiet_mail,
                    1,
                    True,
                    False,
                    0.25,
                    str(gt_root),
                    str(prediction_root),
                    str(project_root),
                    "v2v4real",
                )
        finally:
            os.chdir(old_cwd)
            if had_mail:
                evaluator.mail = old_mail
            else:
                del evaluator.mail
            evaluator.trackingEvaluation = original_evaluation_class
            evaluator.stat = original_stat_class
        if not success:
            raise RuntimeError(
                "Repository AB3DMOT evaluator rejected tracker output: %s"
                % captured.getvalue()[-2000:]
            )
        summary_path = prediction_root.parent / "summary_car_average_eval3D.txt"
        if not summary_path.is_file():
            raise RuntimeError("Repository AB3DMOT evaluator did not write its summary")
        return _parse_summary(summary_path)


def distributed_ab3dmot_metrics(timeline_predictions: dict) -> dict:
    """Gather disjoint timelines, run AB3DMOT once, and broadcast its result."""

    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not distributed:
        return compute_ab3dmot_metrics(timeline_predictions)

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, timeline_predictions)
    status = None
    if rank == 0:
        try:
            merged = {}
            for rank_predictions in gathered:
                overlap = set(merged).intersection(rank_predictions)
                if overlap:
                    raise RuntimeError(
                        "tracking timelines were evaluated on multiple ranks: %s"
                        % sorted(overlap)
                    )
                merged.update(rank_predictions)
            status = {"metrics": compute_ab3dmot_metrics(merged), "error": None}
        except Exception:
            status = {"metrics": None, "error": traceback.format_exc()}
    payload = [status]
    torch.distributed.broadcast_object_list(payload, src=0)
    if payload[0]["error"] is not None:
        raise RuntimeError(
            "Repository AB3DMOT evaluation failed on rank 0:\n%s" % payload[0]["error"]
        )
    return payload[0]["metrics"]


__all__ = [
    "TRACKING_CSV_FIELDS",
    "TRACKING_IOU_THRESHOLD",
    "TRACKING_METRIC_PROTOCOL",
    "compute_ab3dmot_metrics",
    "distributed_ab3dmot_metrics",
    "tracking_frame_prediction",
]
