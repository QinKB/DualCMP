"""Narrow adapter for the bundled AB3DMOT Car association contract.

The bundled ``AB3Dmot/LICENSE`` permits academic/non-profit noncommercial
research only.  This module calls that implementation; it does not copy or
reimplement its GIoU formula.
"""
from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_ab3dmot_association():
    """Load the repository-local Box3D/data_association without global paths."""
    root = Path(__file__).resolve().parents[3] / "AB3Dmot"
    if not (root / "AB3DMOT_libs/matching.py").is_file():
        raise FileNotFoundError("bundled AB3DMOT association is missing: %s" % root)
    original_path = list(sys.path)
    try:
        for path in (str(root), str(root / "Xinshuo_PyToolbox")):
            if path not in sys.path:
                sys.path.insert(0, path)
        box = importlib.import_module("AB3DMOT_libs.box")
        matching = importlib.import_module("AB3DMOT_libs.matching")
    finally:
        sys.path[:] = original_path
    return box.Box3D, matching.data_association


def _box3d_from_lidar(boxes):
    """Use online._prediction_boxes' lidar [x,y,z,h,w,l,yaw] -> Box3D map."""
    box_class, _ = load_ab3dmot_association()
    values = torch.as_tensor(boxes, dtype=torch.float32).detach().cpu()
    if values.ndim != 2 or values.shape[1] != 7 or not torch.isfinite(values).all():
        raise ValueError("association boxes must be finite [N,7]")
    # AB3Dmot Box3D is camera ordered [x,y,z,h,w,l,ry]; online.py uses the
    # same established V2V4Real adapter: lidar [x,y,z] -> [x,z,y].
    return [box_class(x=float(value[0]), y=float(value[2]), z=float(value[1]),
                      h=float(value[3]), w=float(value[4]), l=float(value[5]),
                      ry=float(value[6])) for value in values]


def associate_car_giou(proposals, tracks):
    """Return ``{track_index: proposal_index}`` using AB3DMOT Car defaults."""
    _, data_association = load_ab3dmot_association()
    dets, trks = _box3d_from_lidar(proposals), _box3d_from_lidar(tracks)
    matches, _, _, _, _ = data_association(
        dets, trks, metric="giou_3d", threshold=-0.2, algm="hungar")
    return {int(track_index): int(proposal_index)
            for proposal_index, track_index in matches.tolist()}
