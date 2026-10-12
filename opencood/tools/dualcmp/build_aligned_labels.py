"""Rebuild train labels using CMP's existing +1000 CAV ID namespace.

Copy CMP BasePostprocessor.generate_object_center's identity mapping and
V2V4Real projection/dedup conventions. Keep KITTI/CAV0 label interfaces.
Never edit evaluator-native legacy labels or invent/interpolate trajectories.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import _load_yaml_cached
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_index import SceneSequenceIndex
from opencood.data_utils.v2v4real.datasets import GT_RANGE
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.utils import box_utils_v2v4real
from opencood.utils.transformation_utils import x1_to_x2


def build_scene(inputs):
    root, output, ordinal, scene = inputs
    index = SceneSequenceIndex(root)
    keys = index.receiver_frame_keys(scene, "0")
    lines = []
    source_digest = hashlib.sha256()
    for frame_ordinal, key in enumerate(keys):
        reference = _load_yaml_cached(str(index.receiver_record(scene, "0", key).yaml_path))["lidar_pose"]
        actors = {}
        for source in index.source_ids(scene):
            record = index.source_record(scene, source, key)
            if record is None:
                continue
            raw_bytes = record.yaml_path.read_bytes()
            source_digest.update(source.encode() + b"\0" + key.encode() + b"\0" + raw_bytes)
            raw = _load_yaml_cached(str(record.yaml_path))
            for raw_id, actor in raw.get("vehicles", {}).items():
                if not 0 <= int(raw_id) < 1000 or not -1 <= int(actor.get("ass_id", -1)) < 1000:
                    raise RuntimeError("CMP +1000 namespace assumption is incompatible with raw IDs")
            projected = {}
            # DMSTrack labels originate from LateFusionDataset, NOT the
            # intermediate-fusion detection target path. Crop in source space
            # first, project float32 corners, dedup, then final CAV0 GT crop.
            box_utils_v2v4real.project_world_objects(
                raw.get("vehicles", {}), projected, np.eye(4), GT_RANGE, "hwl")
            if not projected:
                continue
            raw_ids = list(projected)
            corners = box_utils_v2v4real.boxes_to_corners_3d(
                np.vstack([projected[raw_id]["coord"] for raw_id in raw_ids]), "hwl")
            transform = torch.from_numpy(x1_to_x2(
                np.asarray(raw["lidar_pose"]), np.asarray(reference))).float()
            corners = box_utils_v2v4real.project_box3d(torch.from_numpy(corners).float(), transform)
            for raw_id, corner in zip(raw_ids, corners):
                associated = int(projected[raw_id]["ass_id"])
                identity = associated if associated != -1 else int(raw_id) + 1000 * int(source)
                actors.setdefault(identity, corner)
        if actors:
            identifiers = sorted(actors)
            corners = torch.stack([actors[identity] for identity in identifiers])
            keep = box_utils_v2v4real.get_mask_for_boxes_within_range_torch(corners)
            boxes = box_utils_v2v4real.corner_to_center(corners[keep].numpy(), order="hwl")
            for identity, box in zip(np.asarray(identifiers)[keep.numpy()], boxes):
                x, y, z, height, width, length, yaw = map(float, box)
                values = [height, width, length, x, z, y, yaw]
                lines.append("%d %d Car 0 0 0 0 0 0 0 %s" %
                             (frame_ordinal, identity, " ".join("%.9g" % value for value in values)))
    name = "%04d.txt" % ordinal
    label_bytes = ("\n".join(lines) + "\n").encode()
    (Path(output) / name).write_bytes(label_bytes)
    return {"scene": scene, "file": name, "frames": len(keys), "rows": len(lines),
            "sha256": hashlib.sha256(label_bytes).hexdigest(),
            "source_yaml_sha256": source_digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--tracker-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    output = Path(args.output)
    manifest_path = output.parent / "aligned_labels_manifest.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite labels/provenance")
    hypes, _ = load_checkpoint_hypes(DEFAULT_HYPES, torch.load(args.tracker_checkpoint, map_location="cpu"))
    index = SceneSequenceIndex(args.root)
    lidar_range = list(hypes["preprocess"]["cav_lidar_range"])
    output.mkdir(parents=True)
    inputs = [(args.root, str(output), ordinal, scene)
              for ordinal, scene in enumerate(index.scene_ids())]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        scenes = list(executor.map(build_scene, inputs))
    seqmap = ["%04d empty 000000 %06d" % (ordinal, scene["frames"] - 1)
              for ordinal, scene in enumerate(scenes)]
    (output.parent / "v2v4real_train_evaluate_tracking.seqmap.val").write_text("\n".join(seqmap) + "\n")
    manifest = {"version": "aligned-cmp-identities-v1", "split": "train", "namespace_offset": 1000,
                "reference": "Reference/Code/CMP/opencood/data_utils/v2v4real/post_processor/base_postprocessor.py::generate_object_center",
                "coordinate_frame": "capture_cav0", "heading_unit": "radians", "scenes": scenes,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "detector_lidar_range": lidar_range, "gt_range": GT_RANGE,
                "geometry_reference": "DMSTrack/V2V4Real/opencood/data_utils/datasets/late_fusion_dataset.py::get_item_single_car; post_processor/base_postprocessor.py::generate_gt_bbx",
                "geometry_contract": "source GT range -> float32 corners -> CAV0 transform -> ID dedup -> all XY corners in CAV0 GT range",
                "dedup": "first source for explicit shared association ID"}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "scenes": len(scenes), "rows": sum(scene["rows"] for scene in scenes),
                      "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
