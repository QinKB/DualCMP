"""Explicit, hash-checked raw acquisition boundaries for aligned forecasting."""
import hashlib
import json
from pathlib import Path


DEFAULT_BOUNDARIES = Path(__file__).resolve().parents[3] / "hypes_yaml/dualcmp_sequence_boundaries_v1.json"


def configure_aligned_boundaries(index):
    data = DEFAULT_BOUNDARIES.read_bytes()
    manifest = json.loads(data)
    scene = manifest["scene_id"]
    boundaries = {}
    if scene in index.scene_ids():
        for evidence in manifest["evidence"]:
            record = index.source_record(scene, evidence["source"], evidence["frame"])
            if record is None or hashlib.sha256(record.yaml_path.read_bytes()).hexdigest() != evidence["sha256"]:
                raise RuntimeError("audited acquisition-boundary YAML is missing or changed")
        source = index.source_ids(scene)[0]
        boundaries[scene] = tuple(index.receiver_record(scene, source, frame).frame_index
                                  for frame in manifest["boundaries"])
    index.set_sequence_boundaries(boundaries)
    return {"version": manifest["version"], "sha256": hashlib.sha256(data).hexdigest(),
            "boundaries": {scene: list(frames) for scene, frames in boundaries.items()},
            "rule": manifest["rule"]}
