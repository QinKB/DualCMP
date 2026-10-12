"""Read-only audit of unassociated cross-CAV annotation ID namespaces.

Compare the exact DMSTrack +100 and CMP +1000 mapping. Same-frame distances
describe potential aliases; they never cause sample filtering in this script.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_dataset import _load_yaml_cached
from opencood.data_utils.v2v4real.datasets.dualcmp_sequence_index import SceneSequenceIndex


def audit_scene(inputs):
    scene, sources = inputs
    max_local = max_associated = -1
    counters = {str(offset): {"frames_with_aliases": 0, "aliased_ids": 0,
                              "aliased_ids_separated_gt_3_6m": 0} for offset in (100, 1000)}
    examples = []
    by_key = {source: {record.frame_key: record for record in records}
              for source, records in sources.items()}
    for key in sorted(by_key["0"]):
        data = {source: _load_yaml_cached(str(records[key].yaml_path))
                for source, records in by_key.items() if key in records}
        reference = np.asarray(data["0"]["lidar_pose"])
        items = []
        for source, content in data.items():
            transform = np.linalg.inv(reference) @ np.asarray(content["lidar_pose"])
            for raw_id, actor in content.get("vehicles", {}).items():
                raw_id, associated = int(raw_id), int(actor.get("ass_id", -1))
                max_local = max(max_local, raw_id)
                max_associated = max(max_associated, associated)
                if associated != -1:
                    continue  # Explicit cross-view association is not an alias.
                point = transform @ np.asarray(list(actor["location"]) + [1.])
                items.append((int(source), raw_id, point[:2]))
        for offset in (100, 1000):
            groups = {}
            for source, raw_id, point in items:
                groups.setdefault(raw_id + offset * source, []).append((source, raw_id, point))
            aliased = {identifier: origins for identifier, origins in groups.items()
                       if len({origin[0] for origin in origins}) > 1}
            counters[str(offset)]["frames_with_aliases"] += int(bool(aliased))
            counters[str(offset)]["aliased_ids"] += len(aliased)
            for identifier, origins in aliased.items():
                distance = max(float(np.linalg.norm(a[2] - b[2])) for a in origins for b in origins)
                counters[str(offset)]["aliased_ids_separated_gt_3_6m"] += int(distance > 3.6)
                if offset == 100 and len(examples) < 8:
                    examples.append({"frame": key, "mapped_id": identifier,
                                     "distance_m": distance,
                                     "origins": [{"source": a[0], "raw_id": a[1], "xy_in_cav0": a[2].tolist()}
                                                 for a in origins]})
    return {"scene": scene, "frames": len(by_key["0"]), "max_local_id": max_local,
            "max_associated_id": max_associated, "mapping_offsets": counters, "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite annotation evidence")
    result = {"scope": "all complete raw train/test YAML; unassociated cross-CAV identity aliases only",
              "caveat": "potential aliases and descriptive 3.6m separation, not automatic validity or exclusion rules",
              "splits": {}}
    for split in ("train", "test"):
        index = SceneSequenceIndex(str(Path(args.root) / split))
        inputs = [(scene, {source: [index.source_record(scene, source, key)
                                   for key in index.receiver_frame_keys(scene, source)]
                           for source in index.source_ids(scene)}) for scene in index.scene_ids()]
        with ProcessPoolExecutor(max_workers=8) as executor:
            scenes = list(executor.map(audit_scene, inputs))
        result["splits"][split] = {
            "scenes": scenes, "frames": sum(scene["frames"] for scene in scenes),
            "max_local_id": max(scene["max_local_id"] for scene in scenes),
            "max_associated_id": max(scene["max_associated_id"] for scene in scenes),
            "mapping_offsets": {str(offset): {key: sum(scene["mapping_offsets"][str(offset)][key] for scene in scenes)
                                               for key in ("frames_with_aliases", "aliased_ids", "aliased_ids_separated_gt_3_6m")}
                                for offset in (100, 1000)}}
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({split: {key: value for key, value in data.items() if key != "scenes"}
                      for split, data in result["splits"].items()}))


if __name__ == "__main__":
    main()
