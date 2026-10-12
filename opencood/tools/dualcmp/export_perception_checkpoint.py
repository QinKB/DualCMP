"""Export joint-trained perception for the existing frozen-frontend B stage.

No forecasting weights or target data are retained. This is weight extraction,
not a new tracker, a Phase-A training result, or a forecasting-quality claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from opencood.models.dualcmp.model import DualCmpClipModel
from opencood.tools.dualcmp.detector_config import DEFAULT_HYPES, load_checkpoint_hypes
from opencood.tools.dualcmp.online import _checkpoint_provenance, _sha256_file
from opencood.tools.dualcmp.train_tracking_ddp import (
    _atomic_save, _protocol_sha256, _source_tree_sha256,
)


FORECAST_PREFIXES = ("forecast_head.", "actor_source_tokenizer.", "source_temporal_encoder.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError("refusing to overwrite perception weights")
    payload = torch.load(args.checkpoint, map_location="cpu")
    source = _checkpoint_provenance(args.checkpoint, payload, allow_dirty_smoke=True)
    original = payload["provenance"]
    if "init_checkpoint_sha256" not in original or original.get("partial_epoch"):
        raise ValueError("export requires a completed native joint-training checkpoint")
    hypes, detector_provenance = load_checkpoint_hypes(DEFAULT_HYPES, payload)
    weights = {name: value for name, value in payload["model"].items()
               if not name.startswith(FORECAST_PREFIXES)}
    model = DualCmpClipModel(hypes["model"]["args"], weights["anchors"],
                            hypes["preprocess"]["cav_lidar_range"])
    model.load_state_dict(weights, strict=True)
    assert not any(name.startswith(FORECAST_PREFIXES) for name in model.state_dict())
    repository = Path(__file__).resolve().parents[3]
    provenance = dict(detector_provenance)
    provenance.update({
        "source_state": "dirty-smoke",
        "base_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repository)).decode().strip(),
        "source_tree_sha256": _source_tree_sha256(),
        "protocol_sha256": _protocol_sha256(), "stage": "tracking",
        "detector_compression": original["detector_compression"],
        "tracking_label_contract": original["tracking_label_contract"],
        "tracking_label_scope": original["tracking_label_scope"],
        "perception_export": {
            "origin": "joint_training_weight_extraction_not_phase_a_training",
            "checkpoint": args.checkpoint, "checkpoint_sha256": _sha256_file(args.checkpoint),
            "source_provenance": source, "joint_steps": payload["step"],
            "joint_training_provenance": original,
            "discarded_prefixes": list(FORECAST_PREFIXES),
            "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save({"model": model.state_dict(), "provenance": provenance}, destination)
    print(json.dumps({"output": str(destination), "sha256": _sha256_file(destination),
                      "perception_keys": len(weights), "forecast_keys": 0,
                      "scope": "frozen frontend initialization; no forecasting-quality claim"}))


if __name__ == "__main__":
    main()
