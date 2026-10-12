"""Detector configuration provenance and checkpoint-compatible loading."""
from __future__ import annotations

import copy
import hashlib
import json

from opencood.hypes_yaml.yaml_utils import load_yaml


DEFAULT_HYPES = "opencood/hypes_yaml/point_pillar_v2v4real.yaml"
LEGACY_HYPES = "opencood/hypes_yaml/point_pillar_cobevt_multiego_1x_v2v4real.yaml"
CONFIG_KEYS = ("model", "preprocess", "postprocess", "wild_setting")
DETECTOR_SOURCE = "ucla-mobility/V2V4Real"
DETECTOR_SOURCE_REVISION = "5a821e13753bafc611f95c47bc1a306acdcb0f7c"


def _json_compatible(value):
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_compatible(value.tolist())
    return copy.deepcopy(value)


def detector_config_snapshot(hypes):
    return {key: _json_compatible(hypes[key]) for key in CONFIG_KEYS}


def detector_config_sha256(snapshot):
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def detector_provenance(hypes):
    snapshot = detector_config_snapshot(hypes)
    return {"detector_source": DETECTOR_SOURCE,
            "detector_source_revision": DETECTOR_SOURCE_REVISION,
            "detector_config": snapshot,
            "detector_configuration_sha256": detector_config_sha256(snapshot)}


def inherited_detector_provenance(payload):
    provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    fields = ("detector_source", "detector_source_revision", "detector_config",
              "detector_configuration_sha256")
    if not any(field in provenance for field in fields):
        return {}
    if any(field not in provenance for field in fields):
        raise RuntimeError("checkpoint detector provenance is incomplete")
    snapshot = provenance.get("detector_config")
    digest = provenance.get("detector_configuration_sha256")
    if not isinstance(snapshot, dict) or set(snapshot) != set(CONFIG_KEYS):
        raise RuntimeError("checkpoint detector configuration is invalid")
    if digest != detector_config_sha256(snapshot):
        raise RuntimeError("checkpoint detector configuration hash mismatch")
    if (provenance["detector_source"] != DETECTOR_SOURCE or
            provenance["detector_source_revision"] != DETECTOR_SOURCE_REVISION):
        raise RuntimeError("checkpoint detector source provenance mismatch")
    return {key: copy.deepcopy(provenance[key]) for key in fields}


def load_checkpoint_hypes(path, payload):
    """Restore a checkpoint's detector/data settings, or the explicit CMP legacy yaml."""
    inherited = inherited_detector_provenance(payload)
    hypes = load_yaml(path if inherited else LEGACY_HYPES)
    if inherited:
        hypes.update(inherited["detector_config"])
    return hypes, inherited
