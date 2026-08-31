"""Public data contracts for causal DualCMP clips.

Runtime data deliberately contains only observations and timing/geometry
metadata.  All boxes, identities and future trajectories live in target data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Iterable, Tuple


FORBIDDEN_RUNTIME_KEYS = frozenset({
    "actor_ids", "ass_id", "future_mask", "future_xy", "gt_box_tensor",
    "label_dict", "matched_car_id", "object_bbx_center", "object_bbx_mask",
    "object_ids", "source_actor_ids", "source_boxes", "source_box_mask",
    "target_frames", "targets", "vehicles",
})


def find_forbidden_runtime_keys(
    value: Any,
    forbidden_keys: Iterable[str] = FORBIDDEN_RUNTIME_KEYS,
    path: Tuple[str, ...] = (),
) -> Tuple[Tuple[str, ...], ...]:
    """Return all nested target-only keys found in a runtime payload."""

    forbidden = set(forbidden_keys)
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = path + (str(key),)
            if str(key) in forbidden:
                found.append(child_path)
            found.extend(find_forbidden_runtime_keys(child, forbidden, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(find_forbidden_runtime_keys(child, forbidden, path + (str(index),)))
    return tuple(found)


def assert_runtime_is_supervision_free(value: Any) -> None:
    """Fail closed if GT, actor identity, or future supervision leaks runtime."""

    violations = find_forbidden_runtime_keys(value)
    if violations:
        raise ValueError(
            "Runtime payload contains supervision-only keys: "
            + ", ".join(".".join(item) for item in violations)
        )
