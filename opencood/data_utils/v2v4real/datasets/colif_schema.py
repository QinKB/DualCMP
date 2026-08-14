"""Runtime/target boundary checks used by CoLIF data and inference paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Iterable, Tuple


FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "actor_ids",
        "ass_id",
        "future_mask",
        "future_xy",
        "gt_box_tensor",
        "label_dict",
        "matched_car_id",
        "object_bbx_center",
        "object_bbx_mask",
        "object_ids",
        "target_frames",
        "targets",
        "vehicles",
    }
)


def find_forbidden_runtime_keys(
    value: Any,
    forbidden_keys: Iterable[str] = FORBIDDEN_RUNTIME_KEYS,
    path: Tuple[str, ...] = (),
) -> Tuple[Tuple[str, ...], ...]:
    """Recursively return paths to supervision-only keys in runtime data."""

    forbidden = set(forbidden_keys)
    found = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            child_path = path + (key_string,)
            if key_string in forbidden:
                found.append(child_path)
            found.extend(find_forbidden_runtime_keys(child, forbidden, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found.extend(
                find_forbidden_runtime_keys(child, forbidden, path + (str(index),))
            )
    return tuple(found)


def assert_runtime_is_supervision_free(value: Any) -> None:
    """Fail closed when a runtime frame contains GT or future supervision."""

    violations = find_forbidden_runtime_keys(value)
    if violations:
        formatted = ", ".join(".".join(path) for path in violations)
        raise ValueError("Runtime payload contains supervision-only keys: %s" % formatted)
