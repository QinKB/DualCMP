"""Shared CMP-style receiver/ego selection for CoLIF entry points."""

from __future__ import annotations

from collections.abc import Iterable


def _validate_receiver_ids(receiver_ids: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(receiver_id) for receiver_id in receiver_ids)
    if not result:
        raise ValueError("receiver IDs must be a non-empty list")
    if any(receiver_id < 0 for receiver_id in result):
        raise ValueError("receiver IDs must be non-negative integers")
    if len(set(result)) != len(result):
        raise ValueError("receiver IDs must be unique")
    return result


def parse_receiver_ids(value: str) -> tuple[int, ...]:
    """Parse the ordered CLI protocol, e.g. ``0`` or ``0,1``."""

    if not isinstance(value, str):
        raise ValueError("receiver IDs must be a comma-separated string")
    tokens = [token.strip() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError("receiver IDs must be a non-empty comma-separated list")
    try:
        return _validate_receiver_ids(int(token) for token in tokens)
    except ValueError as error:
        raise ValueError("receiver IDs must be non-negative integers") from error


def apply_receiver_override(hypes: dict, raw_value: str | None) -> tuple[int, ...]:
    """Apply a CLI override, or normalize the receiver IDs already in hypes."""

    fusion_args = hypes["fusion"]["args"]
    receiver_ids = (
        _validate_receiver_ids(fusion_args.get("receiver_ids", [0]))
        if raw_value is None
        else parse_receiver_ids(raw_value)
    )
    fusion_args["receiver_ids"] = list(receiver_ids)
    return receiver_ids
