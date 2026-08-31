"""Lightweight causal actor state primitives for DualCMP."""

from .geometry import propagate_state, wrap_angle
from .state import ActorState

__all__ = ["ActorState", "propagate_state", "wrap_angle"]
