"""Tensor-only receiver-centric runtime actor state."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch


_GEOMETRY_DTYPES = (torch.float32, torch.float64)


@dataclass(frozen=True)
class ActorState:
    """Allocated actors are valid; active actors are emitted this frame.

    A valid but inactive slot remains allocated for a kept miss and may retain
    its box, feature, and causal history.  ``active_mask`` only controls this
    frame's emitted tracks.  Geometry state must remain outside AMP autocast in
    FP32, or use explicit FP64; FP16 and BF16 are not supported.
    """

    feature: torch.Tensor
    box: torch.Tensor
    velocity: torch.Tensor
    score: torch.Tensor
    quality: torch.Tensor
    valid_mask: torch.Tensor
    active_mask: torch.Tensor
    age: torch.Tensor
    miss_count: torch.Tensor
    runtime_id: torch.Tensor
    next_runtime_id: torch.Tensor
    last_update_time_s: torch.Tensor
    history_box: torch.Tensor
    history_mask: torch.Tensor
    receiver_pose: torch.Tensor
    receiver_time_s: torch.Tensor

    @classmethod
    def empty(cls, batch_size, max_actors, feature_dim, history_len,
              receiver_pose, receiver_time_s, dtype=None):
        """Allocate an empty state with receiver pose/time already attached."""
        if not isinstance(receiver_pose, torch.Tensor) or not isinstance(receiver_time_s, torch.Tensor):
            raise TypeError("receiver_pose and receiver_time_s must be tensors")
        if not receiver_pose.is_floating_point() or not receiver_time_s.is_floating_point():
            raise TypeError("receiver pose/time must use floating dtype")
        if receiver_pose.dtype not in _GEOMETRY_DTYPES or receiver_time_s.dtype not in _GEOMETRY_DTYPES:
            raise TypeError("ActorState geometry supports only torch.float32 or torch.float64")
        if dtype is None:
            dtype = receiver_pose.dtype
        if dtype not in _GEOMETRY_DTYPES:
            raise TypeError("ActorState dtype must be torch.float32 or torch.float64")
        if receiver_pose.shape != (batch_size, 4, 4) or receiver_time_s.shape != (batch_size,):
            raise ValueError("receiver pose/time must have [B,4,4] and [B] shapes")
        if receiver_pose.dtype != receiver_time_s.dtype or receiver_pose.device != receiver_time_s.device:
            raise ValueError("receiver pose/time must share dtype/device")
        if dtype != receiver_pose.dtype:
            raise ValueError("ActorState.empty does not implicitly convert receiver pose/time dtype")
        if not torch.isfinite(receiver_pose).all() or not torch.isfinite(receiver_time_s).all():
            raise ValueError("receiver pose/time must be finite")
        device = receiver_pose.device
        state = cls(
            feature=torch.zeros(batch_size, max_actors, feature_dim, device=device, dtype=dtype),
            box=torch.zeros(batch_size, max_actors, 7, device=device, dtype=dtype),
            velocity=torch.zeros(batch_size, max_actors, 2, device=device, dtype=dtype),
            score=torch.zeros(batch_size, max_actors, device=device, dtype=dtype),
            quality=torch.zeros(batch_size, max_actors, device=device, dtype=dtype),
            valid_mask=torch.zeros(batch_size, max_actors, device=device, dtype=torch.bool),
            active_mask=torch.zeros(batch_size, max_actors, device=device, dtype=torch.bool),
            age=torch.zeros(batch_size, max_actors, device=device, dtype=torch.long),
            miss_count=torch.zeros(batch_size, max_actors, device=device, dtype=torch.long),
            runtime_id=torch.full((batch_size, max_actors), -1, device=device, dtype=torch.long),
            next_runtime_id=torch.zeros(batch_size, device=device, dtype=torch.long),
            last_update_time_s=torch.full((batch_size, max_actors), float("nan"), device=device, dtype=dtype),
            history_box=torch.zeros(batch_size, max_actors, history_len, 7, device=device, dtype=dtype),
            history_mask=torch.zeros(batch_size, max_actors, history_len, device=device, dtype=torch.bool),
            receiver_pose=receiver_pose,
            receiver_time_s=receiver_time_s,
        )
        state.validate()
        return state

    def validate(self):
        """Fail closed on shape, dtype, device, finite-value, or mask errors."""
        shape = self.valid_mask.shape
        if self.valid_mask.ndim != 2 or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must have shape [B,A] and bool dtype")
        if self.active_mask.shape != shape or self.active_mask.dtype != torch.bool:
            raise ValueError("active_mask must match valid_mask with bool dtype")
        if torch.any(self.active_mask & ~self.valid_mask):
            raise ValueError("active actors must be allocated valid actors")
        batch_size, max_actors = shape
        float_fields = {
            "feature": self.feature, "box": self.box, "velocity": self.velocity,
            "score": self.score, "quality": self.quality,
            "history_box": self.history_box, "receiver_pose": self.receiver_pose,
            "receiver_time_s": self.receiver_time_s,
        }
        reference = self.feature
        if not reference.is_floating_point():
            raise TypeError("feature must use floating dtype")
        if reference.dtype not in _GEOMETRY_DTYPES:
            raise TypeError("ActorState geometry supports only torch.float32 or torch.float64")
        for name, value in float_fields.items():
            if not value.is_floating_point() or value.dtype != reference.dtype or value.device != reference.device:
                raise ValueError("%s must share ActorState floating dtype/device" % name)
            if not torch.isfinite(value).all():
                raise ValueError("%s must be finite" % name)
        if self.feature.shape[:2] != shape or self.feature.ndim != 3:
            raise ValueError("feature must have shape [B,A,F]")
        if self.box.shape != (batch_size, max_actors, 7):
            raise ValueError("box must have shape [B,A,7]")
        if self.velocity.shape != (batch_size, max_actors, 2):
            raise ValueError("velocity must have shape [B,A,2]")
        if self.score.shape != shape or self.quality.shape != shape:
            raise ValueError("score and quality must have shape [B,A]")
        if self.history_box.ndim != 4 or self.history_box.shape[:2] != shape or self.history_box.shape[-1] != 7:
            raise ValueError("history_box must have shape [B,A,H,7]")
        if self.history_mask.shape != self.history_box.shape[:-1] or self.history_mask.dtype != torch.bool:
            raise ValueError("history_mask must match history_box [B,A,H]")
        if torch.any(self.history_mask & ~self.valid_mask[:, :, None]):
            raise ValueError("history_mask can only be set for valid actors")
        if self.receiver_pose.shape != (batch_size, 4, 4):
            raise ValueError("receiver_pose must have shape [B,4,4]")
        if self.receiver_time_s.shape != (batch_size,):
            raise ValueError("receiver_time_s must have shape [B]")
        for name, value, expected in (
            ("age", self.age, shape), ("miss_count", self.miss_count, shape),
            ("runtime_id", self.runtime_id, shape), ("next_runtime_id", self.next_runtime_id, (batch_size,)),
        ):
            if value.shape != expected or value.dtype != torch.long or value.device != reference.device:
                raise ValueError("%s must use long dtype on the state device" % name)
        if self.last_update_time_s.shape != shape or self.last_update_time_s.dtype != reference.dtype or self.last_update_time_s.device != reference.device:
            raise ValueError("last_update_time_s must have shape [B,A] on the state dtype/device")
        finite_updates = torch.isfinite(self.last_update_time_s)
        if torch.any(finite_updates & ~self.valid_mask) or torch.any(~finite_updates & self.valid_mask):
            raise ValueError("last_update_time_s must be finite exactly for valid actors")
        if torch.any(self.runtime_id[~self.valid_mask] != -1) or torch.any(self.runtime_id[self.valid_mask] < 0):
            raise ValueError("runtime_id must be non-negative exactly for valid actors")
        if torch.any(self.next_runtime_id < 0):
            raise ValueError("next_runtime_id must be non-negative")
        if torch.any(self.age < 0) or torch.any(self.miss_count < 0):
            raise ValueError("age and miss_count must be non-negative")
        for row in range(batch_size):
            row_ids = self.runtime_id[row][self.valid_mask[row]]
            if row_ids.numel() and torch.unique(row_ids).numel() != row_ids.numel():
                raise ValueError("valid runtime_id values must be unique per row")
            if row_ids.numel() and self.next_runtime_id[row] <= row_ids.max():
                raise ValueError("next_runtime_id must exceed every valid runtime_id per row")

    def detached(self):
        """Return a state whose tensors do not retain the prior autograd graph."""
        return ActorState(**{item.name: getattr(self, item.name).detach() for item in fields(self)})

    def reset_rows(self, reset_mask, receiver_pose: Optional[torch.Tensor] = None,
                   receiver_time_s: Optional[torch.Tensor] = None,
                   reset_id_counter_mask: Optional[torch.Tensor] = None):
        """Clear requested rows while preserving sequence counters by default.

        ``runtime_id`` is scoped to one ``(scene, receiver)`` sequence.  Only
        an explicit reset row may also reset its ``next_runtime_id`` counter.
        """
        self.validate()
        if reset_mask.shape != self.receiver_time_s.shape or reset_mask.dtype != torch.bool or reset_mask.device != self.feature.device:
            raise ValueError("reset_mask must be bool [B] on the state device")
        if reset_id_counter_mask is None:
            reset_id_counter_mask = torch.zeros_like(reset_mask)
        if reset_id_counter_mask.shape != reset_mask.shape or reset_id_counter_mask.dtype != torch.bool or reset_id_counter_mask.device != self.feature.device:
            raise ValueError("reset_id_counter_mask must be bool [B] on the state device")
        if torch.any(reset_id_counter_mask & ~reset_mask):
            raise ValueError("reset_id_counter_mask must be a reset_mask subset")
        pose = self.receiver_pose if receiver_pose is None else receiver_pose
        time_s = self.receiver_time_s if receiver_time_s is None else receiver_time_s
        if pose.shape != self.receiver_pose.shape or time_s.shape != self.receiver_time_s.shape:
            raise ValueError("reset pose/time must match state batch shapes")
        if pose.dtype != self.feature.dtype or time_s.dtype != self.feature.dtype or pose.device != self.feature.device or time_s.device != self.feature.device:
            raise ValueError("reset pose/time must match state dtype/device")
        if not torch.isfinite(pose).all() or not torch.isfinite(time_s).all():
            raise ValueError("reset pose/time must be finite")
        def row_mask(value):
            return reset_mask.view((reset_mask.shape[0],) + (1,) * (value.ndim - 1))
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "receiver_pose":
                values[item.name] = torch.where(row_mask(value), pose, value)
            elif item.name == "receiver_time_s":
                values[item.name] = torch.where(reset_mask, time_s, value)
            elif item.name == "runtime_id":
                values[item.name] = torch.where(row_mask(value), torch.full_like(value, -1), value)
            elif item.name == "next_runtime_id":
                values[item.name] = torch.where(reset_id_counter_mask, torch.zeros_like(value), value)
            elif item.name in ("valid_mask", "active_mask", "history_mask"):
                values[item.name] = torch.where(row_mask(value), torch.zeros_like(value), value)
            elif item.name == "last_update_time_s":
                values[item.name] = torch.where(row_mask(value), torch.full_like(value, float("nan")), value)
            else:
                values[item.name] = torch.where(row_mask(value), torch.zeros_like(value), value)
        result = ActorState(**values)
        result.validate()
        return result
