"""Stable imports for the MOTR persistent-query transformer."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOTR_ROOT = PROJECT_ROOT / "third_party" / "MOTR"
MOTR_OPS_ROOT = MOTR_ROOT / "models" / "ops"

for path in (MOTR_ROOT, MOTR_OPS_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

try:
    from models.deformable_transformer import DeformableTransformer
    from util.misc import inverse_sigmoid
except (ImportError, OSError) as exc:  # pragma: no cover - environment diagnostic
    raise ImportError(
        "Pinned MOTR dependencies are unavailable. Build "
        "third_party/MOTR/models/ops before importing CoLIF."
    ) from exc


__all__ = [
    "DeformableTransformer",
    "inverse_sigmoid",
]
