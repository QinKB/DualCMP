"""Stable imports for the two upstream modules reused by fused CoLIF.

MOTR supplies the persistent-query deformable transformer, while CMP's MTR
package supplies the trajectory context encoder and motion decoder.  Keeping
both imports in one deterministic module avoids depending on stale bytecode or
on whichever project root happened to appear first in ``PYTHONPATH``.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOTR_ROOT = PROJECT_ROOT / "third_party" / "MOTR"
MOTR_OPS_ROOT = MOTR_ROOT / "models" / "ops"
MTR_ROOT = PROJECT_ROOT / "MTR"

for path in (MTR_ROOT, MOTR_ROOT, MOTR_OPS_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

try:
    from models.deformable_transformer import DeformableTransformer
    from util.misc import inverse_sigmoid
    from mtr.models_v2v4real.context_encoder.mtr_encoder import MTREncoder
    from mtr.models_v2v4real.motion_decoder.mtr_decoder import MTRDecoder
    from mtr.utils.motion_utils import batch_nms
except (ImportError, OSError) as exc:  # pragma: no cover - environment diagnostic
    raise ImportError(
        "Pinned MOTR/CMP-MTR dependencies are unavailable. Build "
        "third_party/MOTR/models/ops and MTR/mtr/ops before importing CoLIF."
    ) from exc


__all__ = [
    "DeformableTransformer",
    "inverse_sigmoid",
    "MTREncoder",
    "MTRDecoder",
    "batch_nms",
]
