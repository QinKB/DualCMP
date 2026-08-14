"""Stable imports for CMP's bundled V2V4Real MTR implementation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MTR_ROOT = PROJECT_ROOT / "MTR"
if str(MTR_ROOT) not in sys.path:
    sys.path.insert(0, str(MTR_ROOT))

from mtr.models_v2v4real.motion_decoder.mtr_decoder import MTRDecoder
from mtr.utils.motion_utils import batch_nms


__all__ = ["MTRDecoder", "batch_nms"]
