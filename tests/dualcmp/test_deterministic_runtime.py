"""Formal B/C entries share the validated strict runtime, not model changes."""
import os

import torch

from opencood.tools.dualcmp.train_forecast_ddp import _configure_deterministic_runtime
from opencood.tools.dualcmp.train_joint_ddp import _configure_deterministic_runtime as joint_runtime


def test_formal_entries_configure_runtime_before_cuda():
    assert joint_runtime is _configure_deterministic_runtime
    assert not torch.cuda.is_initialized()
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
                os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    try:
        contract = _configure_deterministic_runtime(True)
        assert contract["deterministic_algorithms"] is True
        assert contract["cudnn_deterministic"] is True
        assert contract["cudnn_benchmark"] is False
        assert contract["cublas_workspace"] == ":4096:8"
        assert not torch.cuda.is_initialized()
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.deterministic = previous[1]
        torch.backends.cudnn.benchmark = previous[2]
        if previous[3] is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous[3]
