"""Native trainer with explicit PyTorch deterministic-runtime diagnostics."""
import json
import os
from pathlib import Path
import sys

import torch
from opencood.tools.dualcmp import train_forecast_ddp


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
train_forecast_ddp.main()
if int(os.environ.get("RANK", 0)) == 0:
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    (output / "deterministic_runtime.json").write_text(json.dumps({
        "torch_version": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "scope": "controlled runtime experiment; not a claim about custom CUDA kernels"}, indent=2))
