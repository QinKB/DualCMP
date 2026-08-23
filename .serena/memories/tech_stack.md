# Tech stack
- Python 3.10, PyTorch 2.1.1/CUDA 12.1, NCCL DDP, pytest, and the `coop` Conda environment.
- Cooperative perception is based on OpenCOOD/CMP PointPillar + CoBEVT; CoLIF model code is under `opencood/models/colif` with train/eval entry points under `opencood/tools`.
- V2V4Real strict-online training and evaluation semantics are canonical in `docs/colif_training.md` and `model_design.md`; verify installed versions before changing dependencies.
