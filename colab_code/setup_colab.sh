#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/colab_code/requirements-colab.txt"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GiB):", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1))
PY
