#!/usr/bin/env bash
# Environment setup for the DEM generation project.
# Uses uv to build a clean Python 3.11 venv (system python is 3.8, too old).
# Disk-disciplined: cleans the uv/pip caches afterward (24GB disk budget).
set -euo pipefail

PROJ=/home/claude/dem-gen
cd "$PROJ"

echo "==== [1/5] install uv ===="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
uv --version

echo "==== [2/5] create venv (python 3.11) ===="
uv venv --python 3.11 "$PROJ/.venv"
# shellcheck disable=SC1091
source "$PROJ/.venv/bin/activate"
python --version

echo "==== [3/5] install torch (cu121) ===="
# Install torch from the cu121 index explicitly to get CUDA wheels.
uv pip install --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1

echo "==== [4/5] install remaining requirements ===="
uv pip install \
  numpy==1.26.4 scipy==1.13.1 \
  rasterio==1.3.10 pyproj==3.6.1 shapely==2.0.4 \
  matplotlib==3.7.5 Pillow==10.4.0 scikit-image==0.21.0 scikit-learn==1.3.2 \
  POT==0.9.4 tqdm==4.66.5 pyyaml==6.0.2 einops==0.8.0 tensorboard==2.14.0 requests==2.32.3

echo "==== [5/5] verify + clean caches ===="
python - <<'PY'
import torch, numpy, scipy, rasterio, matplotlib, skimage, sklearn, ot, einops
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available(),
      "cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0),
          "vram_GB", round(torch.cuda.get_device_properties(0).total_memory/1e9, 1))
print("numpy", numpy.__version__, "scipy", scipy.__version__,
      "rasterio", rasterio.__version__, "skimage", skimage.__version__)
import rasterio as rio
print("rasterio GDAL", rio.__gdal_version__)
PY

uv cache clean || true
echo "==== DISK USAGE ===="
df -h /home/claude | tail -1
du -sh "$PROJ/.venv" 2>/dev/null || true
echo "==== SETUP COMPLETE ===="
