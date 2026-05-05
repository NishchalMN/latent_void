#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venvs/latent_void_py310"
INPAINT_ROOT="${ROOT}/external/Inpaint360GS"

echo "=== Setting up Inpaint360GS dependencies ==="
echo "ROOT: ${ROOT}"
echo "VENV: ${VENV}"

source "${VENV}/bin/activate"
export OPENBLAS_NUM_THREADS=1

echo "=== Step 1: Install Python dependencies ==="
pip install --no-deps open3d 2>/dev/null || pip install open3d
pip install lpips scikit-learn plyfile opencv-python-headless torchmetrics imageio mmcv timm pytorch_fid 2>/dev/null || true

echo "=== Step 2: Compile Inpaint360GS rasterizer submodules ==="
CUDA_HOME="${CUDA_HOME:-$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))}"
export CUDA_HOME
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"

if ! python -c "import diff_gaussian_rasterization_inpaint360gs" 2>/dev/null; then
    echo "Compiling Inpaint360GS diff-gaussian-rasterization..."
    pip install "${INPAINT_ROOT}/submodules/diff-gaussian-rasterization" --no-build-isolation 2>&1 | tail -5
fi

if ! python -c "import simple_knn" 2>/dev/null; then
    echo "Compiling simple-knn..."
    pip install "${INPAINT_ROOT}/gaussian_splatting/submodules/simple-knn" --no-build-isolation 2>&1 | tail -5
fi

echo "=== Step 3: Install Inpaint360GS package ==="
pip install -e "${INPAINT_ROOT}" --no-deps 2>/dev/null || pip install -e "${INPAINT_ROOT}"

echo "=== Step 4: Download CropFormer weights ==="
WEIGHT_DIR="${INPAINT_ROOT}/seg/weight"
mkdir -p "${WEIGHT_DIR}"

CROPFORMER_WEIGHT="${WEIGHT_DIR}/CropFormer_hornet_3x_03823a.pth"
if [ ! -f "${CROPFORMER_WEIGHT}" ]; then
    echo "Downloading CropFormer weights..."
    python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='qqlu1992/Adobe_EntitySeg',
    filename='CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth',
    repo_type='dataset'
)
shutil.copy2(path, '${CROPFORMER_WEIGHT}')
print(f'Downloaded to ${CROPFORMER_WEIGHT}')
" 2>&1 || echo "WARNING: CropFormer weight download failed. Will need manual download."
fi

SAM_WEIGHT="${WEIGHT_DIR}/sam_vit_h_4b8939.pth"
if [ ! -f "${SAM_WEIGHT}" ]; then
    echo "Downloading SAM ViT-H weights..."
    python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='ybelkada/segment-anything',
    filename='checkpoints/sam_vit_h_4b8939.pth',
    repo_type='model'
)
shutil.copy2(path, '${SAM_WEIGHT}')
print(f'Downloaded to ${SAM_WEIGHT}')
" 2>&1 || echo "WARNING: SAM weight download failed."
fi

echo "=== Step 5: Setup LaMa ==="
LAMA_DIR="${INPAINT_ROOT}/LaMa"
pip install -r "${LAMA_DIR}/requirements.txt" 2>/dev/null || true

LAMA_MODEL_DIR="${LAMA_DIR}/big-lama"
if [ ! -d "${LAMA_MODEL_DIR}" ]; then
    echo "Downloading LaMa big-lama model..."
    cd "${LAMA_DIR}"
    python -c "
import os, subprocess
model_url = 'https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip'
if not os.path.exists('big-lama.zip'):
    subprocess.run(['wget', '-q', model_url, '-O', 'big-lama.zip'], check=True)
if not os.path.exists('big-lama'):
    subprocess.run(['unzip', '-q', 'big-lama.zip'], check=True)
print('LaMa model ready')
" 2>&1 || echo "WARNING: LaMa model download failed."
    cd "${ROOT}"
fi

echo "=== Step 6: Verify imports ==="
python -c "
import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')
try:
    import diff_gaussian_rasterization_inpaint360gs; print('inpaint360gs rasterizer: OK')
except: print('inpaint360gs rasterizer: FAIL')
try:
    import simple_knn; print('simple_knn: OK')
except: print('simple_knn: FAIL')
try:
    import open3d; print(f'open3d: OK ({open3d.__version__})')
except: print('open3d: FAIL')
try:
    import lpips; print('lpips: OK')
except: print('lpips: FAIL')
try:
    import sklearn; print('sklearn: OK')
except: print('sklearn: FAIL')
try:
    import cv2; print('cv2: OK')
except: print('cv2: FAIL')
try:
    import plyfile; print('plyfile: OK')
except: print('plyfile: FAIL')
try:
    import scipy; print('scipy: OK')
except: print('scipy: FAIL')
"

echo "=== Setup complete ==="
