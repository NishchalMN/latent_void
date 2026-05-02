#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/gnanesh/scratch.msml612pcs3/latent_void}"
PY_MODULE="${PY_MODULE:-python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.3.0/gcc/11.3.0/zen2}"
VENV="${VENV:-${ROOT}/.venvs/latent_void_py310}"

cd "${ROOT}"
mkdir -p external data/downloads data/inpaint360gs checkpoints logs

clone_or_update() {
  local url="$1"
  local dst="$2"
  if [[ -d "${dst}/.git" ]]; then
    git -C "${dst}" pull --ff-only || true
  else
    git clone --depth 1 "${url}" "${dst}"
  fi
}

clone_or_update_recursive() {
  local url="$1"
  local dst="$2"
  if [[ -d "${dst}/.git" ]]; then
    git -C "${dst}" pull --ff-only || true
    git -C "${dst}" submodule update --init --recursive
  else
    git clone --recursive --depth 1 "${url}" "${dst}"
  fi
}

clone_or_update https://github.com/chenguolin/DiffSplat.git external/DiffSplat
clone_or_update https://github.com/facebookresearch/sam3.git external/sam3
clone_or_update https://github.com/dfki-av/Inpaint360GS.git external/Inpaint360GS

module load "${PY_MODULE}" >/dev/null 2>&1 || module load "${PY_MODULE}"
if [[ "${INSTALL_GPU_DEPS:-0}" == "1" ]]; then
  module load "${CUDA_MODULE}" >/dev/null 2>&1 || module load "${CUDA_MODULE}" || true
fi
if [[ ! -d "${VENV}" ]]; then
  python -m venv "${VENV}"
fi

unset PYTHONPATH
source "${VENV}/bin/activate"
python -m pip install --upgrade wheel setuptools
python -m pip install \
  "numpy>=1.26,<2" PyYAML gdown huggingface_hub pillow tqdm \
  requests "requests[socks]" idna certifi charset-normalizer urllib3 beautifulsoup4

if [[ "${INSTALL_GPU_DEPS:-0}" == "1" ]]; then
  CUDA_MODULE="${CUDA_MODULE:-cuda/12.3.0/gcc/11.3.0/zen2}"
  module load "${CUDA_MODULE}" >/dev/null 2>&1 || module load "${CUDA_MODULE}" || true
  python -m pip install torch torchvision --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  python -m pip install -e external/sam3
  python -m pip install -r external/DiffSplat/settings/requirements.txt
  python -m pip install "${MARIGOLD_DIFFUSERS_SPEC:-diffusers>=0.34,<0.36}" transformers accelerate safetensors
  mkdir -p external/DiffSplat/extensions
  clone_or_update_recursive https://github.com/BaowenZ/RaDe-GS.git external/DiffSplat/extensions/RaDe-GS
  export MAX_JOBS="${MAX_JOBS:-4}"
  if command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  fi
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
  python -m pip install --no-build-isolation external/DiffSplat/extensions/RaDe-GS/submodules/diff-gaussian-rasterization
fi

DIFFSPLAT_CKPT_DIR="${DIFFSPLAT_CKPT_DIR:-${ROOT}/checkpoints/diffsplat}"
if [[ "${DOWNLOAD_DIFFSPLAT_CKPTS:-1}" == "1" ]]; then
  if [[ -d "${DIFFSPLAT_CKPT_DIR}/gsvae_gobj265k_sdxl_fp16" && -d "${DIFFSPLAT_CKPT_DIR}/gsrecon_gobj265k_cnp_even4" ]]; then
    echo "[exists] DiffSplat checkpoints under ${DIFFSPLAT_CKPT_DIR}"
  else
    python external/DiffSplat/download_ckpt.py --model_type pas --local_dir "${DIFFSPLAT_CKPT_DIR}"
  fi
fi

python -m latent_void validate-config --config configs/zaratan_inpaint360gs_bag.yaml

echo "[ok] Zaratan lightweight deps ready at ${VENV}"
