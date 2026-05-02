#!/bin/bash
#SBATCH --job-name=latent-void-sam3
#SBATCH --account=msml612pcs3-class
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CONFIG="${1:-configs/zaratan_inpaint360gs_bag.yaml}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
EXTRA_ARGS=("$@")
mkdir -p logs

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
if [[ -d .venvs/latent_void_py310 ]]; then
  unset PYTHONPATH
  source .venvs/latent_void_py310/bin/activate
fi

echo "[latent_void] running SAM 3 multiview segmentation on $(hostname)"
python -m latent_void segment --config "${CONFIG}" "${EXTRA_ARGS[@]}"
