#!/bin/bash
#SBATCH --job-name=latent-void-inpaint
#SBATCH --account=msml612pcs3-class
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CONFIG="${1:-configs/inpaint360gs_example.yaml}"
mkdir -p logs

module load python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
if [[ -d .venvs/latent_void_py310 ]]; then
  unset PYTHONPATH
  source .venvs/latent_void_py310/bin/activate
fi

echo "[latent_void] running full configured pipeline on $(hostname)"
python -m latent_void run --config "${CONFIG}"
