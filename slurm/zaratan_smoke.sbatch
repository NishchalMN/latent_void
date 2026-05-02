#!/bin/bash
#SBATCH --job-name=latent-void-smoke
#SBATCH --account=msml612pcs3-class
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

CONFIG="${1:-configs/inpaint360gs_example.yaml}"
mkdir -p logs

echo "[latent_void] running on $(hostname)"
echo "[latent_void] config: ${CONFIG}"

python3 -m latent_void validate-config --config "${CONFIG}"
python3 -m latent_void discover-dataset --config "${CONFIG}"
python3 -m latent_void reconstruct --config "${CONFIG}" --dry-run
python3 -m latent_void segment --config "${CONFIG}" --dry-run
