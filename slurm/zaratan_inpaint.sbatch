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

echo "[latent_void] running full configured pipeline on $(hostname)"
python3 -m latent_void run --config "${CONFIG}"
