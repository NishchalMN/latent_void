#!/bin/bash
#SBATCH --job-name=latent-void-smoke
#SBATCH --account=msml612pcs3-class
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
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

echo "[latent_void] running on $(hostname)"
echo "[latent_void] config: ${CONFIG}"

python -m latent_void validate-config --config "${CONFIG}"
python -m latent_void discover-dataset --config "${CONFIG}"
python -m latent_void prepare-geometry --config "${CONFIG}" --dry-run
python -m latent_void reconstruct --config "${CONFIG}" --dry-run
python -m latent_void segment --config "${CONFIG}" --dry-run
