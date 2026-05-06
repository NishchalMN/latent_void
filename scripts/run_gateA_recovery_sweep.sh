#!/usr/bin/env bash
set -euo pipefail

# Gate-A recovery sweep:
# Improve source reconstruction quality (direct_gs_grid) before inpainting.
#
# Usage:
#   bash scripts/run_gateA_recovery_sweep.sh

ROOT="/scratch/zt1/project/msml612pcs3/user/gnanesh/latent_void"
cd "$ROOT"

source .venvs/latent_void_py310/bin/activate
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PATCH_DATASET="runs/scene_patch_training_long/multiscene_patch_dataset/scene_patch_dataset.json"
if [[ ! -f "$PATCH_DATASET" ]]; then
  echo "Missing patch dataset: $PATCH_DATASET" >&2
  exit 1
fi

run_one() {
  local name="$1"
  shift
  local out_dir="runs/source_recovery/${name}"
  local log_path="runs/source_recovery/${name}.log"
  mkdir -p "$(dirname "$log_path")"

  echo "============================================================"
  echo "Run: $name"
  echo "Output: $out_dir"
  echo "Log: $log_path"
  echo "============================================================"

  python -u tools/finetune_gsrecon_scene_patches.py \
    --patch-dataset "$PATCH_DATASET" \
    --output-dir "$out_dir" \
    --steps 500 \
    --batch-size 1 \
    --lr 1e-5 \
    --device cuda \
    --sample-id-contains bag \
    --max-views 8 \
    --eval-interval 100 \
    --fixed-eval-interval 100 \
    --fixed-eval-samples 1 \
    --foreground-weight 2.0 \
    --background-weight 0.5 \
    --alpha-foreground-weight 2.0 \
    --l1-weight 0.2 \
    --ssim-weight 0.1 \
    --alpha-bce-weight 0.05 \
    --alpha-dice-weight 0.05 \
    "$@" 2>&1 | tee "$log_path"
}

run_one "bag_heads_500" --trainable heads
run_one "bag_heads_and_embed_500" --trainable heads_and_embed
run_one "bag_last_blocks_500" --trainable last_blocks --train-last-blocks 2

echo "Sweep complete."
echo "Inspect:"
echo "  runs/source_recovery/*/eval/"
echo "  runs/source_recovery/*.log"
