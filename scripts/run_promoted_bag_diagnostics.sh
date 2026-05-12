#!/usr/bin/env bash
set -euo pipefail

# Export + diagnose bag with promoted GSRecon fine-tune checkpoint.

ROOT="/scratch/zt1/project/msml612pcs3/user/gnanesh/latent_void"
cd "$ROOT"

source .venvs/latent_void_py310/bin/activate
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

INIT_MODEL_STATE="runs/source_recovery/bag_last_blocks_500/gsrecon_scene_patch_finetuned.pt"
OUT_DIR="runs/source_recovery/bag_last_blocks_500_export"
DIAG_DIR="runs/source_recovery/bag_last_blocks_500_render_diagnostics"

if [[ ! -f "$INIT_MODEL_STATE" ]]; then
  echo "Missing promoted checkpoint: $INIT_MODEL_STATE" >&2
  exit 1
fi

python -u tools/run_gsrecon_export.py \
  --diffsplat-root external/DiffSplat \
  --weights checkpoints/diffsplat/gsrecon_gobj265k_cnp_even4 \
  --init-model-state "$INIT_MODEL_STATE" \
  --gsvae-weights checkpoints/diffsplat/gsvae_gobj265k_sdxl_fp16 \
  --sdxl-vae-path checkpoints/diffsplat_aux/sdxl-vae-fp16-fix \
  --tiny-vae-path checkpoints/diffsplat_aux/taesdxl \
  --dataset-root data/inpaint360 \
  --scene bag \
  --geometry-manifest runs/inpaint360gs_bag/geometry/geometry_manifest.json \
  --output-dir "$OUT_DIR"

python -u tools/diagnose_diffsplat_render.py \
  --diffsplat-root external/DiffSplat \
  --gsvae-weights checkpoints/diffsplat/gsvae_gobj265k_sdxl_fp16 \
  --sdxl-vae-path checkpoints/diffsplat_aux/sdxl-vae-fp16-fix \
  --tiny-vae-path checkpoints/diffsplat_aux/taesdxl \
  --gaussian-npz "${OUT_DIR}/gaussians.npz" \
  --gs-grid-path "${OUT_DIR}/gs_grid.npy" \
  --latent-path "${OUT_DIR}/latent.npy" \
  --compare-latent-path runs/inpaint360gs_bag/inpaint/latent_inpainted.npy \
  --output-dir "$DIAG_DIR"

echo "Promoted bag diagnostics complete."
echo "Inspect:"
echo "  ${DIAG_DIR}/direct_gs_grid/"
echo "  ${DIAG_DIR}/latent_reconstruction/"
echo "  ${DIAG_DIR}/edited_latent/"
