#!/usr/bin/env bash
set -euo pipefail

# Track B launcher for bag scene (native latent denoiser path).
# Run this inside tmux pane attached to H100.
#
# Usage:
#   bash scripts/run_trackb_bag_tmux.sh
# Optional overrides:
#   DATASET_SAMPLES=1024 TRAIN_STEPS=12000 BATCH_SIZE=8 BASE_CHANNELS=128 bash scripts/run_trackb_bag_tmux.sh

ROOT="/scratch/zt1/project/msml612pcs3/user/gnanesh/latent_void"
cd "$ROOT"

source .venvs/latent_void_py310/bin/activate
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

DATASET_SAMPLES="${DATASET_SAMPLES:-512}"
TRAIN_STEPS="${TRAIN_STEPS:-8000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-128}"

RUN_ROOT="runs/native_latent_bag_train_v1"
TRAIN_OUT="${RUN_ROOT}/h100_train_${TRAIN_STEPS}"
WEIGHTS_PATH="${TRAIN_OUT}/masked_latent_denoiser.pt"
CFG="configs/zaratan_inpaint360gs_bag_masked_denoiser.yaml"

echo "[1/6] Checking required bag artifacts..."
for p in \
  "runs/inpaint360gs_bag/gsrecon/latent.npy" \
  "runs/inpaint360gs_bag/gsrecon/gaussians.npz" \
  "runs/inpaint360gs_bag/void/latent_void_mask.npy"
do
  if [[ ! -f "$p" ]]; then
    echo "Missing required file: $p" >&2
    exit 1
  fi
done

echo "[2/6] Generating masked latent dataset..."
GEN_ARGS=(
  --latent-path runs/inpaint360gs_bag/gsrecon/latent.npy
  --gaussian-npz runs/inpaint360gs_bag/gsrecon/gaussians.npz
  --output-dir "$RUN_ROOT"
  --num-samples "$DATASET_SAMPLES"
  --mask-mode mixed
)
if [[ -f "runs/inpaint360gs_bag/local_patch/local_patch_manifest.json" ]]; then
  GEN_ARGS+=(--patch-manifest runs/inpaint360gs_bag/local_patch/local_patch_manifest.json)
fi
python tools/generate_native_latent_training_data.py "${GEN_ARGS[@]}"

echo "[3/6] Training masked latent denoiser..."
python -u tools/train_masked_latent_denoiser.py \
  --dataset-manifest "${RUN_ROOT}/dataset_manifest.json" \
  --output-dir "$TRAIN_OUT" \
  --steps "$TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --base-channels "$BASE_CHANNELS" \
  --lr 1e-4 \
  --device cuda \
  --log-interval 50

if [[ ! -f "$WEIGHTS_PATH" ]]; then
  echo "Training finished but weights not found: $WEIGHTS_PATH" >&2
  exit 1
fi

echo "[4/6] Writing config override with trained checkpoint..."
TRAIN_STEPS="$TRAIN_STEPS" python - <<'PY'
import yaml
import os
cfg_path = "configs/zaratan_inpaint360gs_bag_masked_denoiser.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)
weights = "runs/native_latent_bag_train_v1/h100_train_%s/masked_latent_denoiser.pt" % os.environ["TRAIN_STEPS"]
cfg.setdefault("checkpoints", {})["latent_inpaint_weights"] = weights
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("Updated", cfg_path, "latent_inpaint_weights =", weights)
PY

echo "[5/6] Running fuse -> inpaint -> render..."
python -m latent_void fuse --config "$CFG"
python -m latent_void inpaint --config "$CFG"
python -m latent_void render --config "$CFG"

echo "[6/6] Writing diagnostics renders..."
python tools/diagnose_diffsplat_render.py \
  --diffsplat-root external/DiffSplat \
  --gsvae-weights checkpoints/diffsplat/gsvae_gobj265k_sdxl_fp16 \
  --sdxl-vae-path checkpoints/diffsplat_aux/sdxl-vae-fp16-fix \
  --tiny-vae-path checkpoints/diffsplat_aux/taesdxl \
  --gaussian-npz runs/inpaint360gs_bag/gsrecon/gaussians.npz \
  --gs-grid-path runs/inpaint360gs_bag/gsrecon/gs_grid.npy \
  --latent-path runs/inpaint360gs_bag/gsrecon/latent.npy \
  --compare-latent-path runs/inpaint360gs_bag/inpaint/latent_inpainted.npy \
  --output-dir runs/inpaint360gs_bag/render_diagnostics_h100_v1

echo "Track B complete."
echo "Check outputs:"
echo "  - ${TRAIN_OUT}/train_masked_latent_denoiser_status.json"
echo "  - runs/inpaint360gs_bag/inpaint/latent_inpainted.npy"
echo "  - runs/inpaint360gs_bag/renders/"
echo "  - runs/inpaint360gs_bag/render_diagnostics_h100_v1/"
