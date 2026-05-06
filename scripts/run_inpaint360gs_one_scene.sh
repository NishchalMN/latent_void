#!/usr/bin/env bash
# One-command full Inpaint360GS pipeline for a single scene, with three 3D inpaint
# budgets (5000 / 12000 / 20000). Usage:
#   bash scripts/run_inpaint360gs_one_scene.sh car
#   bash scripts/run_inpaint360gs_one_scene.sh cube
#
# Optional env:
#   RESOLUTION=2          (default 2)
#   FINETUNE_ITERS="5000 12000 20000"
#   SKIP_FID=1            (default: pass --skip-fid-eval)
#   START_STAGE=1         (e.g. 10 to rerun only fusion+inpaint+eval)
#   SKIP_SEG=1            (--skip-seg: skip stages 1–5 if already done)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="${1:?Usage: $0 <scene_name>}"

VENV="${ROOT}/.venvs/latent_void_py310"
RESOLUTION="${RESOLUTION:-2}"
START_STAGE="${START_STAGE:-1}"
FINETUNE_ITERS="${FINETUNE_ITERS:-5000 12000 20000}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTHONPATH="${ROOT}/external/Inpaint360GS:${ROOT}/external/Inpaint360GS/gaussian_splatting:${PYTHONPATH:-}"

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

mkdir -p "${ROOT}/output/inpaint360/${SCENE}/logs"

SKIP_SEG_FLAG=()
if [[ "${SKIP_SEG:-0}" == "1" ]]; then
  SKIP_SEG_FLAG=(--skip-seg)
fi

SKIP_FID_FLAG=(--skip-fid-eval)
if [[ "${SKIP_FID:-1}" == "0" ]]; then
  SKIP_FID_FLAG=()
fi

cd "${ROOT}"
python -u tools/run_inpaint360gs_full.py \
  --scenes "${SCENE}" \
  --resolution "${RESOLUTION}" \
  --start-stage "${START_STAGE}" \
  --finetune-iterations ${FINETUNE_ITERS} \
  "${SKIP_SEG_FLAG[@]}" \
  "${SKIP_FID_FLAG[@]}" \
  --data-root "${ROOT}/data/inpaint360" \
  --output-root "${ROOT}/output/inpaint360" \
  2>&1 | tee "${ROOT}/output/inpaint360/${SCENE}/logs/full_pipeline_${SCENE}.log"
