#!/usr/bin/env bash
# Full Inpaint360GS for one scene with virtual-depth prep (affine + planar hole depth)
# after LaMa and before PLY fusion — same recipe as bag (see
# project_memory/INPAINT360GS_BAG_VIRTUAL_DEPTH_FUSION_RECIPE.md).
#
# Usage (GPU node, repo root implied — script cds internally):
#   bash scripts/run_inpaint360gs_scene_virtual_depth_full.sh car
#   REMOVAL_ITER=iteration_2000 VIRTUAL_MASK_SUBDIR=inpaint_2d_unseen_mask_virtual \
#     bash scripts/run_inpaint360gs_scene_virtual_depth_full.sh cube
#
# Env (after phase B starts):
#   FINETUNE_ITERS                        default 12000
#   INPAINT360_CHECKPOINT_VIDEO_ITERS    default "5000 8000" (empty string to disable)
#   SKIP_SEG START_STAGE SKIP_FID as in run_inpaint360gs_one_scene.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="${1:?Usage: $0 <scene_name>}"

REMOVAL_ITER="${REMOVAL_ITER:-iteration_2000}"
MASK_REL="${VIRTUAL_MASK_SUBDIR:-inpaint_2d_unseen_mask_virtual}"

if [[ -x "${ROOT}/scripts/install_inpaint360gs_overrides.sh" ]]; then
  bash "${ROOT}/scripts/install_inpaint360gs_overrides.sh"
fi

export FINETUNE_ITERS="${FINETUNE_ITERS:-12000}"

echo "--- Phase A: stages 1–9, stop after LaMa postprocess ---"
STOP_AFTER_STAGE=9 START_STAGE="${START_STAGE:-1}" \
  bash "${ROOT}/scripts/run_inpaint360gs_one_scene.sh" "${SCENE}"

VIRT_BASE="${ROOT}/output/inpaint360/${SCENE}/virtual/ours_object_removal/${REMOVAL_ITER}"
COMPLETED="${VIRT_BASE}/depth_completed"
HOLE="${VIRT_BASE}/depth"
MASK="${ROOT}/data/inpaint360/${SCENE}/${MASK_REL}"

for need in "${COMPLETED}" "${HOLE}" "${MASK}"; do
  if [[ ! -d "${need}" ]]; then
    echo "ERROR: missing directory: ${need}" >&2
    echo "  Fix REMOVAL_ITER (${REMOVAL_ITER}) or VIRTUAL_MASK_SUBDIR (${MASK_REL})." >&2
    exit 1
  fi
done

echo "--- Virtual depth: affine align depth_completed ---"
python "${ROOT}/tools/inpaint360_align_completed_depth.py" \
  --completed-dir "${COMPLETED}" \
  --hole-dir "${HOLE}" \
  --mask-dir "${MASK}" \
  --backup \
  --ring-width "${RING_WIDTH:-5}"

echo "--- Virtual depth: planar projection onto hole plane ---"
python "${ROOT}/tools/inpaint360_project_completed_to_hole_plane.py" \
  --completed-dir "${COMPLETED}" \
  --hole-dir "${HOLE}" \
  --mask-dir "${MASK}" \
  --backup \
  --ring-width "${RING_WIDTH:-5}"

echo "--- Phase B: fusion + 3D inpaint (${FINETUNE_ITERS} iters) + eval ---"
START_STAGE=10 STOP_AFTER_STAGE="" \
  bash "${ROOT}/scripts/run_inpaint360gs_one_scene.sh" "${SCENE}"
