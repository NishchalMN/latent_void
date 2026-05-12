#!/usr/bin/env bash
# Four matching RGB orbit videos for bag (same resolution & trajectory style as vanilla 3DGS):
#   1) BEFORE  — vanilla 3DGS, object present (iteration 30k)
#   2) REMOVAL — Gaussians after object removal (hole / void)
#   3) AFTER   — inpaint finetune 5k
#   4) AFTER   — inpaint finetune 20k
#
# All MP4s and frames go under */video_rgb/ so layout matches (full-frame RGB only).
#
# Usage (GPU node, repo root):
#   bash scripts/render_bag_before_after_videos.sh
#
# Requires checkpoints:
#   output/inpaint360/bag/3dgs_output/point_cloud/iteration_30000/point_cloud.ply
#   output/inpaint360/bag/point_cloud_object_removal/iteration_2000/point_cloud.ply
#   output/inpaint360/bag/point_cloud_object_inpaint_virtual/iteration_{5000,20000}/point_cloud.ply

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="bag"
DATA="${ROOT}/data/inpaint360/${SCENE}"
GSOUT="${ROOT}/output/inpaint360/${SCENE}/3dgs_output"
MODEL="${ROOT}/output/inpaint360/${SCENE}"
INPAINT_ROOT="${ROOT}/external/Inpaint360GS"
GS_ROOT="${INPAINT_ROOT}/gaussian_splatting"
LOGDIR="${MODEL}/logs"
mkdir -p "${LOGDIR}"

VENV="${ROOT}/.venvs/latent_void_py310"
if [[ -f "${VENV}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV}/bin/activate"
fi

export PYTHONPATH="${INPAINT_ROOT}:${GS_ROOT}:${PYTHONPATH:-}"
VIDEO_SUBDIR="video_rgb"

echo "=== [1/4] BEFORE: vanilla 3DGS (object present), iteration_30000 ==="
export GAUSSIAN_SPLAT_VIDEO_SUBDIR="${VIDEO_SUBDIR}"
cd "${GS_ROOT}"
python -u render.py \
  -s "${DATA}" \
  -m "${GSOUT}" \
  --iteration 30000 \
  --skip_train --skip_test \
  --render_video \
  2>&1 | tee "${LOGDIR}/video_rgb_before_vanilla_30k.log"
unset GAUSSIAN_SPLAT_VIDEO_SUBDIR

RGB_ARGS=(--video_rgb_only --video_subdir "${VIDEO_SUBDIR}")

echo "=== [2/4] REMOVAL: hole (after object removal, before 3D inpaint) ==="
cd "${INPAINT_ROOT}"
python -u render.py \
  -s "${DATA}" \
  -m "${MODEL}" \
  --iteration _object_removal/iteration_2000 \
  --skip_train --skip_test \
  --render_video \
  "${RGB_ARGS[@]}" \
  2>&1 | tee "${LOGDIR}/video_rgb_after_removal_2000.log"

echo "=== [3/4] AFTER: inpaint finetune 5k ==="
python -u render.py \
  -s "${DATA}" \
  -m "${MODEL}" \
  --iteration _object_inpaint_virtual/iteration_5000 \
  --skip_train --skip_test \
  --render_video \
  "${RGB_ARGS[@]}" \
  2>&1 | tee "${LOGDIR}/video_rgb_after_inpaint_5k.log"

echo "=== [4/4] AFTER: inpaint finetune 20k ==="
python -u render.py \
  -s "${DATA}" \
  -m "${MODEL}" \
  --iteration _object_inpaint_virtual/iteration_20000 \
  --skip_train --skip_test \
  --render_video \
  "${RGB_ARGS[@]}" \
  2>&1 | tee "${LOGDIR}/video_rgb_after_inpaint_20k.log"

echo ""
echo "Done. RGB orbit videos (compare these — same framing, only content differs):"
echo "  1) BEFORE:   ${GSOUT}/${VIDEO_SUBDIR}/ours_30000/final_video.mp4"
echo "  2) REMOVAL:  ${MODEL}/${VIDEO_SUBDIR}/ours__object_removal/iteration_2000/final_video.mp4"
echo "  3) INPAINT5k:  ${MODEL}/${VIDEO_SUBDIR}/ours__object_inpaint_virtual/iteration_5000/final_video.mp4"
echo "  4) INPAINT20k: ${MODEL}/${VIDEO_SUBDIR}/ours__object_inpaint_virtual/iteration_20000/final_video.mp4"
