#!/usr/bin/env bash
# Single-invocation SAM3-direct virtual orbit pipeline:
#   1) Train vanilla 3DGS to iteration_30000 if missing (matches prune default).
#   2) Copy point_cloud/iteration_30000 -> iteration_2000 if missing (LaMa virtual paths use iteration_2000).
#   3) Run tools/sam3_direct/run_sam3_direct_pipeline.py (SAM3 train+virtual, LaMa, optional depth fix, stage 10–11).
#
# Usage (repo root):
#   bash tools/run_sam3_direct_one_shot.sh cone_red 8 "cone" --circle-radius 1.0
# Extra args are forwarded to run_sam3_direct_pipeline.py (e.g. --sam3-root, --finetune-iters).
#
# Prerequisites:
#   data/inpaint360/<SCENE>/ with COLMAP sparse/ and images_<RESOLUTION>/
#   external/Inpaint360GS/config/object_removal/inpaint360/<SCENE>.json
#   external/Inpaint360GS/config/object_inpaint/inpaint360/<SCENE>.json
# (copy from car.json if needed.)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="${1:?usage: $0 SCENE RESOLUTION PROMPT [extra args for run_sam3_direct_pipeline.py...]}"
RES="${2:?resolution (must match images_<r>, e.g. 8 for images_8)}"
PROMPT="${3:?SAM3 text prompt}"
shift 3 || true

cd "$ROOT"
export PYTHONPATH="$ROOT/external/Inpaint360GS:$ROOT/external/Inpaint360GS/gaussian_splatting:${PYTHONPATH:-}"

GS_OUT="$ROOT/output/inpaint360/$SCENE/3dgs_output"
PLY30="$GS_OUT/point_cloud/iteration_30000/point_cloud.ply"
PC2000="$GS_OUT/point_cloud/iteration_2000"

DATA="$ROOT/data/inpaint360/$SCENE"
IMGDIR="$DATA/images_$RES"
for f in "$ROOT/external/Inpaint360GS/config/object_removal/inpaint360/${SCENE}.json" \
         "$ROOT/external/Inpaint360GS/config/object_inpaint/inpaint360/${SCENE}.json"; do
  if [[ ! -f "$f" ]]; then
    echo "WARN: missing $f — copy from car.json (or another template) before fusion/inpaint." >&2
  fi
done
if [[ ! -d "$IMGDIR" ]]; then
  echo "ERROR: expected training images at $IMGDIR" >&2
  exit 1
fi

if [[ ! -f "$PLY30" ]]; then
  echo "[one-shot] Training vanilla 3DGS -> $PLY30"
  ( cd "$ROOT/external/Inpaint360GS" && python gaussian_splatting/train.py \
      -s "$DATA" \
      -m "$GS_OUT" \
      --init_mode sparse \
      --eval \
      --resolution "$RES" )
else
  echo "[one-shot] Skip train (exists): $PLY30"
fi

if [[ ! -d "$PC2000" ]]; then
  echo "[one-shot] Shim: copy iteration_30000 -> iteration_2000 (LaMa / virtual paths)"
  cp -a "$GS_OUT/point_cloud/iteration_30000" "$PC2000"
else
  echo "[one-shot] Skip shim (exists): $PC2000"
fi

echo "[one-shot] SAM3 direct pipeline (LaMa + depth fix + stage 10–11)..."
exec python "$ROOT/tools/sam3_direct/run_sam3_direct_pipeline.py" \
  --scene "$SCENE" \
  --resolution "$RES" \
  --prompt "$PROMPT" \
  --run-depth-fix \
  --ring-width 8 \
  --finetune-iters 5000 \
  "$@"
