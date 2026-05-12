#!/usr/bin/env bash
# Full Inpaint360GS pipeline for one scene (all 11 stages via driver).
#
# Usage:
#   bash scripts/run_inpaint360gs_scene_full.sh car
#   FINETUNE_ITERS="5000" bash scripts/run_inpaint360gs_scene_full.sh cube
#   INPAINT360_CHECKPOINT_VIDEO_ITERS="5000 8000" FINETUNE_ITERS="12000" bash scripts/run_inpaint360gs_scene_full.sh bag
#
# Same as run_inpaint360gs_one_scene.sh but defaults FINETUNE_ITERS to a single 12000 budget
# unless you override FINETUNE_ITERS in the environment.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="${1:?Usage: $0 <scene_name>}"

export FINETUNE_ITERS="${FINETUNE_ITERS:-12000}"

exec bash "${ROOT}/scripts/run_inpaint360gs_one_scene.sh" "${SCENE}"
