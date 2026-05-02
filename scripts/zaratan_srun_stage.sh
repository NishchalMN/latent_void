#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/zaratan_srun_stage.sh STAGE CONFIG [latent_void args...]

Stages:
  geometry      Run Marigold depth/normal/coordinate preprocessing.
  reconstruct  Run DiffSplat GSRecon/GSVAE export.
  segment      Run SAM 3 multi-view segmentation.
  finish       Run fuse, latent inpaint, and render diagnostics.
  run          Run the full pipeline end to end.

Examples:
  scripts/zaratan_srun_stage.sh geometry configs/zaratan_inpaint360gs_bag.yaml \
    --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_srun_h100

  SLURM_TIME=02:00:00 scripts/zaratan_srun_stage.sh finish configs/zaratan_inpaint360gs_bag.yaml \
    --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_srun_h100

Environment overrides:
  SLURM_ACCOUNT          default: msml612pcs3-class
  SLURM_PARTITION        default: gpu-h100
  SLURM_GRES             default: gpu:h100:1
  SLURM_TIME             default: stage-specific
  SLURM_CPUS_PER_TASK    default: 8
  SLURM_MEM              default: 64G
  LATENT_VOID_VENV       default: .venvs/latent_void_py310
  LATENT_VOID_MODULE     default: python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

STAGE="$1"
CONFIG="$2"
shift 2

case "$STAGE" in
  geometry) DEFAULT_TIME="01:00:00" ;;
  reconstruct) DEFAULT_TIME="01:00:00" ;;
  segment) DEFAULT_TIME="01:00:00" ;;
  finish) DEFAULT_TIME="02:00:00" ;;
  run) DEFAULT_TIME="04:00:00" ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    usage >&2
    exit 2
    ;;
esac

ACCOUNT="${SLURM_ACCOUNT:-msml612pcs3-class}"
PARTITION="${SLURM_PARTITION:-gpu-h100}"
GRES="${SLURM_GRES:-gpu:h100:1}"
TIME_LIMIT="${SLURM_TIME:-$DEFAULT_TIME}"
CPUS="${SLURM_CPUS_PER_TASK:-8}"
MEM="${SLURM_MEM:-64G}"
VENV="${LATENT_VOID_VENV:-.venvs/latent_void_py310}"
PY_MODULE="${LATENT_VOID_MODULE:-python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2}"

exec srun \
  --job-name="latent-void-${STAGE}-srun" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --time="$TIME_LIMIT" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  bash -lc '
set -euo pipefail

stage="$1"
config="$2"
venv="$3"
py_module="$4"
shift 4

echo "[latent_void] srun stage=${stage} host=$(hostname)"
module load "$py_module"
unset PYTHONPATH
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
source "$venv/bin/activate"

case "$stage" in
  geometry)
    exec python -m latent_void prepare-geometry --config "$config" "$@"
    ;;
  reconstruct)
    exec python -m latent_void reconstruct --config "$config" "$@"
    ;;
  segment)
    exec python -m latent_void segment --config "$config" "$@"
    ;;
  finish)
    exec python -m latent_void run --config "$config" --skip-geometry --skip-reconstruct --skip-segment "$@"
    ;;
  run)
    exec python -m latent_void run --config "$config" "$@"
    ;;
esac
' _ "$STAGE" "$CONFIG" "$VENV" "$PY_MODULE" "$@"
