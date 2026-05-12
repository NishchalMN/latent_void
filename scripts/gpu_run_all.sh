#!/bin/bash
# Run full Inpaint360GS pipeline on GPU node.
# All deps already installed from previous run.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="${ROOT}/external/Inpaint360GS:${ROOT}/external/Inpaint360GS/gaussian_splatting:${PYTHONPATH}"

echo "=== Starting pipeline at $(date) ==="
echo "Root: ${ROOT}"

SCENES="car bag cube garden_toys truck fruits"

python -u tools/run_inpaint360gs_full.py \
    --scenes ${SCENES} \
    --resolution 2 \
    --data-root "${ROOT}/data/inpaint360" \
    --output-root "${ROOT}/output/inpaint360" \
    2>&1

echo ""
echo "=== Pipeline complete at $(date) ==="

if [ -f "${ROOT}/tools/evaluate_inpaint_quality.py" ]; then
    python -u "${ROOT}/tools/evaluate_inpaint_quality.py" \
        --output-root "${ROOT}/output/inpaint360" \
        --data-root "${ROOT}/data/inpaint360" \
        --scenes ${SCENES} \
        2>&1 || echo "Evaluation had errors (non-fatal)"
fi

echo "=== All done at $(date) ==="
