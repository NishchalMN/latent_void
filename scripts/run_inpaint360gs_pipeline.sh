#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${ROOT}/.venvs/latent_void_py310"
INPAINT_ROOT="${ROOT}/external/Inpaint360GS"
DATA_ROOT="${ROOT}/data/inpaint360"
OUTPUT_ROOT="${ROOT}/output/inpaint360"

export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

source "${VENV}/bin/activate"

DATASET_NAME="inpaint360"
RESOLUTION="${RESOLUTION:-2}"
SCENES="${SCENES:-car bag cube garden_toys truck fruits}"

echo "=============================================="
echo "  Inpaint360GS Full Pipeline"
echo "  Scenes: ${SCENES}"
echo "  Resolution: ${RESOLUTION}"
echo "=============================================="

cd "${INPAINT_ROOT}"
export PYTHONPATH="${INPAINT_ROOT}:${INPAINT_ROOT}/gaussian_splatting:${INPAINT_ROOT}/seg/detectron2:${PYTHONPATH}"

for SCENE in ${SCENES}; do
    echo ""
    echo "================================================="
    echo "  Processing scene: ${SCENE}"
    echo "================================================="

    SCENE_DATA="${DATA_ROOT}/${SCENE}"
    SCENE_OUTPUT="${OUTPUT_ROOT}/${SCENE}"

    if [ ! -d "${SCENE_DATA}" ]; then
        echo "ERROR: Scene data not found: ${SCENE_DATA}"
        continue
    fi

    # ----------------------------------------------------------
    # Stage 1: Train vanilla 3DGS
    # ----------------------------------------------------------
    GS_OUTPUT="${SCENE_OUTPUT}/3dgs_output"
    if [ -f "${GS_OUTPUT}/point_cloud/iteration_30000/point_cloud.ply" ]; then
        echo "[Stage 1] 3DGS already trained for ${SCENE}, skipping."
    else
        echo "[Stage 1] Training vanilla 3DGS for ${SCENE} (30k iterations, resolution=${RESOLUTION})..."
        python gaussian_splatting/train.py \
            -s "${SCENE_DATA}" \
            -m "${GS_OUTPUT}" \
            --init_mode "sparse" \
            --eval \
            --resolution ${RESOLUTION} \
            2>&1 | tee "${SCENE_OUTPUT}/train_3dgs.log"
        echo "[Stage 1] 3DGS training complete for ${SCENE}."
    fi

    # ----------------------------------------------------------
    # Stage 2: 2D segmentation with HQ-SAM
    # ----------------------------------------------------------
    MASK_DIR="${SCENE_DATA}/images_${RESOLUTION}_num"
    if [ -d "${MASK_DIR}" ] && [ "$(ls -A ${MASK_DIR} 2>/dev/null | head -1)" != "" ]; then
        echo "[Stage 2] HQ-SAM masks already exist for ${SCENE}, skipping."
    else
        echo "[Stage 2] Running HQ-SAM segmentation for ${SCENE}..."
        python seg/raw_mask_sam.py \
            --dataset_path "${DATA_ROOT}/" \
            --scene_name "${SCENE}" \
            --image_folder "images_${RESOLUTION}" \
            --method hqsam \
            2>&1 | tee "${SCENE_OUTPUT}/seg_hqsam.log"
        echo "[Stage 2] HQ-SAM segmentation complete for ${SCENE}."
    fi

    # ----------------------------------------------------------
    # Stage 3: 3D mask association
    # ----------------------------------------------------------
    ASSOC_DIR="${SCENE_DATA}/associated_hqsam"
    if [ -d "${ASSOC_DIR}" ] && [ "$(ls -A ${ASSOC_DIR} 2>/dev/null | head -1)" != "" ]; then
        echo "[Stage 3] 3D mask association already done for ${SCENE}, skipping."
    else
        echo "[Stage 3] Running 3D mask association for ${SCENE}..."
        python seg/mask_associate.py \
            --source_path "${SCENE_DATA}" \
            --model_path "${GS_OUTPUT}" \
            --resolution ${RESOLUTION} \
            --mask_generator hqsam \
            --eval \
            2>&1 | tee "${SCENE_OUTPUT}/mask_associate.log"
        echo "[Stage 3] 3D mask association complete for ${SCENE}."
    fi

    # ----------------------------------------------------------
    # Stage 4: Add label numbers
    # ----------------------------------------------------------
    echo "[Stage 4] Adding label numbers for ${SCENE}..."
    python tools/add_label_num_hqsam.py \
        --source_path "${SCENE_DATA}" \
        --resolution ${RESOLUTION} \
        --mask_generator hqsam \
        2>&1 | tee "${SCENE_OUTPUT}/add_labels.log" || true

    # ----------------------------------------------------------
    # Stage 5: Semantic distillation
    # ----------------------------------------------------------
    DISTILL_CHECK="${SCENE_OUTPUT}/point_cloud"
    if [ -d "${DISTILL_CHECK}" ]; then
        echo "[Stage 5] Semantic distillation already done for ${SCENE}, skipping."
    else
        echo "[Stage 5] Running semantic distillation for ${SCENE}..."
        python seg/distillation.py \
            --source_path "${SCENE_DATA}" \
            --model_path "${SCENE_OUTPUT}" \
            --vanilla_3dgs_path "${GS_OUTPUT}" \
            --resolution ${RESOLUTION} \
            --object_path "associated_hqsam" \
            --eval \
            2>&1 | tee "${SCENE_OUTPUT}/distillation.log"
        echo "[Stage 5] Semantic distillation complete for ${SCENE}."
    fi

    echo "[DONE] Segmentation pipeline complete for ${SCENE}."
    echo "  Next: run object removal and inpainting (separate stages)."
    echo ""
done

echo "=============================================="
echo "  All scenes processed through segmentation."
echo "  Run scripts/run_inpaint360gs_remove_inpaint.sh next."
echo "=============================================="
