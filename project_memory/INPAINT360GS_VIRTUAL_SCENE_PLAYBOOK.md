# Inpaint360GS virtual orbit playbook (any scene)

Use this as the **single ordered checklist** for **SAM3-direct → prune → virtual orbit → LaMa → depth sanity → fusion → 3D inpaint**, on **any** scene (not tied to `car` / `cube`). Replace placeholders consistently:

| Placeholder | Meaning |
|-------------|---------|
| `PROJECT` | Absolute path to `latent_void` checkout |
| `SCENE` | Dataset folder name under `data/inpaint360/` and `output/inpaint360/` |
| `MODEL_OUT` | Inpaint360GS **`-m`** root (often `PROJECT/output/inpaint360/SCENE`) |
| `BASE_GS` | Full vanilla **non-pruned** 3DGS output dir used for virtual orbit copy |
| `PRUNED_GS` | Pruned Gaussian splat output used for hole renders |

**Car caveat:** fusion geometry was fixed (see `INPAINT360GS_CAR_VIRTUAL_FUSION_GRID_FIX.md`), but **orbit RGB after stage 10b** may still collapse (thin strip / mostly black). That points at **finetune / Gaussians / orbit renderer**, not at fusion paths alone. Trying another scene is reasonable; keep this playbook identical and tune **mask quality**, **`supp_ply`**, and **`finetune_iteration`** per scene.

---

## 0. One-time: GPU checkout + Inpaint360GS patches

`external/Inpaint360GS` is **gitignored** on clones; refresh before serious runs:

```bash
cd PROJECT
python3 tools/sync_inpaint360gs_upstream_overrides.py
# or: bash scripts/install_inpaint360gs_overrides.sh
```

Always run Inpaint360GS heavy scripts with:

```bash
export PYTHONPATH=$PROJECT/external/Inpaint360GS:$PROJECT/external/Inpaint360GS/gaussian_splatting:$PYTHONPATH
```

---

## 1. Data + configs for `SCENE`

- **COLMAP scene:** `PROJECT/data/inpaint360/SCENE/` with `images/`, `sparse/`, etc. (same contract as upstream Inpaint360GS).
- **Output root:** `PROJECT/output/inpaint360/SCENE/` will hold `3dgs_output`, `point_cloud/iteration_2000`, `virtual/…`, LaMa ingest, etc.

**JSON configs** (copy from a working template such as `car`, then edit paths only if needed):

- `external/Inpaint360GS/config/object_removal/inpaint360/SCENE.json`
- `external/Inpaint360GS/config/object_inpaint/inpaint360/SCENE.json`

Key fields to sanity-check for **your** object layout: `select_obj_id`, `circle_radius`, `finetune_iteration`, `removal_thresh`.

---

## 2. Train base 3DGS (full scene)

Train or reuse **`MODEL_OUT/3dgs_output`** (vanilla GS checkpoint used as **`BASE_GS`** below). This must match whatever path you pass as `--base-model` for orbit rendering.

---

## 3. SAM3 masks on **training** views + prune → pruned GS

Use your established SAM3-direct flow (`tools/run_sam3_multiview.py`, `tools/prune_3dgs_with_inpaint360gs_masks.py`, or `tools/sam3_direct/run_sam3_direct_pipeline.py`). End state:

- A **pruned** Gaussian checkpoint directory → set **`PRUNED_GS`** to that tree’s **`.../3dgs_output`** (or equivalent).

---

## 4. Virtual orbits (paired full vs hole)

From **`PROJECT`**:

```bash
export PYTHONPATH=$PROJECT/external/Inpaint360GS:$PROJECT/external/Inpaint360GS/gaussian_splatting:$PYTHONPATH

python tools/render_inpaint360_virtual_orbits.py \
  --scene SCENE \
  --base-model PROJECT/output/inpaint360/SCENE/3dgs_output \
  --pruned-model PROJECT/output/inpaint360/PRUNED_REL_PATH/3dgs_output
```

Expected layout:

- `MODEL_OUT/virtual/ours_2000/renders/` (+ `depth/`) — **full** model  
- `MODEL_OUT/virtual/ours_object_removal/iteration_2000/renders/` (+ `depth/`) — **pruned** hole  

If smear at frame borders is severe, re-run orbit with a smaller `--circle-radius` (script flag if exposed) so the object stays more centered.

---

## 5. SAM3 on **virtual** RGB (full branch only)

Build manifest listing **`MODEL_OUT/virtual/ours_2000/renders/*.png`**, run **`tools/run_sam3_multiview.py`** with your prompt, export masks to:

`PROJECT/data/inpaint360/SCENE/inpaint_2d_unseen_mask_virtual/00000.png` … **same stems as renders**.

Convert `.npy` → PNG with stems aligned to **`sam3_results.json`** if using the repo helper (`run_sam3_direct_pipeline._npy_masks_to_png_255`).

---

## 6. Hole hardening (optional, recommended if hole fringe is muddy)

Black fringe outside a tight SAM mask breaks LaMa unless masks expand with RGB:

```bash
cd PROJECT
python tools/harden_pruned_virtual_renders.py --scene SCENE --backup \
  --diff-vs-full --diff-edge-band 60 --diff-max-pruned-luma 155 \
  --dark-fringe-luma 105 --dark-fringe-iters 20 \
  --update-masks --backup-masks
```

Tune `--diff-edge-band` / luma; avoid huge `--diff-near-dilate` disks without `--diff-edge-band`.

---

## 7. LaMa color + depth (**cwd matters**)

```bash
cd PROJECT/external/Inpaint360GS

python tools/prepare_lama_data.py \
  -s ../../data/inpaint360/SCENE -m ../../output/inpaint360/SCENE -r 2 --inpaint2lama

( cd LaMa && TORCH_HOME=$PWD PYTHONPATH=$PWD python bin/predict_color.py --data_name <DATA_NAME> )
( cd LaMa && TORCH_HOME=$PWD PYTHONPATH=$PWD python bin/predict_depth.py --data_name <DATA_NAME> )

python tools/prepare_lama_data.py \
  -s ../../data/inpaint360/SCENE -m ../../output/inpaint360/SCENE -r 2
```

`<DATA_NAME>` must match what **`prepare_lama_data.py --inpaint2lama`** registered (often like `360_car_virtual`; check LaMa `bin/` defaults or the folder created under `LaMa/output/`).

After this: **`data/.../images_inpaint_unseen_virtual/`** holds LaMa RGB; **`MODEL_OUT/virtual/.../depth_completed/`** holds filled depth.

---

## 8. Depth grid sanity (avoid silent skips + bad fusion)

**Problem:** `depth_completed/*.npy` can be **different H×W** than **`virtual/.../depth/*.npy`** (renderer). Then ring-align scripts skip; old fusion crashed on mask length mismatch.

**Fix workflow:**

1. If `inpaint360_align_completed_depth.py` prints `shape mismatch` for every frame, resize **`depth_completed`** to match **`depth`** per stem (bilinear on depth values), **then** run align + plane scripts.

2. Ring affine align (writes mainly **inside mask**):

```bash
cd PROJECT
python tools/inpaint360_align_completed_depth.py \
  --completed-dir output/inpaint360/SCENE/virtual/ours_object_removal/iteration_2000/depth_completed \
  --hole-dir output/inpaint360/SCENE/virtual/ours_object_removal/iteration_2000/depth \
  --mask-dir data/inpaint360/SCENE/inpaint_2d_unseen_mask_virtual \
  --backup --ring-width 8
```

3. Planar projection on masked region (uses hole depth ring):

```bash
python tools/inpaint360_project_completed_to_hole_plane.py \
  --completed-dir output/inpaint360/SCENE/virtual/ours_object_removal/iteration_2000/depth_completed \
  --hole-dir output/inpaint360/SCENE/virtual/ours_object_removal/iteration_2000/depth \
  --mask-dir data/inpaint360/SCENE/inpaint_2d_unseen_mask_virtual \
  --backup --ring-width 8
```

Expect **`updated=30`** (or your orbit frame count), not all skipped.

---

## 9. PLY fusion (stage 10a)

**Always** run the copy inside **`external/Inpaint360GS`** (patched fusion resizes `depth_completed` → hole grid and aligns RGB/mask):

```bash
cd PROJECT/external/Inpaint360GS
python edit_object_removal_plyfusion.py \
  -s ../../data/inpaint360/SCENE \
  -m ../../output/inpaint360/SCENE \
  --config_file config/object_removal/inpaint360/SCENE.json
```

**Sanity (MeshLab):** compare same stem:

- `MODEL_OUT/virtual/ours_object_removal/iteration_2000/fused_mask_col_dep_ply/XXXXX.ply`
- `.../fused_hole_col_dep_ply/XXXXX.ply`  

Inpainted patch should sit **on** the support surface, not a floating tilted sheet.

---

## 10. 3D inpaint finetune (stage 10b)

Pick **`supp_ply`** from **`fused_mask_col_dep_ply/`** (flat, compact stem).

```bash
cd PROJECT/external/Inpaint360GS
export PYTHONPATH=$PROJECT/external/Inpaint360GS:$PROJECT/external/Inpaint360GS/gaussian_splatting:$PYTHONPATH

# Often helps when full optimized scene collapses:
export INPAINT360_KEEP_FULL_OPTIMIZED=0

python edit_object_inpaint.py \
  -s ../../data/inpaint360/SCENE \
  -m ../../output/inpaint360/SCENE \
  --config_file config/object_inpaint/inpaint360/SCENE.json \
  --resolution 2 \
  --supp_ply ../../output/inpaint360/SCENE/virtual/ours_object_removal/iteration_2000/fused_mask_col_dep_ply/STEM.ply \
  --render_video \
  --skip_train --skip_test
```

- **`--skip_train --skip_test`** avoids misleading **black** tiles on real cameras when supervision is **virtual-only**. Judge **`video/`** (orbit) first.
- **`object_inpaint/.../SCENE.json`**: lower **`finetune_iteration`** (e.g. 2000–3000) if densify destroys the scene.
- Orbit PNGs may be **RGB | semantics** side-by-side; interpret **left half** as RGB (see `external/Inpaint360GS/render.py` → `render_video_func_wriva`). For clean RGB-only orbit, use **`render.py --render_video --video_rgb_only`** against the saved inpaint checkpoint if needed.

**Checkpoint path:**

`MODEL_OUT/point_cloud_object_inpaint_virtual/iteration_<N>/point_cloud.ply`

---

## 11. Metrics (optional)

```bash
cd PROJECT
python tools/run_inpaint360gs_full.py \
  --scenes SCENE \
  --resolution 2 \
  --start-stage 11 \
  --skip-seg \
  --data-root data/inpaint360 \
  --output-root output/inpaint360
```

Add **`--skip-fid-eval`** if FID fails offline.

---

## 12. Related docs

| Doc | Role |
|-----|------|
| `project_memory/INPAINT360GS_SAM3_DIRECT_GUIDE.md` | SAM3 virtual + LaMa ordering details |
| `project_memory/INPAINT360GS_CAR_VIRTUAL_FUSION_GRID_FIX.md` | Depth vs hole resolution + fusion code behavior |
| `project_memory/INPAINT360GS_BAG_VIRTUAL_DEPTH_FUSION_RECIPE.md` | Bag-specific depth ring / plane intuition |

---

## Why dropping `car` can still follow this doc

The playbook is **scene-agnostic**. For a new scene you mainly change **`SCENE`**, **`PRUNED_GS`**, prompts, and JSON knobs. If orbit video stays broken after fusion looks good in MeshLab, treat it as a **stage 10b / optimization** research issue, not a missing LaMa step.
