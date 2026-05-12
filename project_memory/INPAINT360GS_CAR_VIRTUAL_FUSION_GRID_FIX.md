# Inpaint360GS (`car`): virtual fusion grid fix (verified 2026-05)

This note explains **what broke**, **what we changed**, and **why fusion + MeshLab looked good afterward** (see qualitative screenshot `good fix.png` at repo root: coplanar inpainted patch on the pavement instead of a tilted / floating slab).

## What was wrong

Two separate issues showed up on the `car` virtual orbit path:

### A) Resolution mismatch: `depth_completed` vs renderer `depth`

LaMa / `prepare_lama_data` sometimes wrote `depth_completed/*.npy` at a **different H×W** than the pruned virtual render depth in `virtual/.../depth/*.npy` (e.g. `(1005, 1789)` vs `(1079, 1920)`).

**Consequences:**

- **`tools/inpaint360_align_completed_depth.py`** and **`tools/inpaint360_project_completed_to_hole_plane.py`** skipped every frame (`shape mismatch`) until you manually resized—or until fusion fixed the grid.
- **`edit_object_removal_plyfusion.py`**: if fusion used full-res **RGB/mask** but **small-res** `depth_completed` for `create_point_cloud`, you got **different pixel counts** than the flattened mask →  
  `IndexError: boolean index did not match indexed array` inside `ply_color_fusion`  
  (classic: `1797945` points vs `2071680` mask entries = `1005×1789` vs `1079×1920`).

### B) Stale / duplicate fusion scripts

An older **`tools/edit_object_removal_plyfusion.py`** fused **without** resizing inpaint RGB/mask to the depth grid; line numbers in tracebacks could match that legacy path. Always run fusion from **`external/Inpaint360GS/edit_object_removal_plyfusion.py`** (or a copy synced from `upstream_overrides/inpaint360gs/`).

## What we changed in code (canonical behavior)

Tracked mirrors + `external/Inpaint360GS/` (after sync / install overrides):

1. **`edit_object_removal_plyfusion.py`**
   - **`_squeeze_depth_2d`**: depth `.npy` may be `HxW`, `1xHxW`, or `HxWx1`; fusion requires strict **2D** `HxW`.
   - For each virtual frame, load **`depth_hole`** first, load **`depth_completed`**, then if shapes differ **resize `depth_completed` to `depth_hole` with bilinear interpolation** so the depth grid matches the **renderer** intrinsics / `renders` / `depth`.
   - Then **`_align_image_and_mask_to_depth`**: resize inpaint RGB + virtual mask to that same grid.
   - **Assert** `len(points) == len(colors) == len(mask)` before `ply_color_fusion` with a clear error if not.

2. **`utils/point_utils.py` → `ply_color_fusion`**
   - Raises an explicit **`ValueError`** on length mismatch instead of a NumPy boolean-index crash.

3. **`tools/edit_object_removal_plyfusion.py`**
   - Replaced legacy copy so accidental imports do not resurrect the old behavior.

After a successful run you may see logs like:

```text
[fuse] resized depth_completed -> hole grid 1079x1920 for view 00000
```

Optional: also run the **depth ring align + plane projection** (below) for best **world-space** agreement with the pavement; fusion grid fix alone fixes **count/projection consistency**; plane fit fixes **height/tilt**.

## Recipe that worked for `car` (order)

1. Ensure `depth_completed` and `depth` are the **same shape** (manual resize script if needed, or rely on fusion resize + later re-export of depth for other tools).

2. **Ring affine align** (only works when shapes match):

```bash
python tools/inpaint360_align_completed_depth.py \
  --completed-dir output/inpaint360/car/virtual/ours_object_removal/iteration_2000/depth_completed \
  --hole-dir output/inpaint360/car/virtual/ours_object_removal/iteration_2000/depth \
  --mask-dir data/inpaint360/car/inpaint_2d_unseen_mask_virtual \
  --backup \
  --ring-width 8
```

3. **Planar hole overwrite** on masked `depth_completed`:

```bash
python tools/inpaint360_project_completed_to_hole_plane.py \
  --completed-dir output/inpaint360/car/virtual/ours_object_removal/iteration_2000/depth_completed \
  --hole-dir output/inpaint360/car/virtual/ours_object_removal/iteration_2000/depth \
  --mask-dir data/inpaint360/car/inpaint_2d_unseen_mask_virtual \
  --backup \
  --ring-width 8
```

4. **Fusion** (uses fixed script):

```bash
cd external/Inpaint360GS
python edit_object_removal_plyfusion.py \
  -s ../../data/inpaint360/car \
  -m ../../output/inpaint360/car \
  --config_file config/object_removal/inpaint360/car.json
```

5. Inspect **`fused_mask_col_dep_ply/*.ply`** in MeshLab — expect patch **on the same plane** as the cobbles (reference: `good fix.png`).

6. Stage 10b: **`edit_object_inpaint.py`** with a good **`--supp_ply`** chosen from the new `fused_mask_col_dep_ply/`; consider `INPAINT360_KEEP_FULL_OPTIMIZED=0` if the full scene collapse reappears.

## Syncing patched Inpaint360GS on cluster nodes

`external/` is gitignored. From a clean checkout:

```bash
python3 tools/sync_inpaint360gs_upstream_overrides.py
# and/or
bash scripts/install_inpaint360gs_overrides.sh
```

## Related docs

- Bag virtual depth + fusion recipe: `project_memory/INPAINT360GS_BAG_VIRTUAL_DEPTH_FUSION_RECIPE.md`
- SAM3-direct virtual path: `project_memory/INPAINT360GS_SAM3_DIRECT_GUIDE.md`
