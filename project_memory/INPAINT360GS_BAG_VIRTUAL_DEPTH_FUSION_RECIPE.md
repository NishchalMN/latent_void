# Inpaint360GS (`bag`): virtual fusion depth fixes (verified 2026-05-06)

This documents what finally made the fused patch align with the table and stop floating in MeshLab (`fused_mask_col_dep_ply` vs `fused_hole_col_dep_ply`). Use this recipe after LaMa (stage 9) and **before** or instead of blindly re-training 3D inpaint.

## Why `git status` never shows `external/Inpaint360GS/`

The directory `external/` is in `.gitignore`. Patched scripts are tracked under `upstream_overrides/inpaint360gs/` plus `scripts/install_inpaint360gs_overrides.sh`. Refresh copies with:

```bash
python3 tools/sync_inpaint360gs_upstream_overrides.py
```

## Symptom checklist

- Fused masked PLY and hole PLY disagree in world space (offset / tilt).
- Patch looks plausible from one orbit angle only (sheet geometry).
- Spikes/noise in fused clouds (often mask fragmentation; see optional mask filter).

## Order of operations (recommended)

Replace `PROJECT` with your latent_void checkout path. Scene `bag`, removal iter `iteration_2000` as in upstream layout.

### 1) Affine align `depth_completed` to renderer `depth` (hole path)

Uses a ring outside the inpaint mask; fits `z_hole ≈ a * z_completed + b` robustly; **writes only masked pixels** in `depth_completed`.

```bash
cd PROJECT
python tools/inpaint360_align_completed_depth.py \
  --completed-dir output/inpaint360/bag/virtual/ours_object_removal/iteration_2000/depth_completed \
  --hole-dir output/inpaint360/bag/virtual/ours_object_removal/iteration_2000/depth \
  --mask-dir data/inpaint360/bag/inpaint_2d_unseen_mask_virtual \
  --backup \
  --ring-width 5
```

### 2) Planar projection of masked depths (fixes “correct from one angle only”)

Fit `z = a*x + b*y + c` on **hole depth** (`depth/`) in a ring outside the mask, then overwrite **masked** `depth_completed` with that planar surface per pixel — consistent with fusion’s pinhole convention.

```bash
python tools/inpaint360_project_completed_to_hole_plane.py \
  --completed-dir output/inpaint360/bag/virtual/ours_object_removal/iteration_2000/depth_completed \
  --hole-dir output/inpaint360/bag/virtual/ours_object_removal/iteration_2000/depth \
  --mask-dir data/inpaint360/bag/inpaint_2d_unseen_mask_virtual \
  --backup \
  --ring-width 5
```

If ring samples are scarce, bump `--ring-width` to `8`.

### 3) (Optional) Virtual mask cleanup

If fusion still has spikes/islands:

```bash
python tools/inpaint360_filter_virtual_masks.py \
  --mask-dir data/inpaint360/bag/inpaint_2d_unseen_mask_virtual \
  --backup-dir data/inpaint360/bag/inpaint_2d_unseen_mask_virtual_backup \
  --min-area 1200 \
  --open-kernel 3 \
  --erode-kernel 2 \
  --erode-iters 1
```

Restore from `--backup-dir` if too aggressive.

### 4) PLY fusion (stage 10a)

Default fusion uses **`world_view_transform`** for `c2w` (renderer-consistent). Legacy R/T reconstruction is `--legacy_pose_rt` for debugging only.

```bash
cd PROJECT/external/Inpaint360GS
python edit_object_removal_plyfusion.py \
  -s ../../data/inpaint360/bag \
  -m ../../output/inpaint360/bag \
  --config_file config/object_removal/inpaint360/bag.json
```

**Sanity paths (full paths, `bag` example):**

- Masked fused (LaMa depth in mask):  
  `output/inpaint360/bag/virtual/ours_object_removal/iteration_2000/fused_mask_col_dep_ply/`
- Hole fused (removal renderer depth):  
  `output/inpaint360/bag/virtual/ours_object_removal/iteration_2000/fused_hole_col_dep_ply/`

Inspect matching stems (e.g. `00004.ply`) in MeshLab; they should overlap after steps 1–2.

### 5) 3D inpaint (stage 10b)

`edit_object_inpaint.py` auto-picks `supp_ply` from flatness score over `fused_mask_col_dep_ply/*.ply`, or override:

```bash
python edit_object_inpaint.py \
  -s ../../data/inpaint360/bag \
  -m ../../output/inpaint360/bag \
  --config_file config/object_inpaint/inpaint360/bag.json \
  --resolution 2 \
  --render_video
# Optional:
# --supp_ply ../../output/inpaint360/bag/virtual/ours_object_removal/iteration_2000/fused_mask_col_dep_ply/00004.ply
```

## Related tools (legacy / exploratory)

- `tools/inpaint360_repair_virtual_depth.py`: optional fills (`inpaint`, `median_ring`, `plane_ring`) on `depth_completed` alone; planar hole fix above is preferred when hole+mask alignment matters.

## Repo code changes tied to this recipe

| File | Role |
|------|------|
| `tools/inpaint360_align_completed_depth.py` | Affine align completed ↔ hole depth on ring |
| `tools/inpaint360_project_completed_to_hole_plane.py` | Planar masked depth from hole-depth ring |
| `tools/inpaint360_filter_virtual_masks.py` | Morphological cleanup of virtual masks |
| `tools/sync_inpaint360gs_upstream_overrides.py` | Copy patched Inpaint360GS files from `external/` into `upstream_overrides/` for git |
| `upstream_overrides/inpaint360gs/edit_object_removal_plyfusion.py` | Tracked mirror: `c2w` from `world_view_transform`; `--legacy_pose_rt` |
| `upstream_overrides/inpaint360gs/edit_object_inpaint.py` | Tracked mirror: `auto_select_support_ply` + `--supp_ply` |
| `scripts/install_inpaint360gs_overrides.sh` | Install tracked mirrors into `external/Inpaint360GS/` on GPU nodes |
| `external/Inpaint360GS/*.py` | **Not in git** — install from `upstream_overrides/` after each pull |

## Applying to other scenes

Swap `bag` paths for `car`, `cube`, etc. Confirm `iteration_*` under `virtual/ours_object_removal/` matches your removal checkpoint.
