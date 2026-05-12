# Inpaint360GS: full pipeline runbook (latent_void driver)

This document answers **how to run all 11 stages**, **what was fixed for `bag` virtual fusion**, **how to compare 5k vs 12k without destroying the background**, and **what is left to do**. For depth/fusion detail only, see `INPAINT360GS_BAG_VIRTUAL_DEPTH_FUSION_RECIPE.md`.

---

## The eleven stages (conceptual)

| Stage | What | Typical artifact / check |
|-------|------|---------------------------|
| 1 | Train vanilla 3DGS | `output/inpaint360/<scene>/3dgs_output/` |
| 2 | SAM segmentation | `data/inpaint360/<scene>/raw_sam/` |
| 3 | 3D mask association | logs |
| 4 | Label numbering | `data/inpaint360/<scene>/images_<r>_num/` |
| 5 | Semantic distillation | outputs under scene dir |
| 6 | Target ID → JSON configs | `external/Inpaint360GS/config/object_*/*/ <scene>.json` |
| 7 | Object removal | `virtual/ours_object_removal/iteration_*/renders/` |
| 8 | Virtual trajectory | virtual poses |
| 9 | LaMa color + depth | `depth_completed/`, inpainted RGB |
| 10a | PLY fusion | `fused_mask_col_dep_ply/`, `fused_hole_col_dep_ply/` |
| 10b | 3D Gaussian inpaint finetune | `point_cloud_object_inpaint_virtual/iteration_*` |
| 11 | Metrics | `inpaint_evaluation_results.json` |

---

## Single command: full pipeline for one scene (`car`, `cube`, `bag`, …)

From repo root, GPU node, venv activated (same as `run_inpaint360gs_one_scene.sh`):

```bash
bash scripts/run_inpaint360gs_one_scene.sh car
```

Defaults:

- Data: `data/inpaint360/<scene>/`
- Output: `output/inpaint360/<scene>/`
- Three inpaint budgets: `FINETUNE_ITERS="${FINETUNE_ITERS:-5000 12000 20000}"`

**One budget only (e.g. stable background, shorter train):**

```bash
FINETUNE_ITERS="5000" bash scripts/run_inpaint360gs_one_scene.sh car
```

**Alias wrapper (same behavior, explicit name):**

```bash
bash scripts/run_inpaint360gs_scene_full.sh cube
```

Optional env (documented in `scripts/run_inpaint360gs_one_scene.sh`):

| Variable | Meaning |
|----------|---------|
| `RESOLUTION` | e.g. `2` (default) |
| `FINETUNE_ITERS` | Space-separated list passed to `--finetune-iterations` |
| `START_STAGE` | `1`–`11`; e.g. `10` = fusion + inpaint + eval only |
| `SKIP_SEG=1` | Skip stages 1–5 if already done |
| `SKIP_FID=0` | Enable FID in eval (needs network / cached refs) |
| `INPAINT360_CHECKPOINT_VIDEO_ITERS` | Mid-train videos (see below) |

Direct driver (multi-scene):

```bash
python tools/run_inpaint360gs_full.py \
  --scenes car cube bag \
  --resolution 2 \
  --finetune-iterations 5000 12000 \
  --skip-fid-eval \
  --data-root data/inpaint360 \
  --output-root output/inpaint360
```

---

## Mid-training video while finetuning to 12k

Upstream `edit_object_inpaint.py` only saved **once** at the end; long runs could fix the hole but hurt the surround.

**Now supported:** save PLY + classifier and render an orbit **video** at chosen step counts **during** the same run.

### CLI (manual)

```bash
cd external/Inpaint360GS
python edit_object_inpaint.py \
  -s ../../data/inpaint360/bag \
  -m ../../output/inpaint360/bag \
  --config_file config/object_inpaint/inpaint360/bag.json \
  --resolution 2 \
  --render_video \
  --checkpoint-video-iters 5000 8000
```

With `finetune_iteration: 12000` in JSON, you get checkpoints (and videos under `video/...`) at **5000** and **8000**, plus the usual **12000** save + final video.

### Env (used by `tools/run_inpaint360gs_full.py` stage 10b)

```bash
export INPAINT360_CHECKPOINT_VIDEO_ITERS="5000 8000"
FINETUNE_ITERS="12000" bash scripts/run_inpaint360gs_one_scene.sh bag
```

### Alternative: two separate budgets (no code path change)

```bash
FINETUNE_ITERS="5000 12000" bash scripts/run_inpaint360gs_one_scene.sh bag
```

Runs stage **10b twice** from the same fusion (each full finetune from scratch). Heavier than `--checkpoint-video-iters`, but gives isolated `iteration_5000` and `iteration_12000` trees as today.

---

## 5k only: will the hole fill after depth fixes?

Often **yes** for table-aligned holes:

1. Run the **virtual depth recipe** (affine align + planar hole projection) before fusion — see `INPAINT360GS_BAG_VIRTUAL_DEPTH_FUSION_RECIPE.md`.
2. Use **`FINETUNE_ITERS="5000"`** or keep **`INPAINT360_KEEP_FULL_OPTIMIZED=1`** (default in patched inpaint) so you do not over-fit the surround with long densification.
3. Inspect **`video/` / `video_rgb/`** at 5k; if hole is good and surround is stable, **stop** — no need for 12k.

If the hole is still soft at 5k but surround degrades by 12k, prefer **checkpoints** (`--checkpoint-video-iters 5000`) and pick the earlier checkpoint as the “shipping” result.

---

## What worked for `bag` (summary)

1. **LaMa `depth_completed` misaligned** vs renderer `depth/` → fused masked PLY floated vs hole PLY.
2. **`tools/inpaint360_align_completed_depth.py`** — ring-based `a*z+b` on masked pixels.
3. **`tools/inpaint360_project_completed_to_hole_plane.py`** — planar table constraint from hole depth ring.
4. **`edit_object_removal_plyfusion.py`** — `c2w` from `world_view_transform` (optional `--legacy_pose_rt`).
5. **`edit_object_inpaint.py`** — auto `supp_ply` selection + `--supp_ply` override.
6. Optional **`tools/inpaint360_filter_virtual_masks.py`** if fusion spikes persist.

Tracked mirrors + install: `upstream_overrides/inpaint360gs/` + `scripts/install_inpaint360gs_overrides.sh` + `tools/sync_inpaint360gs_upstream_overrides.py` (because `external/` is gitignored).

---

## What is left in the pipeline?

Depends on goal:

- **Science / report:** Run stages **1–11** on **`car`**, **`cube`**, etc.; fill metric tables; capture before/after videos.
- **Engineering:** After each `git pull` on GPU nodes, run **`bash scripts/install_inpaint360gs_overrides.sh`** so `external/Inpaint360GS/` matches tracked patches.
- **Optional:** Re-run **stage 11** only:  
  `START_STAGE=11 bash scripts/run_inpaint360gs_one_scene.sh bag`

---

## Novelty angles (without rewriting the whole stack)

| Idea | Notes |
|------|--------|
| **Depth-consistency fusion** | Document + compare “vanilla fusion” vs “aligned + planar” (your recipe) — clear reproducible contribution. |
| **Budget–quality tradeoff** | Systematic curve: 3k / 5k / 12k + checkpoint videos; report hole SSIM vs surround LPIPS. |
| **Multi-scene robustness** | Same pipeline on `car`, `cube`, `truck` with honest failures (small export). |
| **Masked metrics only on hole ring** | Already partially in repo tools — emphasize “hole-focused” eval vs full-frame. |
| **Ablate LaMa** | e.g. depth-only repair vs color-only — diagnostic section. |
| **Native latent path** | Tie forward to `latent_void` DiffSplat inpainting as “future work” vs Inpaint360GS 2D→3D scaffold. |

---

## Sequence cheat sheet (fresh scene)

```bash
# 0) One-time on GPU checkout
bash scripts/install_inpaint360gs_overrides.sh

# 1) Full 11 stages, single final budget 12k + videos at 5k and 8k
export INPAINT360_CHECKPOINT_VIDEO_ITERS="5000 8000"
FINETUNE_ITERS="12000" bash scripts/run_inpaint360gs_scene_full.sh car

# 2) If virtual patch floats (post–stage 9), before re-running 10a:
cd /path/to/latent_void
python tools/inpaint360_align_completed_depth.py \
  --completed-dir output/inpaint360/car/virtual/ours_object_removal/iteration_2000/depth_completed \
  --hole-dir output/inpaint360/car/virtual/ours_object_removal/iteration_2000/depth \
  --mask-dir data/inpaint360/car/inpaint_2d_unseen_mask_virtual \
  --backup --ring-width 5
python tools/inpaint360_project_completed_to_hole_plane.py \
  --completed-dir output/inpaint360/car/virtual/ours_object_removal/iteration_2000/depth_completed \
  --hole-dir output/inpaint360/car/virtual/ours_object_removal/iteration_2000/depth \
  --mask-dir data/inpaint360/car/inpaint_2d_unseen_mask_virtual \
  --backup --ring-width 5

# 3) Re-run fusion + inpaint + eval only
START_STAGE=10 FINETUNE_ITERS="12000" bash scripts/run_inpaint360gs_one_scene.sh car
```

Replace `car` with `cube`, `bag`, etc., and match `iteration_*` to your removal checkpoint.
