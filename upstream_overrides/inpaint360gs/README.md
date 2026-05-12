# Patched Inpaint360GS files (tracked in git)

The checkout you run lives under `external/Inpaint360GS/`, which this repo **gitignores**
(see root `.gitignore`). You will therefore **never see** edits there in `git status`.

To keep parity with what you ran on Zaratan/gpu:

## 1) Refresh tracked copies before `git commit` (recommended)

From a checkout where `external/Inpaint360GS/` is populated:

```bash
python3 tools/sync_inpaint360gs_upstream_overrides.py
git add upstream_overrides/inpaint360gs/
git add tools/sync_inpaint360gs_upstream_overrides.py
```

Optional alternate source dir:

```bash
INPAINT360GS_COPY_SRC=/path/to/Inpaint360GS python3 tools/sync_inpaint360gs_upstream_overrides.py
```

## 2) Push install into `external/Inpaint360GS/` on GPU / Zaratan machines

After `git pull` (which brings updates under `upstream_overrides/`):

```bash
bash scripts/install_inpaint360gs_overrides.sh
```

## Files mirrored here

- `edit_object_removal_plyfusion.py` — `c2w` from `world_view_transform`; `--legacy_pose_rt`.
- `edit_object_inpaint.py` — `auto_select_support_ply`, `--supp_ply`, `--checkpoint-video-iters`, optional `KEEP_FULL_OPTIMIZED` path, etc.

If `edit_object_inpaint.py` is missing in this folder, run step (1).

After pulling edits from another machine, refresh both files with:

```bash
python3 tools/sync_inpaint360gs_upstream_overrides.py
```
