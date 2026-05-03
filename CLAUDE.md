# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`latent_void` is a scaffold for native 3D object removal on real multi-view
scenes (first target: Inpaint360GS). It does **not** vendor heavyweight model
code (DiffSplat, SAM 3, Marigold). Each pipeline stage is a thin Python
adapter that renders a shell command template from config and dispatches it.
This lets local development validate configs, manifests, mask/latent shapes,
and Zaratan command rendering without installing GPU dependencies; the real
models run on Zaratan H100s.

The research direction is intentionally **native latent inpainting**, not
per-view 2D inpainting + reprojection (see `project_memory/DECISIONS.md`
D005). Per-view 2D inpainting is acceptable only as a diagnostic.

`project_memory/` holds the canonical project state: read `CONTEXT.md`,
`PLAN.md`, and especially `STATUS.md` after any context reset — they record
what works, what failed, the latest verified H100 run, and what is queued.
Update `STATUS.md` after non-trivial changes.

## Common commands

Local (lightweight venv, no GPU; only `numpy` + `PyYAML` required):

```bash
python3 -m unittest discover -s tests
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
python3 -m latent_void run --config configs/inpaint360gs_example.yaml --dry-run
```

Run a single test class or method:

```bash
python3 -m unittest tests.test_pipeline
python3 -m unittest tests.test_pipeline.TestRunPipeline.test_dry_run_writes_manifest
```

Two tests require Pillow and are skipped locally; they run inside the
Zaratan venv. There is no lint/format/CI configured — do not introduce one
without asking.

Per-stage CLI (each accepts `--config`, repeatable `--set key.path=value`,
and `--dry-run` where it dispatches an external command):

```bash
python3 -m latent_void discover-dataset  --config <cfg>
python3 -m latent_void prepare-geometry  --config <cfg> [--dry-run]
python3 -m latent_void reconstruct       --config <cfg> [--dry-run]
python3 -m latent_void segment           --config <cfg> [--dry-run]
python3 -m latent_void fuse              --config <cfg>
python3 -m latent_void inpaint           --config <cfg> [--dry-run]
python3 -m latent_void render            --config <cfg> [--dry-run]
python3 -m latent_void run               --config <cfg> [--dry-run] [--skip-geometry --skip-reconstruct --skip-segment]
```

Zaratan H100 stages (interactive `srun` from the `zaratan` tmux session):

```bash
scripts/zaratan_srun_stage.sh <geometry|reconstruct|segment|finish|run> <config> [--set ...]
```

`finish` is the post-GPU-stage continuation (`run --skip-geometry
--skip-reconstruct --skip-segment`) for fuse → inpaint → render. Legacy
`slurm/*.sbatch` templates exist but the active workflow is direct `srun`.

## Architecture

### Adapter pattern (read this first)

Every heavy stage in `latent_void/pipeline.py` follows the same shape:

1. Resolve output directories under `project.output_dir` via `run_dirs()`.
2. Look up a shell template at `external.<stage>_command` in the config.
3. Format it with a `values` dict of paths and parameters (e.g.
   `{diffsplat_root}`, `{geometry_manifest}`, `{mask_dir}`,
   `{config_path}`).
4. Dispatch via `latent_void.external.run_command`, which honors `--dry-run`
   by returning the rendered command without executing it.
5. Write a per-stage `*_command.json` and a stage manifest into the run dir.

When extending a stage, prefer adding a placeholder to the `values` dict and
updating the example config template over hardcoding behavior in Python.
The configs in `configs/` are the contract between this repo and the
installed DiffSplat/SAM 3 checkouts on Zaratan.

### Pipeline stages and data contracts

`latent_void/pipeline.py` orchestrates:

- `prepare_geometry` → Marigold depth/normals + COLMAP coordinate maps.
  Implementation in `tools/preprocess_geometry.py` and
  `latent_void/geometry.py`. Camera poses are normalized into a
  DiffSplat-like object-scale frame; supports both `scene_minmax` and
  `diffsplat` (`coord * 0.5 + 0.5`) coord encodings. Output:
  `<run>/geometry/geometry_manifest.json`.
- `run_gsrecon` → DiffSplat GSRecon/GSVAE export via
  `tools/run_gsrecon_export.py`. Writes `gaussians.npz` (with `uvs`,
  `visibility`, and `gaussian_grid_shape`/`latent_shape`/`gs_grid_shape`
  metadata), `latent.npy`, `gs_grid.npy`. Contains compatibility shims for
  Transformers 5 import removals, missing `wandb`, and the Zaratan
  `diff_gaussian_rasterization` build (see `latent_void/diffsplat_compat.py`
  for the rasterizer shim that drops unsupported `require_coord` and
  expands 6-tensor returns to DiffSplat's 8-tensor shape). Don't strip
  these without re-validating on Zaratan.
- `run_segmentation` → SAM 3 multi-view masks via
  `tools/run_sam3_multiview.py`, supporting `--backend
  auto|transformers|repo`. Transformers backend is the H100-tested path;
  the repo backend is an untested fallback. Masks are resized to the
  geometry input resolution so projected Gaussians and SAM masks share
  pixel coordinates during fusion.
- `fuse_void` → fuses per-view 2D masks into a 3D Gaussian deletion mask and
  a latent void mask using projected `uvs [V,G,2]` + `visibility [V,G]`
  from `gaussians.npz`. Logic in `latent_void/masks.py`,
  `latent_void/gaussians.py`, `latent_void/latent.py`. If a future
  GSRecon export only ships positions and cameras, project upstream
  before `fuse`.
- `run_latent_inpaint` → external command if configured, else (when
  `pipeline.allow_fallback_inpaint: true`) a channel-mean fill in
  `latent_void/latent.fallback_inpaint_latent`. **The fallback is plumbing
  only, not research quality** — replacing it with a real masked latent
  denoiser is the active research target (see
  `project_memory/PROGRESS_AND_REMAINING.md`).
- `run_render` → `tools/render_latent_scene.py` decodes the original and
  inpainted GSVAE latents and renders before/after RGB/alpha/depth via
  DiffSplat's renderer.

GSVAE 12-channel grid order is **RGB (3), scale (3), rotation quaternion
(4), opacity (1), depth (1)** — matches DiffSplat upstream. If touching
channel layout, update `latent_void/diffsplat_compat.py` and
`tools/run_gsrecon_export.py` together; render diagnostics depend on this.

### Config system

`latent_void/config.py` loads YAML, expands `${VAR:-default}` env-var
syntax, validates required keys (`validate_config`), and supports dotted
`--set a.b.c=value` overrides applied after load. `_config_path` is
injected and propagated into command templates as `{config_path}`.
`validate-config --strict-paths` additionally asserts referenced
checkpoint/dataset paths exist on disk.

The two reference configs:

- `configs/inpaint360gs_example.yaml` — generic template with env-var
  defaults; used by local dry-run tests.
- `configs/zaratan_inpaint360gs_bag.yaml` — concrete Zaratan paths for the
  `bag` scene; used by H100 runs.

### Local vs Zaratan environments

Local: lightweight venv (`numpy`, `PyYAML` only). The login node
intentionally lacks heavy deps; `fuse` and `inpaint` need NumPy but most
other stages dry-run without it. **Do not import NumPy at module load
time** in code reachable from `validate-config` or `--dry-run` — the
existing pipeline imports NumPy lazily inside stage functions for this
reason.

Zaratan: `.venvs/latent_void_py310` (NumPy pinned to 1.26.4), Python module
`python/3.10.10/gcc/11.3.0/cuda/12.3.0/linux-rhel8-zen2`. Heavy deps
(`torch`, DiffSplat, SAM 3, RaDe-GS rasterizer) installed via
`scripts/setup_zaratan_deps.sh` with `INSTALL_GPU_DEPS=1`. Compute nodes
are offline-only — set `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` and
pre-download Marigold + DiffSplat auxiliary VAE snapshots on the login
node via `scripts/download_marigold.py` and
`scripts/download_diffsplat_aux.py` before geometry/reconstruct.

## Conventions specific to this repo

- Local checkout and Zaratan checkout sync via GitHub
  (`NishchalMN/latent_void`). There is no shared filesystem.
  `scripts/push_main.sh` / `scripts/pull_zaratan.sh` wrap the loop. SSH
  auth has historically failed on Zaratan; HTTPS remote is the fallback.
- Heavy artifacts (`runs/`, `data/`, `checkpoints/`, `external/`,
  `*.log`/`*.out`/`*.err`) are gitignored; do not check them in or expect
  them on a fresh clone.
- When adding a new external command placeholder, also add it to the
  example config and to the `values` dict in `pipeline.py`. The dry-run
  output is the primary regression signal.
- Update `project_memory/STATUS.md` after non-trivial changes so the next
  session can resume without re-deriving state.
