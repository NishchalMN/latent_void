# Zaratan Memory

Last updated: 2026-05-02

## Known Paths

Local repo:

```bash
/fs/nexus-scratch/gnanesh/DiffSplat/latent_void
```

Zaratan repo:

```bash
/home/gnanesh/scratch.msml612pcs3/latent_void
```

GitHub:

```bash
https://github.com/NishchalMN/latent_void.git
git@github-nishchal:NishchalMN/latent_void.git
```

## Known Cluster State

Visible accounts from `sbalance`:

- `msml605-class`
- `msml612pcs3-class`

Use `msml612pcs3-class` by default.

Useful partitions from `sinfo`:

- `debug`
- `standard`
- `gpu-h100`
- `gpu`
- `gpu-a100`
- `gpu-v100`

Use `debug` for dry-run smoke jobs and `gpu-h100` for heavy jobs.

## Sync Workflow

Local:

```bash
git status --short --branch
git add .
git commit -m "..."
git push
```

Zaratan tmux session:

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
git pull --ff-only
git status --short --branch
```

Helper from local machine:

```bash
scripts/pull_zaratan.sh
```

## SSH Note

GitHub SSH clone on Zaratan failed with public-key authentication. HTTPS clone
and pull worked. If the remote ever points at SSH on Zaratan, switch it back:

```bash
git remote set-url origin https://github.com/NishchalMN/latent_void.git
```

## Login-Node Python Note

Zaratan login-node system `python3` had:

- `yaml`: available
- `numpy`: unavailable

The CLI dry-run path was adjusted so config validation and dry-run orchestration
do not import NumPy. Real `fuse` and `inpaint` stages still need NumPy in the
active GPU/model environment.

Project venv:

```bash
unset PYTHONPATH
source .venvs/latent_void_py310/bin/activate
python -c "import numpy; print(numpy.__version__)"
```

Expected: `1.26.4`.

## Smoke Commands

```bash
scripts/setup_zaratan_deps.sh
scripts/download_inpaint360gs.sh
python -m latent_void validate-config --config configs/zaratan_inpaint360gs_bag.yaml --strict-paths
python -m latent_void discover-dataset --config configs/zaratan_inpaint360gs_bag.yaml
python -m latent_void prepare-geometry --config configs/zaratan_inpaint360gs_bag.yaml --dry-run
sbatch slurm/zaratan_smoke.sbatch configs/zaratan_inpaint360gs_bag.yaml
```

Latest login-node discovery result for `bag`: 156 images, 16 selected views,
COLMAP cameras found.

The latest `debug` smoke Slurm job was canceled while pending for resources, so
no smoke job is currently queued.

## Heavy Geometry/Model Setup

SAM 3 auth is active for the current account and the checkpoint downloaded to:

```bash
/home/gnanesh/scratch.msml612pcs3/latent_void/checkpoints/sam3/sam3.pt
```

Install model dependencies before running Marigold or GSRecon on H100:

```bash
INSTALL_GPU_DEPS=1 scripts/setup_zaratan_deps.sh
```

Download Marigold snapshots for offline compute-node use:

```bash
python scripts/download_marigold.py --output-dir checkpoints/marigold
```

That heavy setup installs PyTorch, SAM 3, DiffSplat requirements,
Marigold-compatible Diffusers, Transformers, and the RaDe-GS
`diff-gaussian-rasterization` extension required by DiffSplat imports.

The zero-training geometry stage writes `runs/inpaint360gs_bag/geometry/geometry_manifest.json`.
The GSRecon export stage consumes that manifest and writes `gaussians.npz`,
`gs_grid.npy`, and `latent.npy`.

Staged H100 bring-up:

```bash
sbatch slurm/zaratan_geometry.sbatch configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_mini
sbatch slurm/zaratan_reconstruct.sbatch configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/inpaint360gs_bag_mini
sbatch slurm/zaratan_segment.sbatch configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_mini
sbatch slurm/zaratan_render.sbatch configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_mini
```

If submitting an A100 backup, override both partition and GRES:

```bash
sbatch --partition=gpu-a100 --gres=gpu:a100:1 slurm/zaratan_geometry.sbatch configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_mini_a100
```

Current geometry bring-up:

- `19185136` ran on `gpu-a6-4.zaratan.umd.edu` and failed because
  `tools/preprocess_geometry.py` could not import the local `latent_void`
  package when invoked as a script.
- `7cea7f7` fixes that script import path.
- `19185139` was the replacement mini geometry job. It started on
  `gpu-a6-9` at `2026-05-02T19:29:24` and was later canceled for the Marigold
  model-resolution stall noted below.
- Dependent continuation chain was also canceled:
  - `19186465`: reconstruct after `19185139`
  - `19186466`: SAM 3 segmentation after `19186465`
  - `19186467`: skip-heavy final run after `19186466`; does fuse, fallback
    inpaint, and render diagnostics.
- `19185139` and its dependent chain were canceled after an overlapping process
  diagnostic showed `tools/preprocess_geometry.py` sleeping for 11+ minutes,
  0 MiB GPU usage, and no geometry files. Treat that as a Marigold model
  resolution/download stall on the compute node. Use local
  `checkpoints/marigold/*` snapshots before resubmitting.
- `19185424` was a backup `gpu-a100` geometry job and was canceled.
- `19186443` was a later stray `gpu-a100` geometry submission, estimated for
  `2026-05-02T21:00:00`, and was also canceled.
- A non-submitting `sbatch --test-only --partition=gpu --gres=gpu:h100:1`
  check estimated `2026-05-03T23:53:18`, which is later than the active
  `gpu-h100` job. Keep `19185139`.
- Zaratan has pulled `1b240bf`; post-pull unit tests, py_compile, full dry-run,
  and render dry-run passed.

## Quota Note

The original working-copy group `zt-msml612pcs3` hit BeegFS file-count quota and
blocked writes, including Git object creation. Python caches were deleted, then
the repo tree was moved to group `zt-msml605`:

```bash
chgrp -R zt-msml605 .
find . -type d -exec chmod g-s {} +
```

After that, `.git/write_test` and `git pull --ff-only` succeeded. Keep new
artifacts under this working copy group unless the project-group quota is
explicitly repaired.

## Real Job Command

After creating a real config:

```bash
sbatch slurm/zaratan_inpaint.sbatch configs/zaratan_inpaint360gs_scene.yaml
```
