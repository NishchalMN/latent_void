# Zaratan Runbook

## One-Time Repo Setup

The `zaratan` tmux session already has the working copy at:

```bash
/home/gnanesh/scratch.msml612pcs3/latent_void
```

If the clone is empty or stale:

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
git remote -v
git pull --ff-only
```

If SSH authentication is unavailable on Zaratan:

```bash
git remote set-url origin https://github.com/NishchalMN/latent_void.git
git pull --ff-only
```

## Smoke Job

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
scripts/setup_zaratan_deps.sh
scripts/download_inpaint360gs.sh
python -m latent_void validate-config --config configs/zaratan_inpaint360gs_bag.yaml
python -m latent_void discover-dataset --config configs/zaratan_inpaint360gs_bag.yaml
python -m latent_void prepare-geometry --config configs/zaratan_inpaint360gs_bag.yaml --dry-run
sbatch slurm/zaratan_smoke.sbatch configs/zaratan_inpaint360gs_bag.yaml
```

The smoke job only validates config, dataset discovery, and dry-run command
rendering. It runs on the short CPU `debug` partition and should not spend GPU
time.

`scripts/setup_zaratan_deps.sh` clones/updates DiffSplat, SAM 3, and
Inpaint360GS. It also downloads DiffSplat's PixArt-Sigma checkpoint bundle when
missing. Set `INSTALL_GPU_DEPS=1` for the heavier PyTorch/SAM3/DiffSplat Python
package install. That heavy path also installs Marigold-compatible Diffusers and
DiffSplat's RaDe-GS `diff-gaussian-rasterization` extension. The default path
stays lightweight for config and dataset validation. The script intentionally
does not self-upgrade `pip` inside the active Zaratan venv.

## Staged H100 Jobs

Use the staged jobs while bringing up the real pipeline:

```bash
sbatch slurm/zaratan_geometry.sbatch configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_mini
sbatch slurm/zaratan_reconstruct.sbatch configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/inpaint360gs_bag_mini
sbatch slurm/zaratan_segment.sbatch configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_mini
```

The first command runs the zero-training Marigold geometry preprocessing. The
second consumes `geometry_manifest.json` and runs DiffSplat GSRecon/GSVAE. The
third runs SAM 3 masks for the configured prompt.

## Real Job

Copy and edit the example config with real checkpoint and dataset paths:

```bash
cp configs/inpaint360gs_example.yaml configs/my_scene.yaml
```

Then run:

```bash
sbatch slurm/zaratan_inpaint.sbatch configs/my_scene.yaml
```

Default Slurm settings:

- account: `msml612pcs3-class`
- smoke partition: `debug`
- real inpaint partition: `gpu-h100`
- real inpaint GRES: `gpu:h100:1`

## Known External Requirements

- SAM 3 checkpoint download requires Hugging Face authentication and access to
  the `facebook/sam3` model repo. Run `hf auth login` in the Zaratan environment
  before using `tools/run_sam3_multiview.py`.
- `tools/run_sam3_multiview.py` supports both the local `facebookresearch/sam3`
  repo backend and the official Hugging Face Transformers backend. `--backend
  auto` tries Transformers first and falls back to the repo backend.
- Check access with:

  ```bash
  unset PYTHONPATH
  source .venvs/latent_void_py310/bin/activate
  python scripts/check_sam3_access.py --download
  ```

- DiffSplat does not currently provide a simple GSRecon export CLI in the
  upstream repo. `tools/run_gsrecon_export.py` is the local adapter and consumes
  `geometry_manifest.json`, then exports `gaussians.npz`, `gs_grid.npy`, and
  `latent.npy`.
- Inpaint360GS scenes carry COLMAP camera metadata under `sparse/0`; the local
  loader reads both COLMAP text and binary camera/image files into the run
  manifest.
- The zero-training geometry path uses Marigold depth and normals before
  DiffSplat GSRecon. Install heavy dependencies with:

  ```bash
  INSTALL_GPU_DEPS=1 scripts/setup_zaratan_deps.sh
  ```
