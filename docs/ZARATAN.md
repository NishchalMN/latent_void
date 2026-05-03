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
python scripts/download_marigold.py --output-dir checkpoints/marigold
python scripts/download_diffsplat_aux.py --output-dir checkpoints/diffsplat_aux
python -m latent_void validate-config --config configs/zaratan_inpaint360gs_bag.yaml
python -m latent_void discover-dataset --config configs/zaratan_inpaint360gs_bag.yaml
python -m latent_void prepare-geometry --config configs/zaratan_inpaint360gs_bag.yaml --dry-run
```

These smoke checks validate config, dataset discovery, and dry-run command
rendering before spending GPU time.

`scripts/setup_zaratan_deps.sh` clones/updates DiffSplat, SAM 3, and
Inpaint360GS. It also downloads DiffSplat's PixArt-Sigma checkpoint bundle when
missing. Set `INSTALL_GPU_DEPS=1` for the heavier PyTorch/SAM3/DiffSplat Python
package install. That heavy path also installs Marigold-compatible Diffusers and
DiffSplat's RaDe-GS `diff-gaussian-rasterization` extension. The default path
stays lightweight for config and dataset validation. The script intentionally
does not self-upgrade `pip` inside the active Zaratan venv.

## Staged H100 `srun` Jobs

Use direct `srun` stages while bringing up the real pipeline. tmux is optional
and only helps keep SSH sessions alive; the wrapper itself does not require it.
The wrapper keeps the command interactive and defaults to the
`msml612pcs3-class` account, `gpu-h100` partition, and `gpu:h100:1` GRES:

```bash
scripts/zaratan_srun_stage.sh geometry configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_srun_h100
scripts/zaratan_srun_stage.sh reconstruct configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_srun_h100
scripts/zaratan_srun_stage.sh segment configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_srun_h100
scripts/zaratan_srun_stage.sh finish configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_srun_h100
```

The first command runs the zero-training Marigold geometry preprocessing. The
second consumes `geometry_manifest.json` and runs DiffSplat GSRecon/GSVAE. The
third runs SAM 3 masks for the configured prompt. The render stage decodes the
original and inpainted GSVAE latents into before/after diagnostic views after
`fuse` and `inpaint` have produced the void mask and `latent_inpainted.npy`.

If an A100 smoke is useful while an H100 job is pending, override both the
partition and the GPU type; overriding only the partition leaves the H100 GRES
request in place:

```bash
SLURM_PARTITION=gpu-a100 SLURM_GRES=gpu:a100:1 \
  scripts/zaratan_srun_stage.sh geometry configs/zaratan_inpaint360gs_bag.yaml \
  --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_mini_a100
```

## Real Job

Copy and edit the example config with real checkpoint and dataset paths:

```bash
cp configs/inpaint360gs_example.yaml configs/my_scene.yaml
```

Then run:

```bash
scripts/zaratan_srun_stage.sh run configs/my_scene.yaml
```

Default Slurm settings:

- account: `msml612pcs3-class`
- partition: `gpu-h100`
- GRES: `gpu:h100:1`
- CPU/memory defaults: 8 CPUs and 64 GB memory per stage

## Quota And Group Notes

The working copy was moved to group `zt-msml605` after `zt-msml612pcs3` hit its
BeegFS file-count hard quota and blocked Git writes. If writes start failing
with `Disk quota exceeded`, check both group quotas:

```bash
beegfs-ctl --getquota --gid 102781
beegfs-ctl --getquota --gid 102518
```

The active repo tree should show group `zt-msml605`:

```bash
ls -ld . .git runs logs .venvs data checkpoints
```

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

`scripts/setup_zaratan_deps.sh` downloads Marigold snapshots by default. The
Zaratan config points at `checkpoints/marigold/depth-v1-1` and
`checkpoints/marigold/normals-v1-1` so H100 jobs load local files instead of
trying to reach Hugging Face from a compute node.

The setup script also downloads DiffSplat auxiliary VAE snapshots by default:
`checkpoints/diffsplat_aux/sdxl-vae-fp16-fix` and
`checkpoints/diffsplat_aux/taesdxl`. These are required because DiffSplat's
SDXL GSVAE hardcodes `madebyollin/sdxl-vae-fp16-fix` and `madebyollin/taesdxl`
inside model construction. The local adapters pass those paths explicitly so
offline H100 jobs do not try to resolve Hugging Face repo IDs.
