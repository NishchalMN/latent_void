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
sbatch slurm/zaratan_smoke.sbatch configs/zaratan_inpaint360gs_bag.yaml
```

The smoke job only validates config, dataset discovery, and dry-run command
rendering. It should not spend meaningful GPU time.

`scripts/setup_zaratan_deps.sh` clones/updates DiffSplat, SAM 3, and
Inpaint360GS. It also downloads DiffSplat's PixArt-Sigma checkpoint bundle when
missing. Set `INSTALL_GPU_DEPS=1` for the heavier PyTorch/SAM3/DiffSplat Python
package install; the default path stays lightweight for config and dataset
validation. The script intentionally does not self-upgrade `pip` inside the
active Zaratan venv.

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
- partition: `gpu-h100`
- GRES: `gpu:h100:1`

## Known External Requirements

- SAM 3 checkpoint download requires Hugging Face authentication and access to
  the `facebook/sam3` model repo. Run `hf auth login` in the Zaratan environment
  before using `tools/run_sam3_multiview.py`.
- Check access with:

  ```bash
  unset PYTHONPATH
  source .venvs/latent_void_py310/bin/activate
  python scripts/check_sam3_access.py
  ```

- DiffSplat does not currently provide a simple GSRecon export CLI in the
  upstream repo. `tools/run_gsrecon_export.py` records the expected contract and
  fails loudly until the exporter is implemented against the installed
  checkpoints.
- Inpaint360GS scenes carry COLMAP camera metadata under `sparse/0`; the local
  loader reads both COLMAP text and binary camera/image files into the run
  manifest.
