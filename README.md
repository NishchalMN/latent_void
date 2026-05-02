# latent_void

Native latent 3D object removal for real multi-view scenes.

The first target is Inpaint360GS-style scene data. The pipeline reconstructs a
scene through DiffSplat `GSRecon`, converts the reconstruction into
GSVAE-compatible Gaussian grids, segments prompted objects with SAM 3 across
multiple views, fuses the masks into a true 3D void, and inpaints the void in
latent/Gaussian space.

This repository intentionally keeps heavyweight model code behind adapters.
Local development can validate configs, masks, latent shapes, Slurm scripts, and
orchestration without installing DiffSplat or SAM 3. Zaratan/H100 jobs provide
the real model commands through config values.

## Quick Start

```bash
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
python3 -m unittest discover -s tests
```

On Zaratan:

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
git pull
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
sbatch slurm/zaratan_smoke.sbatch configs/inpaint360gs_example.yaml
```

## Pipeline Stages

1. `discover-dataset`: inspect configured Inpaint360GS scene inputs.
2. `reconstruct`: run the configured DiffSplat/GSRecon command.
3. `segment`: run the configured SAM 3 command or validate provided masks.
4. `fuse`: fuse multi-view masks into Gaussian and latent void masks.
5. `inpaint`: run configured latent inpainting or fallback masked latent fill.
6. `run`: execute the staged pipeline in order.

Every heavy external command supports `--dry-run` through the CLI so Slurm jobs
can be validated before GPU time is spent.

## Configure a Real Run

Create a copy of `configs/inpaint360gs_example.yaml`, then set:

- `dataset.root` and `dataset.scene`
- `checkpoints.diffsplat_root`, `checkpoints.gsrecon_weights`, `checkpoints.gsvae_weights`
- `checkpoints.sam3_root`, `checkpoints.sam3_weights`
- `external.gsrecon_command` for the installed DiffSplat checkout
- `external.sam3_command` for the installed SAM 3 checkout
- `pipeline.gaussian_npz` and `pipeline.latent_npy` if GSRecon writes them somewhere nonstandard

The fuse stage expects the Gaussian `.npz` to contain projected `uvs` with shape
`[views, gaussians, 2]` and `visibility` with shape `[views, gaussians]`.
If the installed renderer exports positions and cameras instead, add that
projection upstream before `fuse`.

## Repository Sync

This folder is the standalone Git repository. The expected remote is:

```bash
git@github-nishchal:NishchalMN/latent_void.git
```

The Zaratan working copy lives at:

```bash
/home/gnanesh/scratch.msml612pcs3/latent_void
```

If SSH keys are unavailable on Zaratan, use the HTTPS remote there:

```bash
git remote set-url origin https://github.com/NishchalMN/latent_void.git
```
