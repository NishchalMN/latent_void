# Status

Last updated: 2026-05-02

## What Works

- `latent_void` is now a standalone Git repo.
- Local repo root:
  `/fs/nexus-scratch/gnanesh/DiffSplat/latent_void`
- GitHub remote:
  `git@github-nishchal:NishchalMN/latent_void.git`
- Zaratan clone:
  `/home/gnanesh/scratch.msml612pcs3/latent_void`
- Branch:
  `main`
- Latest pushed commit before this memory update:
  `6942682 Keep dry-run imports lightweight`

Implemented and verified locally:

- Config validation with env-var default expansion.
- Inpaint360GS dataset discovery.
- DiffSplat/GSRecon external command dry-run rendering.
- SAM 3 external command dry-run rendering.
- Multi-view mask fusion with synthetic projected Gaussian data.
- Latent void mask generation.
- Fallback latent fill for plumbing tests.
- Slurm templates for smoke and inpainting jobs.
- Zaratan runbook and pipeline docs.

Local commands that passed:

```bash
python3 -m unittest discover -s tests
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
python3 -m latent_void run --config configs/inpaint360gs_example.yaml --dry-run
bash -n scripts/pull_zaratan.sh scripts/push_main.sh slurm/zaratan_smoke.sbatch slurm/zaratan_inpaint.sbatch
```

Zaratan commands that passed on the login node:

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
git pull --ff-only
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
python3 -m latent_void run --config configs/inpaint360gs_example.yaml --dry-run
```

## What Fails Or Is Not Ready

- No real H100 job has been submitted yet.
- The example config still contains placeholder dataset/checkpoint paths.
- The installed DiffSplat, GSRecon, SAM 3, and latent inpainting commands have
  not been wired to real Zaratan paths yet.
- Zaratan SSH clone from GitHub failed due to missing public-key auth.
- Zaratan HTTPS clone/pull works.
- Zaratan login-node Python has `yaml` but not `numpy`.
- Tensor-heavy stages such as `fuse` and `inpaint` require a Python environment
  with NumPy, likely inside the GPU/model environment or container.
- The fallback latent inpaint is only a plumbing test. It is not research-quality
  native latent inpainting.
- The Gaussian `.npz` contract currently expects precomputed `uvs` and
  `visibility`. If GSRecon exports positions/cameras instead, projection must be
  added upstream or implemented in this repo.

## Important Current Artifacts

Local files:

- `configs/inpaint360gs_example.yaml`
- `slurm/zaratan_smoke.sbatch`
- `slurm/zaratan_inpaint.sbatch`
- `docs/PIPELINE.md`
- `docs/ZARATAN.md`
- `tests/test_pipeline.py`

Generated local dry-run artifacts are ignored by Git under `runs/`.

## Next Best Step

Create a real config, for example `configs/zaratan_inpaint360gs_scene.yaml`,
with actual paths for:

- Inpaint360GS root and scene name.
- DiffSplat checkout.
- GSRecon weights.
- GSVAE weights.
- SAM 3 checkout.
- SAM 3 weights.
- Output directory on Zaratan scratch.

Then run:

```bash
sbatch slurm/zaratan_smoke.sbatch configs/zaratan_inpaint360gs_scene.yaml
```

After the smoke job succeeds, run the real configured pipeline job:

```bash
sbatch slurm/zaratan_inpaint.sbatch configs/zaratan_inpaint360gs_scene.yaml
```
