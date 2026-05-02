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
- Official external repos cloned on Zaratan under `external/`:
  - DiffSplat
  - SAM 3
  - Inpaint360GS
- Zaratan lightweight Python 3.10 environment created at
  `.venvs/latent_void_py310`.
- Inpaint360GS core dataset downloaded to
  `data/downloads/inpaint360.zip`.
- Inpaint360GS core dataset unpacked under `data/inpaint360` after rerunning
  unzip with `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`.
- DiffSplat PixArt-Sigma checkpoint bundle downloaded under
  `checkpoints/diffsplat`, including:
  - `gsrecon_gobj265k_cnp_even4`
  - `gsvae_gobj265k_sdxl_fp16`
  - `gsdiff_gobj83k_pas_fp16__render`

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
- `configs/zaratan_inpaint360gs_bag.yaml` points at the downloaded Zaratan
  dataset/repo/checkpoint locations.
- The installed DiffSplat and SAM 3 wrappers are wired to real Zaratan paths,
  but the DiffSplat scene exporter and research-quality latent inpainting logic
  are not complete yet.
- Zaratan SSH clone from GitHub failed due to missing public-key auth.
- Zaratan HTTPS clone/pull works.
- Zaratan login-node Python has `yaml` but not `numpy`.
- Tensor-heavy stages such as `fuse` and `inpaint` require a Python environment
  with NumPy, likely inside the GPU/model environment or container.
- Zaratan's Python module prepends global package paths via `PYTHONPATH`; setup
  and Slurm scripts now unset `PYTHONPATH` before activating the project venv.
- The fallback latent inpaint is only a plumbing test. It is not research-quality
  native latent inpainting.
- The Gaussian `.npz` contract currently expects precomputed `uvs` and
  `visibility`. If GSRecon exports positions/cameras instead, projection must be
  added upstream or implemented in this repo.
- SAM 3 checkpoint access requires Hugging Face authentication and accepted
  access to the official Meta SAM 3 model repo.
- Verified SAM 3 blocker on Zaratan:
  - `facebook/sam3` metadata is visible.
  - actual checkpoint file access fails with `GatedRepoError 401`.
  - user must request/accept access and run `hf auth login` on Zaratan.
- DiffSplat upstream has training and generation scripts, but no direct
  `run_gsrecon.py` scene-export CLI. A wrapper contract now exists at
  `tools/run_gsrecon_export.py`; the actual exporter must be implemented once
  checkpoints/environment details are available.
- DiffSplat's public GSRecon checkpoint is trained for GObjaverse-style
  four-view object inputs and expects RGB plus camera-derived Plucker rays and,
  by default, normal/coordinate channels. Inpaint360GS gives real scene RGB and
  COLMAP poses, so a direct RGB-only call is not enough.

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

Validate the Inpaint360GS scene discovery on Zaratan, then resolve credentials
and model-environment blockers:

- GSRecon weights.
- GSVAE weights.
- SAM 3 Hugging Face auth/checkpoints.
- Normal/coordinate-map strategy for DiffSplat GSRecon scene encoding.

Then run:

```bash
sbatch slurm/zaratan_smoke.sbatch configs/zaratan_inpaint360gs_bag.yaml
```

After the smoke job succeeds, run the real configured pipeline job:

```bash
sbatch slurm/zaratan_inpaint.sbatch configs/zaratan_inpaint360gs_scene.yaml
```
