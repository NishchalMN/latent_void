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
  `7cea7f7 Allow geometry tool imports under Slurm`

Implemented and verified locally:

- Config validation with env-var default expansion.
- Inpaint360GS dataset discovery.
- DiffSplat/GSRecon external command dry-run rendering.
- SAM 3 external command dry-run rendering.
- Marigold geometry preprocessing command dry-run rendering.
- Local coordinate-map reprojection from COLMAP cameras and depth maps.
- GSRecon export adapter now accepts `geometry_manifest.json` and is able to
  export `gaussians.npz`, `gs_grid.npy`, and `latent.npy` once DiffSplat GPU
  dependencies are available.
- Zaratan setup now has a heavy dependency path that installs Marigold-capable
  Diffusers plus DiffSplat's RaDe-GS `diff-gaussian-rasterization` extension.
- SAM 3 multiview wrapper can run through the official Hugging Face
  Transformers backend or the cloned `facebookresearch/sam3` repo backend.
- Staged H100 Slurm scripts exist for geometry, GSRecon/GSVAE reconstruction,
  and SAM 3 segmentation.
- Multi-view mask fusion with synthetic projected Gaussian data.
- Latent void mask generation.
- Fallback latent fill for plumbing tests.
- Slurm templates for smoke and inpainting jobs.
- Smoke Slurm template uses the short CPU `debug` partition; real inpainting
  still uses `gpu-h100`.
- Zaratan runbook and pipeline docs.
- Official external repos cloned on Zaratan under `external/`:
  - DiffSplat
  - SAM 3
  - Inpaint360GS
- Zaratan lightweight Python 3.10 environment created at
  `.venvs/latent_void_py310`.
- Zaratan venv has NumPy pinned to `1.26.4` after setup.
- Inpaint360GS core dataset downloaded to
  `data/downloads/inpaint360.zip`.
- Inpaint360GS core dataset unpacked under `data/inpaint360` after rerunning
  unzip with `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`.
- DiffSplat PixArt-Sigma checkpoint bundle downloaded under
  `checkpoints/diffsplat`, including:
  - `gsrecon_gobj265k_cnp_even4`
  - `gsvae_gobj265k_sdxl_fp16`
  - `gsdiff_gobj83k_pas_fp16__render`
- SAM 3 checkpoint download now succeeds on Zaratan after HF auth:
  `checkpoints/sam3/sam3.pt`.
- Zaratan heavy GPU dependency setup now completes with:
  - `torch` / `torchvision`
  - SAM 3 repo package
  - DiffSplat requirements
  - Diffusers 0.35.2 with Marigold pipelines
  - Transformers SAM 3 classes
  - RaDe-GS `diff_gaussian_rasterization`
- Login-node import checks pass for:
  - `torch`
  - `diffusers`
  - `transformers`
  - `sam3`
  - `diff_gaussian_rasterization`
  - `MarigoldDepthPipeline`
  - `MarigoldNormalsPipeline`
  - `Sam3Model`
  - `Sam3Processor`
- Inpaint360GS `bag` scene discovery works on Zaratan:
  - `num_images`: 156
  - selected views with current config: 16
  - COLMAP cameras loaded from binary `sparse/0`
  - first selected view: `IMG_0087`
  - last selected view: `IMG_0102`

Local commands that passed:

```bash
python3 -m unittest discover -s tests
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
python3 -m latent_void run --config configs/inpaint360gs_example.yaml --dry-run
bash -n scripts/pull_zaratan.sh scripts/push_main.sh slurm/zaratan_smoke.sbatch slurm/zaratan_inpaint.sbatch
bash -n scripts/setup_zaratan_deps.sh scripts/download_inpaint360gs.sh
bash -n slurm/zaratan_smoke.sbatch slurm/zaratan_inpaint.sbatch slurm/zaratan_geometry.sbatch slurm/zaratan_reconstruct.sbatch slurm/zaratan_segment.sbatch
python3 -m latent_void run --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/zaratan_bag_dry --dry-run
python3 -m latent_void prepare-geometry --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/zaratan_bag_geometry_dry --dry-run
python3 -m py_compile latent_void/geometry.py latent_void/pipeline.py latent_void/cli.py tools/preprocess_geometry.py tools/run_gsrecon_export.py tools/run_sam3_multiview.py
```

Zaratan commands that passed on the login node:

```bash
cd /home/gnanesh/scratch.msml612pcs3/latent_void
git pull --ff-only
python3 -m latent_void validate-config --config configs/inpaint360gs_example.yaml
python3 -m latent_void run --config configs/inpaint360gs_example.yaml --dry-run
scripts/setup_zaratan_deps.sh
scripts/download_inpaint360gs.sh
python -m latent_void validate-config --config configs/zaratan_inpaint360gs_bag.yaml --strict-paths
python -m latent_void discover-dataset --config configs/zaratan_inpaint360gs_bag.yaml
python scripts/check_sam3_access.py --download
INSTALL_GPU_DEPS=1 DOWNLOAD_DIFFSPLAT_CKPTS=0 MAX_JOBS=4 scripts/setup_zaratan_deps.sh
```

## What Fails Or Is Not Ready

- First real H100 geometry job was submitted and ran:
  - job id: `19185136`
  - node: `gpu-a6-4.zaratan.umd.edu`
  - result: failed quickly because `tools/preprocess_geometry.py` could not
    import `latent_void` when run as a script on the Slurm worker.
  - fix: `7cea7f7 Allow geometry tool imports under Slurm`
- Replacement H100 geometry job is queued:
  - job id: `19185139`
  - partition: `gpu-h100`
  - state at latest check: `PENDING`
  - reason: `Priority`
- Slurm smoke job was submitted twice and canceled while pending:
  - first used the old `gpu-h100` smoke template and was pending for priority.
  - second used the new `debug` CPU template and was pending for resources.
  - no smoke job is left queued.
- `configs/zaratan_inpaint360gs_bag.yaml` points at the downloaded Zaratan
  dataset/repo/checkpoint locations.
- The installed DiffSplat and SAM 3 wrappers are wired to real Zaratan paths.
- DiffSplat scene exporter is implemented but not yet H100-tested with the full
  DiffSplat GPU dependency stack.
- The heavy dependency setup that compiles DiffSplat's Gaussian rasterizer is
  added but not yet fully validated on Zaratan. First attempt installed torch,
  SAM 3, DiffSplat requirements, Diffusers 0.35.2, and Transformers, then failed
  while building `diff-gaussian-rasterization` because pip build isolation could
  not import the installed `torch`. The setup script now uses
  `--no-build-isolation`, loads `cuda/12.3.0/gcc/11.3.0/zen2`, and sets
  `TORCH_CUDA_ARCH_LIST=9.0` for H100 builds. A manual compile succeeded after
  setting `CUDA_HOME` from `nvcc`; setup now overwrites stale `CUDA_HOME` values
  when `nvcc` is visible. Setup also keeps `setuptools<82` because Torch 2.11
  declares that upper bound. If `nvcc` is still hidden in non-interactive
  module execution, setup now falls back to Zaratan's CVMFS CUDA 12.3 install.
- The SAM 3 Transformers/repo backend auto-selection is implemented but not yet
  H100-tested on a real image.
- Research-quality latent inpainting logic is not complete yet.
- Zaratan SSH clone from GitHub failed due to missing public-key auth.
- Zaratan HTTPS clone/pull works.
- Zaratan login-node Python has `yaml` but not `numpy`.
- Tensor-heavy stages such as `fuse` and `inpaint` require a Python environment
  with NumPy, likely inside the GPU/model environment or container.
- Zaratan's Python module prepends global package paths via `PYTHONPATH`; setup
  and Slurm scripts now unset `PYTHONPATH` before activating the project venv.
- Avoid self-upgrading `pip` inside the active Zaratan venv; that step hung
  once after uninstalling the existing pip package and was removed from setup.
- The fallback latent inpaint is only a plumbing test. It is not research-quality
  native latent inpainting.
- The Gaussian `.npz` contract currently expects precomputed `uvs` and
  `visibility`. If GSRecon exports positions/cameras instead, projection must be
  added upstream or implemented in this repo.
- SAM 3 HF auth is resolved for the active Zaratan account.
- DiffSplat upstream has training and generation scripts, but no direct
  `run_gsrecon.py` scene-export CLI. A wrapper contract now exists at
  `tools/run_gsrecon_export.py`; the actual exporter must be implemented once
  checkpoints/environment details are available.
- DiffSplat's public GSRecon checkpoint is trained for GObjaverse-style
  four-view object inputs and expects RGB plus camera-derived Plucker rays and,
  by default, normal/coordinate channels. Inpaint360GS gives real scene RGB and
  COLMAP poses, so `tools/preprocess_geometry.py` now generates Marigold depth,
  Marigold normals, and reprojected coordinate maps.

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

Wait for the replacement H100 geometry job:

```bash
squeue -j 19185139 -o '%.18i %.9P %.30j %.8T %.10M %.10l %.30R'
tail -120 logs/latent-void-geom-19185139.out
tail -120 logs/latent-void-geom-19185139.err
```

After geometry succeeds, continue with:

```bash
sbatch slurm/zaratan_reconstruct.sbatch configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/inpaint360gs_bag_mini
sbatch slurm/zaratan_segment.sbatch configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/inpaint360gs_bag_mini
```

Remaining model-adapter blocker:

- GSVAE/native latent inpainting adapter beyond the fallback plumbing fill.
