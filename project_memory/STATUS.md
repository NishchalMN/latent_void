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
- Latest pushed repo/code commit before the current `srun` wrapper update:
  `9eafb28 Use local Marigold snapshots on Zaratan`

Implemented and verified locally:

- Config validation with env-var default expansion.
- Inpaint360GS dataset discovery.
- DiffSplat/GSRecon external command dry-run rendering.
- SAM 3 external command dry-run rendering.
- Marigold geometry preprocessing command dry-run rendering.
- Local coordinate-map reprojection from COLMAP cameras and depth maps.
- Geometry preprocessing now normalizes COLMAP camera poses into a
  DiffSplat-like object-scale frame before writing Plucker/coordinate inputs.
  It supports both scene min/max coordinate encoding and DiffSplat's
  `coord * 0.5 + 0.5` style encoding.
- GSRecon export adapter now accepts `geometry_manifest.json` and is able to
  export `gaussians.npz`, `gs_grid.npy`, and `latent.npy` once DiffSplat GPU
  dependencies are available.
- Zaratan setup now has a heavy dependency path that installs Marigold-capable
  Diffusers plus DiffSplat's RaDe-GS `diff-gaussian-rasterization` extension.
- SAM 3 multiview wrapper can run through the official Hugging Face
  Transformers backend or the cloned `facebookresearch/sam3` repo backend.
- SAM 3 masks are resized to the geometry input resolution in the Zaratan
  command so mask fusion uses the same pixel grid as projected Gaussians.
- Marigold depth predictions with trailing singleton channels are squeezed to
  2D before resizing/reprojection.
- DiffSplat Gaussian channel metadata now matches the upstream GSVAE order:
  RGB, scale, rotation quaternion, opacity, depth.
- GSRecon export now stores Gaussian grid shape, latent shape, and GSVAE grid
  shape in `gaussians.npz`.
- Latent void masks now use GSRecon's `[B, V, H, W]` Gaussian grid shape and
  the actual `latent.npy` shape instead of flattening all Gaussians into an
  arbitrary square grid.
- Render diagnostics are now wired through `tools/render_latent_scene.py`,
  which decodes scaled GSVAE latents and renders before/after RGB, alpha, and
  depth outputs with DiffSplat's renderer.
- Marigold snapshots are downloaded with `scripts/download_marigold.py`; the
  Zaratan config points at local `checkpoints/marigold/depth-v1-1` and
  `checkpoints/marigold/normals-v1-1` paths so Slurm jobs do not rely on
  compute-node network access to Hugging Face.
- `scripts/zaratan_srun_stage.sh` now runs staged H100 geometry,
  GSRecon/GSVAE reconstruction, SAM 3 segmentation, and final
  fuse/inpaint/render diagnostics through direct interactive `srun`.
- Geometry external command formatting now passes `pipeline.max_views` through
  as `--max-views`, so `--set pipeline.max_views=...` affects Marigold
  preprocessing instead of only the dataset summary.
- GSRecon export now patches DiffSplat's older Transformers import expectation
  by aliasing `transformers.pytorch_utils.apply_chunking_to_forward` onto
  `transformers.modeling_utils` before importing DiffSplat.
- Multi-view mask fusion with synthetic projected Gaussian data.
- Latent void mask generation.
- Fallback latent fill for plumbing tests.
- Legacy Slurm templates remain in `slurm/`, but the active Zaratan workflow
  prefers direct `srun` from tmux.
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
bash -n scripts/zaratan_srun_stage.sh
python3 -m latent_void run --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/zaratan_bag_dry --dry-run
python3 -m latent_void prepare-geometry --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/zaratan_bag_geometry_dry --dry-run
python3 -m py_compile latent_void/geometry.py latent_void/pipeline.py latent_void/cli.py tools/preprocess_geometry.py tools/run_gsrecon_export.py tools/run_sam3_multiview.py
```

Latest local validation after the mask/grid fixes:

```bash
python3 -m py_compile tools/preprocess_geometry.py tools/run_sam3_multiview.py tools/run_gsrecon_export.py latent_void/latent.py latent_void/pipeline.py latent_void/masks.py latent_void/gaussians.py
python3 -m unittest discover -s tests
python3 -m latent_void run --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/local_dry_after_mask_fixes --set pipeline.max_views=4 --dry-run
python3 -m latent_void render --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/local_render_dry --dry-run
python3 -m latent_void prepare-geometry --config configs/zaratan_inpaint360gs_bag.yaml --set pipeline.max_views=4 --set project.output_dir=runs/local_max_views_contract --dry-run
```

The local smoke environment lacks Pillow, so the two Pillow-dependent unit
checks are skipped locally. They should run inside the Zaratan venv.

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

Latest Zaratan post-pull validation:

```bash
git pull --ff-only
python -m unittest discover -s tests
python -m py_compile tools/preprocess_geometry.py tools/run_sam3_multiview.py tools/run_gsrecon_export.py tools/render_latent_scene.py latent_void/latent.py latent_void/pipeline.py latent_void/masks.py latent_void/gaussians.py
python -m latent_void run --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/zaratan_bag_postpull_dry --dry-run
python -m latent_void render --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/zaratan_render_postpull_dry --dry-run
```

Result: 17 tests passed; post-pull dry-run rendered geometry, GSRecon, and SAM
3 commands correctly, including `--resize-to 256` for SAM masks. Render dry-run
also produced the expected `tools/render_latent_scene.py` command.

## What Fails Or Is Not Ready

- First real H100 geometry job was submitted and ran:
  - job id: `19185136`
  - node: `gpu-a6-4.zaratan.umd.edu`
  - result: failed quickly because `tools/preprocess_geometry.py` could not
    import `latent_void` when run as a script on the Slurm worker.
  - fix: `7cea7f7 Allow geometry tool imports under Slurm`
- Replacement H100 geometry job ran and was canceled:
  - job id: `19185139`
  - partition: `gpu-h100`
  - node: `gpu-a6-9`
  - started: `2026-05-02T19:29:24`
  - canceled after diagnostics showed no progress before Marigold inference
- Dependent continuation chain was queued against `19185139` and then canceled:
  - `19186465`: `latent-void-gsrecon`, dependency `afterok:19185139`
  - `19186466`: `latent-void-sam3`, dependency `afterok:19186465`
  - `19186467`: `latent-void-inpaint`, dependency `afterok:19186466`
  - final job runs `python -m latent_void run` with
    `--skip-geometry --skip-reconstruct --skip-segment`, so it performs fusion,
    fallback latent inpaint plumbing, and render diagnostics from the staged
    outputs.
- The first replacement geometry run `19185139` was canceled after diagnostics
  showed the process sleeping for 11+ minutes with 0 MiB GPU usage and no
  geometry files. It was stuck before Marigold inference, likely while resolving
  Hugging Face model files from the compute node. The chain jobs
  `19186465`, `19186466`, and `19186467` were canceled with it and must be
  resubmitted after local Marigold snapshots are downloaded.
- Backup A100 geometry job was briefly submitted with a separate output directory
  and then canceled so the bring-up stays focused on H100:
  - job id: `19185424`
  - partition: `gpu-a100`
  - output: `runs/inpaint360gs_bag_mini_a100`
  - final state at latest check: canceled / no longer in `squeue`
- A later A100 submission initially failed because only the partition was
  overridden and the H100 GRES remained active. The corrected command must
  override both `--partition=gpu-a100` and `--gres=gpu:a100:1`.
- The corrected A100 job was then canceled so the active bring-up remains the
  H100 job:
  - job id: `19186443`
  - estimated start before cancellation: `2026-05-02T21:00:00`
  - final state at latest check: canceled / no longer in `squeue`
- `sbatch --test-only --partition=gpu --gres=gpu:h100:1` estimated a later
  start (`2026-05-03T23:53:18`) than the active `gpu-h100` job, so do not
  replace `19185139` with the generic `gpu` partition.
- Zaratan file-count quota issue encountered and resolved for this working
  copy:
  - project group `zt-msml612pcs3` hit BeegFS file-count hard quota
    (`527979 / 450000` after cache cleanup), causing even `.git/write_test` and
    `git pull --ff-only` to fail with `Disk quota exceeded`.
  - removed Python caches and pip cache, which reduced this project by roughly
    17k files but did not fully solve the group-level quota.
  - changed the working-copy group to `zt-msml605`, which has headroom
    (`1435440 / 4500000` files at the check).
  - after the group change, write tests and `git pull --ff-only` succeeded.
  - `zt-msml612pcs3` returned below quota (`337797 / 450000` files at the
    check).
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
  Marigold normals, and reprojected coordinate maps. Depth is saved and used for
  coordinate reprojection; it is not concatenated into the current public
  GSRecon checkpoint input because upstream `input_depth` is not enabled.

## Important Current Artifacts

Local files:

- `configs/inpaint360gs_example.yaml`
- `scripts/zaratan_srun_stage.sh`
- `slurm/zaratan_smoke.sbatch`
- `slurm/zaratan_inpaint.sbatch`
- `docs/PIPELINE.md`
- `docs/ZARATAN.md`
- `tests/test_pipeline.py`

Generated local dry-run artifacts are ignored by Git under `runs/`.

## Next Best Step

Continue the live direct `srun` H100 bring-up in the `zaratan` tmux session.
Current run directory:

```bash
runs/inpaint360gs_bag_srun_h100
```

Run stages sequentially with `scripts/zaratan_srun_stage.sh`: `geometry`,
`reconstruct`, `segment`, then `finish`.

Latest Zaratan geometry note:

- Direct `srun` geometry job `19186495` completed successfully and wrote
  `runs/inpaint360gs_bag_srun_h100/geometry/geometry_manifest.json`.
- That run processed 16 views because the older geometry command did not yet
  pass the `--set pipeline.max_views=4` override into `tools/preprocess_geometry.py`.
  The local fix after that run adds the missing `--max-views` command contract.
- Direct `srun` reconstruct job `19186528` allocated on
  `gpu-a6-4.zaratan.umd.edu` and failed at DiffSplat import time because
  Zaratan has Transformers `5.7.0`; the symbol
  `apply_chunking_to_forward` moved from `transformers.modeling_utils` to
  `transformers.pytorch_utils`. The adapter shim fixes this for the next run.

Remaining model-adapter blocker:

- GSVAE/native latent inpainting adapter beyond the fallback plumbing fill.
