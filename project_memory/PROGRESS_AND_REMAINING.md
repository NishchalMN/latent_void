# Progress And Remaining Work

Last updated: 2026-05-02

## Goal

Build `latent_void` as a native 3D object-removal pipeline for real multi-view
scenes. The target flow is:

1. Load an Inpaint360GS scene.
2. Generate missing geometry channels from RGB views.
3. Reconstruct a GSVAE-compatible Gaussian scene with DiffSplat GSRecon.
4. Segment an object across calibrated views with SAM 3.
5. Fuse 2D masks into a Gaussian-level 3D deletion mask.
6. Convert the deletion region into a latent void.
7. Inpaint the void in latent/Gaussian space.
8. Decode and render before/after results.

## What We Built

- Initialized `latent_void` as a standalone Git repo.
- Set up GitHub sync:
  - local repo: `/fs/nexus-scratch/gnanesh/DiffSplat/latent_void`
  - GitHub: `NishchalMN/latent_void`
  - Zaratan repo: `/home/gnanesh/scratch.msml612pcs3/latent_void`
- Added config-driven project structure for datasets, checkpoints, prompts,
  geometry settings, pipeline settings, and external tool commands.
- Added Inpaint360GS dataset discovery and COLMAP camera loading.
- Added Marigold-based geometry preprocessing:
  - RGB loading
  - depth estimation
  - normal estimation
  - coordinate-map reprojection from depth plus camera intrinsics/extrinsics
  - geometry manifests for downstream GSRecon
- Added DiffSplat GSRecon/GSVAE export:
  - consumes geometry manifests
  - writes `gaussians.npz`
  - writes `gs_grid.npy`
  - writes `latent.npy`
  - stores projection arrays for multi-view mask fusion
- Added SAM 3 multi-view segmentation wrapper:
  - supports prompt-based segmentation
  - uses Transformers backend on Zaratan
  - writes one mask per selected view
  - resizes masks to match the geometry grid
- Added mask fusion:
  - projects Gaussians into each masked view
  - votes across views
  - creates a Gaussian deletion mask
  - creates a latent void mask
  - writes deleted-Gaussian outputs and manifests
- Added MVP latent inpainting plumbing:
  - current fallback fills masked latent regions enough to prove end-to-end data
    flow
  - this is not the final research-quality inpainting model
- Added DiffSplat/GSVAE render diagnostics:
  - decodes before and after latents
  - renders RGB, alpha, and depth diagnostics
  - supports Zaratan's installed rasterizer through a compatibility shim
- Added Zaratan setup scripts:
  - dataset download
  - Marigold snapshot download
  - DiffSplat auxiliary VAE snapshot download
  - SAM 3 checkpoint/access check
  - GPU dependency setup
  - direct `srun` stage helper
- Added project memory docs for decisions, context, plan, Zaratan workflow,
  status, failures, and fixes.

## Important Fixes

- Fixed script import paths for Slurm/`srun` workers.
- Moved compute-node model use away from online Hugging Face downloads by
  downloading local Marigold and DiffSplat auxiliary VAE snapshots on the login
  node.
- Patched DiffSplat compatibility with Zaratan's newer Transformers install:
  - restored/aliased old modeling utility imports
  - restored `find_pruneable_heads_and_indices`
- Stubbed optional `wandb` imports for inference/export paths.
- Mapped DiffSplat's hardcoded auxiliary VAE repo IDs to local snapshots:
  - `madebyollin/sdxl-vae-fp16-fix`
  - `madebyollin/taesdxl`
- Patched Zaratan rasterizer compatibility:
  - drops unsupported `require_coord`
  - expands older 6-tensor rasterizer outputs into DiffSplat's expected
    8-tensor render shape
- Made render diagnostics accept `raw_depth` when DiffSplat does not return a
  `depth` key.

## Verified H100 MVP Run

The first end-to-end MVP run completed on Zaratan H100:

- Slurm allocation: `19186674`
- Node: `gpu-a6-4`
- Output directory:
  `/home/gnanesh/scratch.msml612pcs3/latent_void/runs/inpaint360gs_bag_srun_h100`
- Dataset/scene: Inpaint360GS `bag`
- Selected views: 16
- GSRecon/GSVAE output:
  - `262,144` Gaussians
  - `gsrecon/gaussians.npz`
  - `gsrecon/gs_grid.npy`
  - `gsrecon/latent.npy`
- SAM 3 output:
  - prompt: `bag`
  - backend: Transformers
  - 16 masks
  - scores roughly `0.92` to `0.97`
- 3D deletion:
  - `1,701` deleted Gaussians out of `262,144`
- Inpaint output:
  - `inpaint/latent_inpainted.npy`
- Render output:
  - `renders/render_status.json` reports `ok: true`
  - 8 before RGB views
  - 8 after RGB views
  - alpha and depth diagnostics
  - sampled PNGs are nonblank

## What Is Working Now

- The repo can be developed locally, committed, pushed, and pulled on Zaratan.
- Zaratan can run heavy stages through interactive `srun`.
- The pipeline can process one real Inpaint360GS scene end to end through:
  geometry -> GSRecon/GSVAE -> SAM 3 -> 3D void -> fallback latent fill ->
  before/after renders.
- All current local unit tests pass:
  - 20 tests total
  - 2 skipped locally due missing Pillow in the local environment

## What Is Left

### 1. Replace Fallback Inpaint With Real Latent Inpainting

The current fallback inpaint proves plumbing, not final quality. The main
remaining research/implementation work is to add a real masked latent denoising
or optimization module that:

- keeps unmasked latents/Gaussians fixed
- updates only the latent void region
- uses scene/context conditioning
- optionally injects attention from the original scene when compatible
- decodes to artifact-free Gaussian content

### 2. Add Self-Supervised Fine-Tuning

If pretrained latent behavior is insufficient, add H100 training:

- use DL3DV-10K and/or configured scene datasets
- create random object-like 3D masks
- reconstruct the known intact scene
- optimize masked latent reconstruction
- add render consistency, opacity, depth, and artifact cleanup losses

### 3. Improve 3D Void Quality

Current Gaussian deletion uses projected mask voting. It should be improved with:

- connected-component cleanup
- mask confidence filtering
- depth-aware filtering
- shadow/effect prompt support
- dilation/erosion controls in 2D and 3D
- safeguards against deleting background Gaussians

### 4. Evaluate Visual Quality

We still need systematic quality checks:

- inspect before/after render grids
- generate comparison videos
- measure multi-view consistency
- detect holes, floaters, and view-dependent artifacts
- log mask coverage and deletion statistics

### 5. Run More Scenes And Prompts

The proven run is one scene and one prompt. Next validation should include:

- more Inpaint360GS scenes
- different object prompts
- optional shadow/effect prompts
- SPIn-NeRF adapter later
- DL3DV loader later for training data

### 6. Harden Zaratan Operations

The current tmux + `srun` workflow works, but should be made smoother:

- keep a documented interactive `srun` recipe
- add resumable stage commands
- avoid stale scrollback confusion in tmux
- add artifact checks after each stage
- add a single summary command for run status

## Current Bottom Line

The infrastructure and first H100 end-to-end MVP are working. The project is no
longer blocked on dataset loading, SAM 3 access, GSRecon input channels,
DiffSplat compatibility, or Zaratan execution.

The main unfinished part is the actual high-quality native latent inpainting
model. Everything around it is now in place enough to support that work.
