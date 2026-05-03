# Progress And Remaining Work

Last updated: 2026-05-03

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

- Implemented the native DiffSplat latent inpainting plan as concrete repo
  contracts:
  - staged render diagnosis with `tools/diagnose_diffsplat_render.py`
  - DiffSplat-style geometry preprocessing profile in the active configs
  - local object/void crop manifest extraction with adjusted intrinsics
  - H100 training-data and masked latent denoiser specifications
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
- From the direct Zaratan checkout, the active config now uses repo-relative
  paths and validates with `--strict-paths`.
- The `bag` source RGB views and SAM 3 mask overlays are visually correct. The
  useful current review image is
  `runs/visual_inspection/inpaint360gs_bag_input_masks_void_preview.png`.
- The pipeline now has an external native latent context-inpaint command at
  `tools/inpaint_latent_context.py`. It keeps unmasked latents fixed and fills
  only the masked void cells, so the Zaratan config no longer relies on the
  internal fallback branch.
- Mask fusion now has configurable score, component-size, erosion, and dilation
  controls, and records those settings in `void_manifest.json`.
- Additional scenes were validated at the dataset and dry-run command-contract
  level: `car` and `garden_toys`.
- All current local unit tests pass:
  - 23 tests total

## What Is Left

### 1. Improve Local Patch Reconstruction

`tools/diagnose_diffsplat_render.py` has now been run on both the existing full
`bag` artifacts and a mask-centered local patch artifact. The results separate:

- GSRecon/direct Gaussian grid failure
- GSVAE reconstruction failure
- edited latent failure

The finding is that full-scene reconstruction is already degraded before
inpainting, while the local patch path is more coherent but still blurry/warped.
Local 3D canonicalization and GObjaverse-style white outside-mask compositing
have now been implemented and tested. The best current variant is the
object-centered white-background patch, but it is still too sparse/blurry for
final results.

The known-good object sample route is now complete. We downloaded the official
GObjaverse example archive, prepared it with `tools/prepare_gobjaverse_sample.py`,
and ran the same GSRecon/GSVAE diagnostic path on H100. The official car sample
renders recognizably, so the DiffSplat install/checkpoints/renderer path is
healthy on in-domain object data.

The next engineering step is no longer more harmonic inpainting or environment
debugging. It is adapting GSRecon/GSVAE or the local patch representation to real
scene patches using the training contract in `configs/native_latent_training_example.yaml`.
The local adaptation mechanics are executable: a 32-sample masked-latent dataset
was generated from the best current `bag` patch, and a tiny H100 smoke trainer
reduced masked latent MSE from `2.1880` to `0.5790`. This validates the
data/optimization loop, not final visual quality.

### 2. Replace Baseline Inpaint With A Learned Latent Denoiser

The fallback branch has been replaced in the active Zaratan config by a
context-only latent inpainting command. This proves the external inpaint contract
and enforces unmasked-latent invariance, but it is still not the final
research-quality model. The H100 training contract now lives in
`docs/NATIVE_DIFFSPLAT_LATENT_INPAINTING.md` and
`configs/native_latent_training_example.yaml`.

The learned denoiser must keep unmasked latents/Gaussians fixed, update only the
latent void region, use scene/camera context, and optimize masked latent plus
held-out render consistency losses. The current smoke trainer should be replaced
with a DiffSplat/PixArt/GSDiff initialized masked denoiser once the reconstruction
target quality is sufficient.

Patch latent void/inpaint was run on the best current patch diagnostic, but it
is not yet meaningful as a final result because patch reconstruction remains the
dominant visual failure.

### 3. Improve 3D Void Quality

Current Gaussian deletion uses projected mask voting. It should be improved with:

- connected-component cleanup
- mask confidence filtering
- depth-aware filtering
- shadow/effect prompt support
- dilation/erosion controls in 2D and 3D
- safeguards against deleting background Gaussians

### 4. Evaluate Visual Quality

We still need systematic quality checks. The immediate finding is that current
DiffSplat/GSVAE render diagnostics for `bag` are not visually interpretable,
even though the input views and SAM masks are sensible. Before evaluating
inpainting quality, fix the reconstruction/render path so before renders look
like the scene.

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

The main unfinished part is still the high-quality native latent inpainting
model, but the latest H100 diagnostics show that learned local-patch
reconstruction/adaptation is also needed before the inpaint quality can be
fairly judged.
