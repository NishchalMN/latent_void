# Pipeline Notes

## Intended MVP

The MVP is native 3D scene removal:

1. Load a real multi-view Inpaint360GS scene with COLMAP camera metadata.
2. Generate missing geometry channels: Marigold depth, Marigold normals, and
   coordinate maps from depth plus COLMAP reprojection. The preprocessing stage
   also normalizes COLMAP camera poses into a DiffSplat-like object-scale frame
   before writing the manifest used by GSRecon.
3. Run DiffSplat `GSRecon` to obtain structured Gaussian grids.
4. Encode those grids into GSVAE splat latents.
5. Render/load calibrated views.
6. Segment prompted objects using SAM 3 from multiple views.
7. Fuse masks into a Gaussian deletion mask and latent void mask.
8. Inpaint only the latent void.
9. Decode to Gaussians and render before/after diagnostics.

The code here does not vendor DiffSplat, SAM 3, or large checkpoints. It exposes
command adapters with explicit manifests and output contracts so Zaratan jobs
can call the installed model repositories.

## External Command Contracts

### Geometry Preprocessing

Configured by `external.geometry_command`. It generates:

- resized RGB tensors in `[0, 1]`
- Marigold depth maps
- Marigold surface normals converted to `[0, 1]`
- raw world coordinate maps from COLMAP reprojection
- normalized coordinate maps for GSRecon
- `geometry_manifest.json`

The default Zaratan config now follows the stricter DiffSplat-style profile:
RGB is composited over white when alpha exists, camera normalization uses the
first selected view as the canonical reference, and coordinate maps use
`geometry.coord_mode: diffsplat` (`coord * 0.5 + 0.5`). The geometry manifest
records these preprocessing choices so reconstruction runs can be compared
against the original GObjaverse contract.

The current public DiffSplat GSRecon/GSVAE options concatenate RGB, normal, and
coordinate maps as model inputs; Marigold depth is saved and used to compute the
coordinate maps. The exported 12-channel GSVAE grid follows DiffSplat's actual
order: RGB, scale, rotation quaternion, opacity, depth.

On Zaratan, Marigold should be pre-downloaded with
`scripts/download_marigold.py` and referenced by local checkpoint paths in the
config. The first H100 geometry attempt stalled before GPU use while resolving
Marigold from Hugging Face inside the Slurm allocation, so compute jobs should
not depend on outbound model downloads.

DiffSplat's SDXL GSVAE also loads two Diffusers VAE repos at model
construction time: `madebyollin/sdxl-vae-fp16-fix` and `madebyollin/taesdxl`.
They must be downloaded on the login node with `scripts/download_diffsplat_aux.py`
and passed through `--sdxl-vae-path` and `--tiny-vae-path`; otherwise offline
H100 jobs fail before GSRecon inference starts.

### GSRecon

Configured by `external.gsrecon_command`. It receives formatted values such as
`{dataset_root}`, `{scene_id}`, and `{gsrecon_dir}`.

Expected useful outputs:

- `gaussians.npz`: Gaussian arrays. Minimum for mask fusion is either
  `positions` with camera metadata or precomputed `uvs` and `visibility`.
  The local exporter also stores `gaussian_grid_shape`, `latent_shape`, and
  `gs_grid_shape` so a Gaussian deletion mask can be mapped back to the correct
  per-view latent grid.
- `latent.npy`: optional GSVAE latent tensor.
- `gs_grid.npy`: optional 12-channel Gaussian grid.
- rendered RGB/depth/alpha views for SAM 3 and diagnostics.

Current caveat: upstream DiffSplat documents GSRecon inference through issue
discussion rather than a packaged CLI. `tools/run_gsrecon_export.py` now
implements the expected adapter path against `geometry_manifest.json`, but it
still needs the full DiffSplat GPU dependency stack before it can run on H100.
The Zaratan setup script installs the RaDe-GS `diff-gaussian-rasterization`
extension in the heavy dependency path because DiffSplat imports that renderer
during model initialization.

The adapter patches known runtime compatibility issues with current Zaratan
packages: Transformers 5 moved/removed a few legacy symbols DiffSplat imports,
and DiffSplat imports `wandb` logging helpers even for inference. The adapter
provides small compatibility shims and validates local auxiliary VAE snapshots
before doing heavy model work.

### SAM 3

Configured by `external.sam3_command`. It receives a JSON manifest containing
views and output paths. It should emit one mask per view as `.npy` or image
files under the configured mask directory.

The local wrapper supports `--backend auto`, `--backend transformers`, and
`--backend repo`. Auto mode prefers the official Hugging Face Transformers SAM3
API and falls back to the cloned `facebookresearch/sam3` repository path. This
keeps Zaratan usable even if one backend has a Python/CUDA dependency mismatch.
Masks are resized to the geometry input resolution in the Zaratan config so
Gaussian projections and SAM masks share pixel coordinates during fusion.

### Latent Inpainting

Configured by `external.latent_inpaint_command`. If omitted and
`pipeline.allow_fallback_inpaint` is true, the local fallback fills masked latent
cells with unmasked channel means. That fallback is for plumbing tests only, not
the final research-quality model.

The active Zaratan config uses `tools/inpaint_latent_context.py` as the first
external native latent baseline. It performs a context-only harmonic fill inside
the latent void and asserts that unmasked latent cells remain unchanged. This is
better than the internal fallback branch because it exercises the external stage
contract, but it is still a baseline rather than the final learned denoiser.

Mask fusion can be tuned with `pipeline.mask_score_threshold`,
`pipeline.mask_min_area`, `pipeline.mask_max_area_fraction`,
`pipeline.mask_erode_pixels`, and `pipeline.mask_dilate_pixels`; these cleanup
settings are recorded in `void_manifest.json`.

### Render Diagnostics

Configured by `external.render_command`. The local `tools/render_latent_scene.py`
adapter decodes scaled GSVAE splat latents back to 12-channel Gaussian grids and
renders RGB, alpha, and depth diagnostics through DiffSplat's renderer. It can
render both the original `latent.npy` and the inpainted
`latent_inpainted.npy`, producing `before/` and `after/` directories under the
run's render folder.

For reconstruction debugging, use `tools/diagnose_diffsplat_render.py`. It
renders direct `gs_grid.npy`, decoded original `latent.npy`, and decoded edited
latent outputs into separate folders so failures can be assigned to GSRecon,
GSVAE reconstruction, or latent editing.

`tools/prepare_gobjaverse_sample.py` prepares the official GObjaverse
`render_data_examples.zip` object for the same GSRecon adapter. It mirrors
DiffSplat's GObjaverse loader conventions and is the preferred sanity check for
whether the installed checkpoints and renderer work on in-domain object data.

### Local Patch Path

`tools/extract_local_patch_manifest.py` creates a local object-scale input
surface from `geometry_manifest.json` plus the SAM mask directory. It crops RGB,
normal, coordinate, depth, and mask arrays around the target object, resizes them
to the DiffSplat input size, and rewrites intrinsics into crop coordinates. It
can also transform raw coordinate maps and camera poses into a local
mask-centered canonical frame with `--canonicalize-3d`, and it composites RGB,
normal, and encoded coordinate channels to white outside the object mask by
default to better match DiffSplat's GObjaverse loader.

The resulting `local_patch_manifest.json` can be passed to the GSRecon exporter
to avoid feeding the full unbounded Inpaint360GS scene into an object-centric
DiffSplat model. Current H100 diagnostics show this is a better debugging
surface than the full scene, but not yet a final-quality reconstruction path.

The native latent training design and H100 data-generation contract live in
`docs/NATIVE_DIFFSPLAT_LATENT_INPAINTING.md`; the corresponding template config
is `configs/native_latent_training_example.yaml`.

The scene-local training path is now split into explicit tools:

- `tools/build_scene_patch_dataset.py` discovers existing geometry/mask runs and
  creates a multi-scene dataset of local patch manifests.
- `latent_void.datasets.DL3DVDataset` adds DL3DV-style calibrated scene discovery
  through the same view interface used by Inpaint360GS.
- `tools/generate_patch_teacher_targets.py` writes input/held-out patch splits
  with RGB, alpha, depth, normal, coord, camera, and intrinsics references.
- `tools/train_recon_adapter.py` runs the first reconstruction-adapter smoke
  loop with RGB/alpha/depth consistency losses while keeping GSVAE frozen by
  contract.
- `tools/evaluate_recon_gates.py` compares direct GS grid diagnostics against
  GSVAE reconstruction diagnostics and records pass/fail gates.
- `tools/generate_native_latent_training_data.py` packages local patch latents
  into self-supervised masked latent samples, and
  `tools/train_masked_latent_denoiser.py` trains a residual masked latent denoiser
  with hard unmasked-cell clamping and optional compatible checkpoint
  initialization.
- `tools/merge_local_inpaint.py` removes or suppresses deleted full-scene
  Gaussians and appends finite, visible decoded local inpainted Gaussians for
  final render diagnostics.

## Zaratan

Use login-node dry runs for smoke checks and direct `srun` allocations on
`gpu-h100` with `msml612pcs3-class` for heavy model stages. The preferred
Zaratan wrapper is `scripts/zaratan_srun_stage.sh`, which runs the same
`python -m latent_void` stages interactively inside the `zaratan` tmux session.
For first H100 validation, run `geometry`, `reconstruct`, `segment`, and
`finish` with `--set project.output_dir=...` pointing at the same mini run
directory.
