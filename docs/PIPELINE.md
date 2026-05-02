# Pipeline Notes

## Intended MVP

The MVP is native 3D scene removal:

1. Load a real multi-view Inpaint360GS scene with COLMAP camera metadata.
2. Generate missing geometry channels: Marigold depth, Marigold normals, and
   coordinate maps from depth plus COLMAP reprojection.
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

### GSRecon

Configured by `external.gsrecon_command`. It receives formatted values such as
`{dataset_root}`, `{scene_id}`, and `{gsrecon_dir}`.

Expected useful outputs:

- `gaussians.npz`: Gaussian arrays. Minimum for mask fusion is either
  `positions` with camera metadata or precomputed `uvs` and `visibility`.
- `latent.npy`: optional GSVAE latent tensor.
- `gs_grid.npy`: optional 12-channel Gaussian grid.
- rendered RGB/depth/alpha views for SAM 3 and diagnostics.

Current caveat: upstream DiffSplat documents GSRecon inference through issue
discussion rather than a packaged CLI. `tools/run_gsrecon_export.py` now
implements the expected adapter path against `geometry_manifest.json`, but it
still needs the full DiffSplat GPU dependency stack before it can run on H100.

### SAM 3

Configured by `external.sam3_command`. It receives a JSON manifest containing
views and output paths. It should emit one mask per view as `.npy` or image
files under the configured mask directory.

### Latent Inpainting

Configured by `external.latent_inpaint_command`. If omitted and
`pipeline.allow_fallback_inpaint` is true, the local fallback fills masked latent
cells with unmasked channel means. That fallback is for plumbing tests only, not
the final research-quality model.

## Zaratan

Use the `debug` partition for dry-run smoke checks and `gpu-h100` with
`msml612pcs3-class` for heavy model stages. The Slurm templates are thin wrappers
around `python3 -m latent_void` so the same configs can run locally and remotely.
