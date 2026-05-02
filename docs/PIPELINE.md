# Pipeline Notes

## Intended MVP

The MVP is native 3D scene removal:

1. Load a real multi-view Inpaint360GS scene with COLMAP camera metadata.
2. Run DiffSplat `GSRecon` to obtain structured Gaussian grids.
3. Encode those grids into GSVAE splat latents.
4. Render/load calibrated views.
5. Segment prompted objects using SAM 3 from multiple views.
6. Fuse masks into a Gaussian deletion mask and latent void mask.
7. Inpaint only the latent void.
8. Decode to Gaussians and render before/after diagnostics.

The code here does not vendor DiffSplat, SAM 3, or large checkpoints. It exposes
command adapters with explicit manifests and output contracts so Zaratan jobs
can call the installed model repositories.

## External Command Contracts

### GSRecon

Configured by `external.gsrecon_command`. It receives formatted values such as
`{dataset_root}`, `{scene_id}`, and `{gsrecon_dir}`.

Expected useful outputs:

- `gaussians.npz`: Gaussian arrays. Minimum for mask fusion is either
  `positions` with camera metadata or precomputed `uvs` and `visibility`.
- `latent.npy`: optional GSVAE latent tensor.
- rendered RGB/depth/alpha views for SAM 3 and diagnostics.

Current caveat: upstream DiffSplat documents GSRecon inference through issue
discussion rather than a packaged CLI, and its public checkpoint is trained for
GObjaverse-style four-view object inputs with RGB plus camera-derived geometry
channels. Real Inpaint360GS RGB scenes therefore need an explicit adapter for
normal/coordinate inputs or a retrained RGB-only scene encoder.

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

Use the `gpu-h100` partition and `msml612pcs3-class` account by default. The
Slurm templates are intentionally thin wrappers around `python3 -m latent_void`
so the same configs can run locally and remotely.
