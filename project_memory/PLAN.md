# Active Plan

Last updated: 2026-05-05

## 2026-05-05 Priority Pivot: Source Reconstruction First

Goal: make native DiffSplat branch renders (`direct_gs_grid`) visually
recognizable and baseline-like before spending additional time on masked latent
inpainting.

Context from latest run:

- Track B executed end-to-end (dataset generation, denoiser training, latent
  inpaint, render diagnostics).
- Training loss converged, but rendered diagnostics remained poor:
  `edited_latent` stayed close to `latent_reconstruction`, and both were far
  from clean Inpaint360GS baseline renders.
- This indicates the main blocker is upstream representation quality
  (GSRecon/GSVAE domain mismatch), not denoiser optimization.

Execution gates (hard requirements):

1. Gate A: `direct_gs_grid` is structurally recognizable.
2. Gate B: `latent_reconstruction` is close to `direct_gs_grid`.
3. Gate C: `edited_latent` improves masked region while preserving context.

Do not scale denoiser training until Gates A and B pass.

### Active Workstream (now)

1. Re-run local-patch and scene-patch GSRecon adaptation sweeps for `bag`:
   - `trainable=heads`
   - `trainable=heads_and_embed`
   - optional `trainable=last_blocks` if first two fail
2. Use fixed evaluation snapshots and early-stop rules to avoid long
   unproductive runs.
3. Compare each run against the current baseline-style references:
   `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_5000/renders/*.png`.
4. Promote only runs that materially improve Gate A visuals.
5. Resume latent inpainting experiments only after Gate A/B quality is met.

## Milestone 1: Real-Scene Native Latent Removal MVP

Target: one Inpaint360GS scene with one prompted object removed, inpainted
through GSVAE/DiffSplat latent space, and rendered as before/after diagnostics.

Implementation stages:

1. Configure a real Inpaint360GS scene on Zaratan.
2. Generate zero-training geometry side channels with Marigold depth, Marigold
   normals, and COLMAP coordinate reprojection.
3. Configure installed DiffSplat/GSRecon paths and pretrained weights.
4. Configure installed SAM 3 paths and pretrained weights.
5. Run GSRecon to create Gaussian grids, projected Gaussian metadata, and latent
   tensors.
6. Run SAM 3 over calibrated scene views with object prompt and optional shadow
   prompt.
7. Fuse SAM masks into a Gaussian deletion mask using projected `uvs` and
   `visibility`.
8. Create a latent void mask using the configured downsample strategy.
9. Run latent inpainting through a configured pretrained or optimization
   backend.
10. Decode or merge the result back into a renderable Gaussian scene.
11. Render before/after views, masks, depth diagnostics, and a turntable video.

## Milestone 2: Artifact Reduction

Target: reduce holes, floaters, shadow leftovers, and view-inconsistent seams.

Planned additions:

- Depth-aware cleanup around void boundaries.
- Opacity and visibility filtering for floaters.
- Shadow prompt path and offset/association logging.
- Diagnostics for per-view mask disagreement.
- Better latent mask dilation/erosion controls.

## Milestone 3: Self-Supervised Fine-Tuning

Target: improve native latent inpainting quality when pretrained components are
insufficient.

Planned additions:

- DL3DV-10K or compatible scene dataset loader.
- Random 3D/object-like mask generator.
- Training config and Slurm template.
- Losses for masked latent reconstruction, render consistency, depth,
  silhouette/opacity, and artifact cleanup.
- Checkpoint/resume/evaluation workflow.

## Current Code Interfaces

- `latent_void.datasets.Inpaint360GSDataset`: scene discovery and view manifest.
- `latent_void.external`: command adapter utilities.
- `latent_void.masks.Sam3CommandAdapter`: SAM 3 command wrapper.
- `latent_void.masks.fuse_gaussian_masks`: multi-view mask fusion.
- `latent_void.latent.latent_mask_from_gaussian_mask`: latent void mask creation.
- `latent_void.pipeline`: stage orchestration.
- `python3 -m latent_void`: CLI entrypoint.

## Acceptance Criteria For MVP

- A real Inpaint360GS scene is loaded from config.
- GSRecon writes the expected Gaussian/latent artifacts.
- SAM 3 writes masks for the target prompt.
- The fuse stage produces `gaussian_deletion_mask.npy` and
  `latent_void_mask.npy`.
- Inpainting writes an edited latent or Gaussian artifact.
- Final outputs include before/after visual renders.
- Unmasked scene regions remain visually stable.
