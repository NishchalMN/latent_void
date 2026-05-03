# Native DiffSplat Latent Inpainting

This note converts the native latent inpainting plan into repo-level contracts.
The final method remains DiffSplat-style: reconstruct or train local Gaussian
latents, mask the void in GSVAE latent space, update only the masked latent
region, decode through GSVAE, and render with DiffSplat's Gaussian renderer.

## Render Path Diagnosis

Use `tools/diagnose_diffsplat_render.py` after `reconstruct`, `fuse`, and
`inpaint` have produced `gs_grid.npy`, `latent.npy`, `gaussians.npz`, and
`latent_inpainted.npy`.

It renders three separate branches:

- `direct_gs_grid/`: the 12-channel GSRecon/GSVAE Gaussian grid before latent
  editing. If this is unreadable, the input adapter or GSRecon domain is failing.
- `latent_reconstruction/`: the original `latent.npy` decoded through GSVAE. If
  direct GS is readable but this is not, the GSVAE encode/decode path is failing.
- `edited_latent/`: the inpainted latent decoded through the same renderer. If
  the first two are readable but this is not, the inpainting stage is failing.

Example H100 command:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tools/diagnose_diffsplat_render.py \
  --diffsplat-root external/DiffSplat \
  --gsvae-weights checkpoints/diffsplat/gsvae_gobj265k_sdxl_fp16 \
  --sdxl-vae-path checkpoints/diffsplat_aux/sdxl-vae-fp16-fix \
  --tiny-vae-path checkpoints/diffsplat_aux/taesdxl \
  --gaussian-npz runs/inpaint360gs_bag/gsrecon/gaussians.npz \
  --gs-grid-path runs/inpaint360gs_bag/gsrecon/gs_grid.npy \
  --latent-path runs/inpaint360gs_bag/gsrecon/latent.npy \
  --compare-latent-path runs/inpaint360gs_bag/inpaint/latent_inpainted.npy \
  --output-dir runs/inpaint360gs_bag/render_diagnostics
```

Run with `--preflight-only` on the login node to validate paths before entering
an H100 allocation.

## DiffSplat Preprocessing Contract

The geometry adapter now records a DiffSplat preprocessing profile in
`geometry_manifest.json`. The Zaratan and example configs use:

- `geometry.preprocessing_profile: diffsplat_gobjaverse`
- `geometry.white_background: true`
- `geometry.camera_reference_index: 0`
- `geometry.coord_mode: diffsplat`

That means RGBA inputs are composited over white before resize, camera poses are
normalized with the first selected view as canonical reference, normal and
coordinate channels are written as CHW arrays in `[0, 1]`, and coordinates use
DiffSplat's `coord * 0.5 + 0.5` encoding instead of scene min/max encoding.

This does not by itself solve the object-vs-full-scene domain mismatch, but it
removes avoidable convention drift before deciding whether training is needed.

For an in-domain sanity check, use the official GObjaverse example archive:

```bash
python tools/prepare_gobjaverse_sample.py \
  --gobjaverse-dir data/gobjaverse_examples/campos_512_v4 \
  --diffsplat-root external/DiffSplat \
  --output-dir runs/visual_inspection/gobjaverse_official_example_geometry \
  --input-res 256 \
  --max-views 40
```

Then run `tools/run_gsrecon_export.py` and `tools/diagnose_diffsplat_render.py`
against that manifest. On Zaratan, this official car sample renders
recognizably, proving the DiffSplat install/checkpoints/renderer are healthy on
in-domain object data.

## Local Scene Patch Latents

Use `tools/extract_local_patch_manifest.py` to generate object-scale crop inputs
around the SAM mask before GSRecon/GSVAE inference:

```bash
python tools/extract_local_patch_manifest.py \
  --geometry-manifest runs/inpaint360gs_bag/geometry/geometry_manifest.json \
  --mask-dir runs/inpaint360gs_bag/masks \
  --output-dir runs/inpaint360gs_bag/local_patch \
  --crop-size 256 \
  --crop-scale 1.75 \
  --canonicalize-3d \
  --canonical-mode object_centered \
  --white-background-outside-mask
```

The tool crops RGB, normals, coordinates, depth, and masks identically, writes a
`local_patch_manifest.json`, and adjusts normalized intrinsics into crop
coordinates. With 3D canonicalization enabled, it estimates a mask-centered
world transform from finite raw coordinate points, rewrites camera poses and raw
coordinate maps into that frame, and composites non-object RGB/normal/coord
pixels to white like the GObjaverse loader.

The next adapter step is to run `tools/run_gsrecon_export.py` against this local
patch manifest rather than the full scene manifest, then merge the decoded local
Gaussian patch back into the original scene.

Latest H100 finding on `bag`: the object-centered white-background local patch
is the most useful diagnostic variant so far, but it is still too blurry/sparse
for final inpainting evaluation. First-view canonicalization caused heavy
coordinate clipping in this scene and was worse.

## H100 Training Data Generation

Research-quality output likely needs fine-tuning on local real-scene patches.
The data generator should create self-supervised examples from intact calibrated
scenes:

- Sources: Inpaint360GS scenes first, then DL3DV-style calibrated scenes once a
  loader exists.
- Patch selection: sample 4 canonical input views plus held-out render views,
  centered on SAM/object proposals or geometry-aware random crop proposals.
- Synthetic masks: generate 3D/object-like masks using projected SAM components,
  depth-consistent ellipsoids, random cuboids in COLMAP space, and dilated
  latent masks.
- Targets: keep the unmasked original patch latent and full intact target latent
  from GSRecon/GSVAE, plus render targets for held-out views.
- Splits: hold out scenes, not just views, so validation measures generalization
  to unseen scene geometry.
- Artifacts per sample: `local_patch_manifest.json`, RGB/normal/coord/depth
  crops, source latent, masked latent, latent mask, camera arrays, and held-out
  RGB/alpha/depth renders.

H100 job shape for the first pass should be conservative: one H100, batch size 1
or 2, gradient accumulation, offline checkpoints, and frequent image grids from
held-out local patches.

The first executable smoke path is:

```bash
python tools/generate_native_latent_training_data.py \
  --latent-path runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_object_centered_white_gsrecon/latent.npy \
  --gaussian-npz runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_object_centered_white_gsrecon/gaussians.npz \
  --patch-manifest runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_object_centered_white/local_patch_manifest.json \
  --output-dir runs/visual_inspection/native_latent_training_smoke \
  --num-samples 32 \
  --mask-mode mixed

python tools/train_masked_latent_smoke.py \
  --dataset-manifest runs/visual_inspection/native_latent_training_smoke/dataset_manifest.json \
  --output-dir runs/visual_inspection/native_latent_training_smoke/h100_smoke \
  --steps 200 \
  --batch-size 8 \
  --device cuda
```

This smoke trainer is intentionally tiny and is not the final model. Its purpose
is to verify that local patch latents, masks, and H100 optimization mechanics are
usable before replacing it with a DiffSplat/PixArt initialized denoiser.

## Masked Latent Denoiser Objective

The learned inpainter should be initialized from the closest compatible
DiffSplat/PixArt/GSDiff latent denoiser, then fine-tuned on masked local GSVAE
latents.

Inputs:

- noisy target latent `z_t`
- masked context latent `z_context`
- latent void mask `m`
- camera or Plucker embeddings matching DiffSplat conventions
- optional text prompt such as `empty background` or the object-removal prompt

Training losses:

- Min-SNR-weighted diffusion noise or velocity prediction on masked latent cells
- direct latent reconstruction loss inside `m`
- hard unmasked preservation loss outside `m`
- GSVAE decode/render RGB loss on held-out views
- alpha/depth consistency loss to suppress floaters and geometry collapse
- optional opacity sparsity inside the void boundary

Sampling rule:

After each denoising step, clamp unmasked cells back to the original context:

```python
z = z_pred * m + z_context * (1.0 - m)
```

This makes unmasked-latent invariance a model invariant rather than a post-hoc
cleanup step.

## Visual Acceptance

The next useful milestones are:

- direct GS grid renders resemble the input local patch
- GSVAE reconstruction renders stay close to direct GS renders
- voided local patch renders remove the target consistently across views
- inpainted renders fill the void while leaving unmasked regions unchanged
- validation logs show masked latent and held-out render losses improving
