# Status

Last updated: 2026-05-03

## 2026-05-03 Direct Zaratan Update

- Implemented the Scene-Local DiffSplat Training Plan scaffolding without
  editing the Cursor plan file:
  - `tools/build_scene_patch_dataset.py` builds multi-scene local patch datasets
    from existing geometry/mask runs and records input/held-out patch view IDs.
  - `latent_void.datasets.DL3DVDataset` and `dataset_from_config` add
    DL3DV-style calibrated scene discovery for the larger reconstruction
    adaptation source.
  - `tools/generate_patch_teacher_targets.py` creates held-out RGB/alpha/depth
    teacher target manifests from local patch manifests.
  - `tools/train_recon_adapter.py` provides a runnable reconstruction-adapter
    smoke trainer with RGB, alpha, and depth losses while keeping GSVAE frozen by
    contract.
  - `tools/evaluate_recon_gates.py` records direct-GS-grid versus GSVAE
    reconstruction diagnostic metrics and early adapter-loss gates.
  - `tools/train_masked_latent_denoiser.py` replaces the tiny smoke trainer for
    the real inpainting path with a residual masked latent denoiser that clamps
    unmasked cells.
  - `tools/merge_local_inpaint.py` merges decoded local inpainted Gaussians into
    a full scene after deletion/visibility/opacity filtering.
- `configs/native_latent_training_example.yaml` now includes example commands
  for the patch builder, teacher targets, recon adapter, gates, masked denoiser,
  and local merge stages.
- Committed the scene-local training scaffold as `4a1baf6` with a plain commit
  message and no Cursor co-author trailer.
- Started real H100 training work in the attached `gpu-a6-4` tmux session using
  the existing Inpaint360GS `bag` geometry/mask artifacts:
  - Patch dataset build:
    `runs/scene_patch_training_h100/patch_dataset/scene_patch_dataset.json`
    (`1` sample, `0` failures).
  - Teacher target generation:
    `runs/scene_patch_training_h100/teacher_targets/teacher_targets.json`
    (`1` sample, `0` failures).
  - Reconstruction-adapter smoke training:
    `runs/scene_patch_training_h100/recon_adapter/train_recon_adapter_status.json`.
    It trained `4` held-out pairs on CUDA for `1000` steps; loss improved from
    `0.2850546837` to `0.0466557033`.
  - Recon gate report:
    `runs/scene_patch_training_h100/recon_gates/recon_gate_report.json`.
    It passed the early adapter-loss gate, but still has no direct-vs-GSVAE
    diagnostic pairs for this trained adapter.
  - Masked latent denoiser mechanics run:
    `runs/scene_patch_training_h100/masked_latent_denoiser/train_masked_latent_denoiser_status.json`.
    It trained on the existing 32-sample native latent smoke dataset for `1000`
    CUDA steps; masked loss improved from `1.9817237854` to `0.0206833798`, and
    `final_context_error` was `0.0`, confirming hard unmasked-cell clamping.
  These are the first executable adaptation/training runs, not yet final-quality
  scene reconstruction or inpainting.
- Added progress logging to `tools/train_recon_adapter.py` and
  `tools/train_masked_latent_denoiser.py`, committed as `d91242d`, so long runs
  report step/loss updates instead of appearing stuck or suspiciously fast.
- Ran a longer H100 masked-latent denoiser job from the available best `bag`
  local-patch latent:
  - Generated `2048` synthetic-mask samples at
    `runs/scene_patch_training_long/native_latent_2048/dataset_manifest.json`.
  - Trained `runs/scene_patch_training_long/masked_latent_denoiser_20k/` for
    `20000` CUDA steps with batch size `8`, initialized from the earlier
    `1000`-step checkpoint.
  - Loss improved from `1.9393244982` to `0.0003723427`; context error remained
    `0.0`, so unmasked latent clamping held throughout.
- Queued multi-scene Inpaint360GS data generation in the attached H100 tmux
  session for `car`, `cube`, `cone_red`, `cone_yellow`, `garden_toys`, `truck`,
  `fruits`, `redbull`, `toys`, and `doppelherz`. The command writes per-scene
  geometry/mask runs under `runs/scene_patch_training_long/scenes/`, then builds
  `runs/scene_patch_training_long/multiscene_patch_dataset/`, generates teacher
  targets, and trains a `10000`-step multi-scene reconstruction adapter. `car`
  geometry and SAM segmentation completed successfully before moving on to
  `cube`.
- Follow-up audit after the H100 allocation ended:
  - The tmux session is back on the login node; the old `gpu-a6-4` connection
    closed after the run completed.
  - All queued multi-scene geometry and SAM stages completed successfully for
    the 10 added Inpaint360GS scenes.
  - `runs/scene_patch_training_long/multiscene_patch_dataset/scene_patch_dataset.json`
    contains `11` samples and `0` failures: the original `bag` run plus `car`,
    `cone_red`, `cone_yellow`, `cube`, `doppelherz`, `fruits`, `garden_toys`,
    `redbull`, `toys`, and `truck`.
  - `runs/scene_patch_training_long/multiscene_teacher_targets/teacher_targets.json`
    contains `11` samples, `0` failures, and `44` held-out target pairs.
  - Multi-scene recon adapter training completed:
    `runs/scene_patch_training_long/recon_adapter_multiscene_10k/train_recon_adapter_status.json`.
    It trained for `10000` CUDA steps over `44` pairs; loss improved from
    `0.3321922123` to `0.0440353006`.
  - Multi-scene recon gate report:
    `runs/scene_patch_training_long/recon_gates_multiscene/recon_gate_report.json`.
    The early adapter-loss gate passed, but direct-GS-grid vs GSVAE diagnostic
    pairs are still absent for this adapted model.
  - Held-out synthetic-mask denoiser inference evaluation:
    `runs/scene_patch_training_long/masked_latent_denoiser_20k_eval/eval_metrics.json`.
    On `256` newly generated masks, mean masked MSE was `0.0003028519`, median
    masked MSE was `0.0002251347`, max masked MSE was `0.0043224539`, and mean
    context error remained `0.0`.
  - Multi-scene recon-adapter inference evaluation:
    `runs/scene_patch_training_long/recon_adapter_multiscene_10k_eval/eval_metrics.json`.
    Across `44` target pairs, mean weighted loss was `0.0480843608`, mean RGB
    MSE was `0.0165630163`, mean alpha MSE was `0.0400669282`, and mean depth
    L1 was `0.1175397943`.
  - Slurm currently shows pending user jobs on `gpu-a100` and `gpu-h100` due to
    priority; no active attached H100 is available in tmux right now.
- Downloaded the official GObjaverse `render_data_examples.zip` archive and ran
  a true in-domain DiffSplat sanity check. Added
  `tools/prepare_gobjaverse_sample.py`, which mirrors DiffSplat's GObjaverse
  loader conventions for RGBA compositing, EXR normal/depth loading, fixed
  intrinsics, camera normalization, normal normalization, and coordinate
  encoding. The H100 GSRecon/GSVAE sanity run succeeded:
  - input sheet: `runs/visual_inspection/gobjaverse_official_example_inputs.png`
  - export: `runs/visual_inspection/gobjaverse_official_example_gsrecon/`
  - render sheet: `runs/visual_inspection/gobjaverse_official_example_render_diagnostics_sheet.png`
  The official car renders are recognizable in both direct GS grid and GSVAE
  reconstruction branches. This confirms the DiffSplat install, checkpoints, and
  renderer path are basically healthy; the unusable `bag` outputs are due to
  real-scene/local-patch domain mismatch and adapter/input quality, not a broken
  H100 environment.
- Added executable native latent adaptation scaffolding:
  - `tools/generate_native_latent_training_data.py` creates self-supervised
    masked latent samples from a local patch `latent.npy`, patch manifest, and
    synthetic/patch masks.
  - `tools/train_masked_latent_smoke.py` runs a tiny masked latent reconstruction
    model to verify sample shapes and H100 optimization mechanics.
- Generated 32 smoke training samples from the current best `bag` local patch:
  `runs/visual_inspection/native_latent_training_smoke/dataset_manifest.json`.
  Ran a 200-step H100 smoke train in tmux session `0`; masked latent MSE dropped
  from `2.1880` to `0.5790`.
  Loss plot: `runs/visual_inspection/native_latent_training_smoke/h100_smoke_loss.png`.
  This proves the adaptation loop is executable, but it is not the final model.
- Added and tested local 3D canonicalization for patch manifests. The extractor
  can now compute a mask-centered transform from finite raw coordinate points,
  rewrite camera poses and coordinate maps, and composite RGB/normal/coord
  channels to white outside the object mask to better match DiffSplat's
  GObjaverse loader.
- H100 patch reconstruction follow-up:
  - First object-centered canonicalization without white masking worsened the
    render.
  - First-view canonicalization mapped the reference camera to `[I | z=1.4]` but
    clipped the coordinate channel heavily for this scene and also rendered
    poorly.
  - Object-centered canonicalization plus white outside-mask compositing produced
    the most object-like result so far:
    `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_object_centered_white_render_diagnostics_sheet.png`.
    It is still too blurry/sparse for final quality.
- Ran local patch latent void/inpaint on the object-centered white-background
  patch as a diagnostic:
  - `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_object_centered_white_inpaint/`
  - `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_object_centered_white_inpaint_render_diagnostics_sheet.png`
  It deleted 13,771 patch Gaussians and filled 1,421 latent cells. The edited
  latent removes most of the already-weak object signal, confirming again that
  reconstruction/adaptation must improve before inpainting quality can be judged.
- Ran the new staged render diagnosis on the existing H100 `bag` artifacts from
  tmux session `0` on `gpu-a6-4`. Outputs:
  - `runs/visual_inspection/inpaint360gs_bag_srun_h100_render_diagnostics/`
  - `runs/visual_inspection/inpaint360gs_bag_srun_h100_staged_render_diagnostics.png`
  The direct `gs_grid.npy` render is already noisy/floatery, the GSVAE
  reconstruction is similarly degraded, and the edited latent mainly adds a
  localized blue fill. This confirms the blocker is upstream of the inpaint
  method: GSRecon/GSVAE reconstruction quality on the full scene is not yet
  reliable enough for judging inpainting.
- Extracted local mask-centered bag patches and ran GSRecon/GSVAE plus staged
  render diagnosis on H100:
  - `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_inputs.png`
  - `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_gsrecon/`
  - `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_render_diagnostics/`
  - `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_render_diagnostics_sheet.png`
  The local patch path is more centered and coherent than the full scene, but
  still soft/warped. This makes local patches the right debugging/training
  surface, while also showing that convention fixes alone are unlikely to be
  sufficient for final quality.
- Implemented the native latent inpainting follow-up plan without modifying the
  Cursor plan file:
  - `tools/diagnose_diffsplat_render.py` separates direct GS grid renders,
    decoded original latent renders, and decoded edited latent renders.
  - Geometry preprocessing now records a DiffSplat GObjaverse-style profile and
    supports alpha-to-white compositing plus configurable first-view camera
    canonicalization. The example and Zaratan configs now use
    `coord_mode: diffsplat`.
  - `tools/extract_local_patch_manifest.py` writes local object/void crop
    manifests with adjusted crop intrinsics for patch-scale GSRecon/GSVAE runs.
  - `docs/NATIVE_DIFFSPLAT_LATENT_INPAINTING.md` and
    `configs/native_latent_training_example.yaml` define the H100 training data
    generation contract and masked latent denoiser objective.
- Work is now being done directly in the Zaratan checkout at
  `/scratch/zt1/project/msml612pcs3/user/gnanesh/latent_void`; tmux remains useful
  for keeping the attached H100 shell alive, but it is not required by the
  project tooling.
- `configs/zaratan_inpaint360gs_bag.yaml` now uses repo-relative paths
  (`data/`, `external/`, `checkpoints/`, `runs/`) so it works from the current
  checkout instead of the older `/home/gnanesh/...` path.
- The completed H100 `bag` run was inspected more carefully. The source geometry
  RGB views and SAM 3 masks are visually sensible; the previous GSVAE render
  sheets are not useful for human progress review. Treat that as a
  reconstruction/render-adapter quality issue, not a SAM prompt issue.
- Useful visual diagnostics were written under `runs/visual_inspection/`:
  - `inpaint360gs_bag_input_masks_void_preview.png`
  - `car_source_views.png`
  - `garden_toys_source_views.png`
  - older GSVAE render sheets remain there for record, but they are not a good
    visual progress signal.
- Added `tools/inpaint_latent_context.py`, a first external native latent
  inpainting backend. It performs context-only harmonic/Jacobi fill over masked
  latent cells and checks that unmasked latent cells remain fixed. This replaces
  the pipeline fallback branch for the Zaratan config, but it is still a baseline
  rather than the final learned denoiser.
- Regenerated the existing `bag` void and inpaint outputs through the configured
  external command:
  - `void/void_manifest.json` still reports 1,701 deleted Gaussians.
  - `inpaint/context_inpaint_status.json` reports 70 masked latent cells and 128
    context-fill iterations.
- Added mask quality controls before fusion:
  `pipeline.mask_score_threshold`, `mask_min_area`, `mask_max_area_fraction`,
  `mask_erode_pixels`, and `mask_dilate_pixels`. The void manifest records these
  cleanup settings.
- Added a DiffSplat compatibility stub for optional `ImageReward` imports, which
  avoids loading the unused `datasets` metric stack during render/model imports.
- Validation passed:
  - `python -m unittest discover -s tests` -> 23 tests passed.
  - `python -m latent_void validate-config --config configs/zaratan_inpaint360gs_bag.yaml --strict-paths`
  - dry-run staged pipelines for `bag`, `car`, and `garden_toys` with 4 views.

Current blocker:

- The pipeline can create masks, voids, and latent edits, but the current
  DiffSplat/GSVAE render output for the real `bag` scene is not visually
  interpretable. Before judging inpainting quality, fix or replace the
  GSRecon/GSVAE scene reconstruction/render path so before renders resemble the
  input scene.

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
- Latest pushed repo/code commit:
  `91c992f Handle DiffSplat raw depth render outputs`

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
  by aliasing legacy `transformers.modeling_utils` helpers from
  `transformers.pytorch_utils` and locally restoring
  `find_pruneable_heads_and_indices` before importing DiffSplat.
- DiffSplat auxiliary VAE snapshots are now explicit config inputs:
  `checkpoints.sdxl_vae_path` and `checkpoints.tiny_vae_path`. The GSRecon and
  render adapters map DiffSplat's hardcoded `madebyollin/sdxl-vae-fp16-fix`
  and `madebyollin/taesdxl` repo IDs to those local snapshots.
- `scripts/download_diffsplat_aux.py` downloads those VAE snapshots on the
  login node, and `scripts/setup_zaratan_deps.sh` calls it by default.
- DiffSplat rasterizer compatibility is patched at runtime for Zaratan's
  installed `diff_gaussian_rasterization` build. The shim drops DiffSplat's
  newer `require_coord` setting when the installed rasterizer does not support
  it and expands 6-tensor rasterizer returns to DiffSplat's expected 8-tensor
  shape.
- Render diagnostics tolerate DiffSplat returning `raw_depth` instead of
  `depth`.
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

Latest real H100 MVP run:

- Active output directory:
  `runs/inpaint360gs_bag_srun_h100`
- Interactive allocation:
  `srun` job `19186674` on `gpu-a6-4`, inside remote tmux session
  `latent_void:srun`.
- Geometry completed with Marigold-derived depth/normals and coordinate maps:
  `geometry/geometry_manifest.json`, 16 views.
- GSRecon/GSVAE export completed:
  `gsrecon/gaussians.npz`, `gsrecon/gs_grid.npy`, `gsrecon/latent.npy`.
  Status reported 262,144 Gaussians, 16 projection views, and input views
  `IMG_0087` through `IMG_0090`.
- SAM 3 segmentation completed through the Transformers backend with prompt
  `bag`. It wrote 16 resized masks in `masks/`; scores were high
  (`~0.92` to `~0.97`) and masks occupied roughly `1.2%` to `1.3%` of each
  256x256 view.
- Gaussian mask fusion completed:
  `void/void_manifest.json` reports 1,701 deleted Gaussians out of 262,144.
- MVP fallback latent inpaint completed:
  `inpaint/latent_inpainted.npy`.
- GSVAE/DiffSplat render diagnostics completed:
  `renders/render_status.json` is `ok: true`, with 8 before RGB views, 8 after
  RGB views, 8 alpha maps per side, and 8 depth arrays per side. Pixel stats on
  sampled PNGs are nonblank.

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
python3 -m latent_void reconstruct --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/local_reconstruct_command_aux --dry-run
python3 -m latent_void render --config configs/zaratan_inpaint360gs_bag.yaml --set project.output_dir=runs/local_render_command_aux --dry-run
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
- DiffSplat scene exporter is now H100-tested for the Inpaint360GS `bag` scene
  through GSRecon/GSVAE export.
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
- SAM 3 Transformers backend is now H100-tested on the Inpaint360GS `bag`
  scene. The repo backend remains an untested fallback.
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
- The first retry inside interactive `srun` exposed the next missing legacy
  helper: `find_pruneable_heads_and_indices`. It is absent from Transformers
  `5.7.0`, so the adapter now restores the older implementation locally.
- The following retry reached DiffSplat imports and failed on missing `wandb`.
  The adapter now installs a no-op `wandb` module before importing DiffSplat,
  because GSRecon export does not need experiment logging.
- The first `wandb` stub needed a `ModuleSpec`; without it, importlib checks
  raised `ValueError: wandb.__spec__ is None`. The stub now sets
  `wandb.__spec__`.
- The next reconstruct failure was:
  `OSError: madebyollin/sdxl-vae-fp16-fix does not appear to have a file named
  config.json`. This exposed another hidden DiffSplat dependency: SDXL GSVAE
  constructs both the SDXL fp16 VAE and TinyAE from hardcoded Hugging Face repo
  IDs. The fix is to download local snapshots with
  `scripts/download_diffsplat_aux.py` and map those repo IDs to local paths in
  the adapters.

Remaining model-adapter blocker:

- GSVAE/native latent inpainting adapter beyond the fallback plumbing fill.
