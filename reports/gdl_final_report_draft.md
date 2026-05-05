# Native Latent 3D Gaussian Object Removal with SAM 3 and DiffSplat

Final Project Report Draft

Course: Generative Deep Learning

Team members: TODO

Date: May 5, 2026

Repository: `latent_void`, `NishchalMN/latent_void`

## Abstract

This project studies object removal in real multi-view 3D scenes. The goal is to
remove a prompted object from a calibrated scene, create a true 3D deletion
region, and fill the missing content in a latent Gaussian-splat representation
rather than independently inpainting many 2D views.

Our implementation, `latent_void`, is a standalone staged pipeline for
Inpaint360GS-style scenes. It loads RGB images and COLMAP cameras, generates the
geometry channels required by DiffSplat GSRecon using Marigold depth and normal
estimators, reconstructs a GSVAE-compatible Gaussian grid, uses SAM 3 to segment
the target object across multiple calibrated views, fuses the masks into a
Gaussian-level deletion mask, maps the deletion to a latent void, and renders
before/after diagnostics. The current system completes an end-to-end H100 MVP on
one real Inpaint360GS scene and now includes scene-local patch training
scaffolds, a masked latent denoiser, reconstruction-adapter experiments, and
visual baselines. The main remaining gap is not compute plumbing but visual
quality: real-scene DiffSplat reconstruction is still too weak for
final-quality native latent inpainting.

## 1. Problem Statement

Object removal in 3D scenes is harder than object removal in a single image. A
2D image inpainter only needs the edited image to look plausible from one view.
A 3D scene editor must also preserve multi-view consistency, avoid holes and
floaters, respect geometry, and remain stable under novel-view rendering.

The project investigates this question:

Can we remove objects from a real multi-view scene by operating directly on a
3D Gaussian latent representation, instead of inpainting each view separately
and projecting the edits back into 3D?

The intended contribution is an implementation framework and MVP experiment for
native latent 3D object removal:

- Use real multi-view scenes rather than generated object-only scenes.
- Use pretrained models wherever possible.
- Use SAM 3 concept prompts to identify the object.
- Convert multi-view object masks into a true 3D Gaussian void.
- Inpaint inside the GSVAE/DiffSplat latent space while preserving the rest of
  the scene.

## 2. Background and Related Work

3D Gaussian Splatting represents a scene using anisotropic 3D Gaussian
primitives and supports efficient real-time radiance-field rendering. Kerbl et
al. introduced the method as a practical alternative to slower volumetric NeRF
rendering for high-quality novel-view synthesis [1].

DiffSplat extends Gaussian splats into a generative setting. It introduces a
structured Gaussian-splat grid and a Gaussian-splat VAE, and it uses a
lightweight reconstruction model to produce multi-view Gaussian grids for
scalable curation [2]. Our project repurposes DiffSplat less as a text-to-3D
generator and more as a scene encoder/latent representation. This matters
because the original DiffSplat training setup is object-centric, while our
target is real indoor/outdoor scenes.

SPIn-NeRF and Inpaint360GS motivate the problem setting. SPIn-NeRF formulates
3D inpainting as object segmentation plus perceptual inpainting in a NeRF
scene, explicitly emphasizing multi-view consistency and geometric validity [5].
Inpaint360GS targets object-aware inpainting for 360-degree Gaussian-splat
scenes, using 2D segmentation distilled into 3D and virtual views for context
[4]. Our project follows the same high-level problem but tests a different
engineering direction: using DiffSplat/GSVAE latents as the inpainting domain.

SAM 3 provides promptable concept segmentation for images and videos. It can
detect, segment, and track objects specified by short noun phrases or examples
[3]. We use SAM 3 as the front end for object selection across multiple
calibrated RGB views.

Marigold is used to close a practical input gap. The available DiffSplat
GSRecon checkpoint expects RGB, normal, and coordinate maps, while scene
datasets such as Inpaint360GS and SPIn-NeRF provide RGB images and camera
metadata. Marigold depth and normal models give zero-shot geometry estimates
from RGB views [6, 7], and COLMAP cameras allow coordinate maps to be computed
by reprojection [9].

DL3DV-10K is a candidate dataset for later self-supervised training because it
contains large-scale real scene-level video data, including bounded and
unbounded scenes [8].

## 3. Approach

### 3.1 Scene Input

The MVP target is an Inpaint360GS scene. The loader reads:

- RGB views from the configured image directory.
- COLMAP camera intrinsics and extrinsics from `sparse/0`.
- Scene and run settings from YAML configuration.
- A text prompt naming the object to remove.

The code keeps dataset and checkpoint paths config-driven so the same pipeline
can later support SPIn-NeRF and DL3DV-style adapters.

### 3.2 Geometry Preprocessing

DiffSplat GSRecon does not accept RGB-only input in the checkpoint used for this
project. The preprocessing stage generates the missing channels:

- Depth maps from Marigold depth.
- Surface normals from Marigold normals.
- Coordinate maps from camera intrinsics, extrinsics, and estimated depth.
- Normalized camera frames suitable for DiffSplat-style reconstruction.

This stage writes a `geometry_manifest.json` that records every view, image
path, geometry tensor path, camera matrix, and scaled intrinsic.

### 3.3 GSRecon and GSVAE Encoding

The `tools/run_gsrecon_export.py` adapter loads the DiffSplat repository and
checkpoints on Zaratan, applies compatibility patches for the installed Python
stack, and runs GSRecon plus GSVAE encoding. It exports:

- `gaussians.npz`: positions, color, scale, rotation, opacity, projections, and
  visibility.
- `gs_grid.npy`: structured Gaussian grid.
- `latent.npy`: scaled GSVAE latent tensor.

The exported projection arrays are important because they connect each Gaussian
to the calibrated image masks used during object deletion.

### 3.4 SAM 3 Multi-View Segmentation

The `tools/run_sam3_multiview.py` wrapper runs SAM 3 over each selected view
using the object prompt. It supports the Hugging Face Transformers SAM 3 API and
the cloned official repository backend. For the MVP, the object prompt was
`bag`. Each output mask is resized to the same geometry resolution used by
GSRecon so that projected Gaussian coordinates and mask pixels agree.

### 3.5 3D Mask Fusion

The mask fusion stage projects each Gaussian into each segmented view, samples
the corresponding binary mask, and computes a masked-visible vote ratio. A
Gaussian is deleted if its score exceeds the configured threshold. The result is
a 3D deletion mask, not just a set of 2D holes.

The deletion mask is also reshaped through the Gaussian grid metadata and
resized into the latent spatial layout. This produces `latent_void_mask.npy`,
which marks where in the GSVAE latent tensor the object has been removed.

### 3.6 Latent Inpainting

The current MVP includes a fallback latent fill that replaces masked latent
cells with channel-wise means from unmasked cells. This is intentionally a
plumbing baseline. It proves that the pipeline can create a latent void, produce
an edited latent tensor, decode/render it, and preserve the data contracts
needed for the final method.

The planned final inpainting module should replace this fallback with masked
latent denoising or optimization:

- Keep unmasked latent/Gaussian regions fixed.
- Update only the void.
- Condition on surrounding scene context.
- Optionally reuse original-scene attention when compatible.
- Optimize render consistency, depth, opacity, and artifact penalties.

## 4. Implementation

The project is implemented as a standalone repository called `latent_void`.
Important files and folders are:

- `latent_void/`: core Python package for config, datasets, COLMAP parsing,
  geometry, masks, latent utilities, Gaussian utilities, and pipeline stages.
- `tools/preprocess_geometry.py`: Marigold depth/normal and coordinate-map
  preprocessing.
- `tools/run_gsrecon_export.py`: DiffSplat GSRecon/GSVAE export adapter.
- `tools/run_sam3_multiview.py`: SAM 3 multi-view prompt segmentation adapter.
- `tools/render_latent_scene.py`: before/after DiffSplat render diagnostics.
- `tools/extract_local_patch_manifest.py`: object-centered local patch
  extraction and 3D canonicalization.
- `tools/build_scene_patch_dataset.py`: multi-scene local patch dataset
  construction.
- `tools/generate_patch_teacher_targets.py`: held-out patch target generation.
- `tools/train_recon_adapter.py`: lightweight reconstruction-adapter training.
- `tools/train_masked_latent_denoiser.py`: residual masked latent denoiser with
  hard unmasked-cell clamping.
- `tools/merge_local_inpaint.py`: local inpainted Gaussian merge/compositing
  scaffold.
- `tools/run_inpaint360gs_full.py`: official Inpaint360GS baseline pipeline
  orchestration and evaluation helper.
- `configs/zaratan_inpaint360gs_bag.yaml`: H100 configuration for the first
  Inpaint360GS `bag` scene run.
- `configs/native_latent_training_example.yaml`: example commands for patch
  data, teacher targets, recon gates, denoiser training, and merge stages.
- `scripts/zaratan_srun_stage.sh`: direct interactive `srun` helper for H100
  execution.
- `project_memory/`: persistent notes on context, decisions, status, failures,
  and remaining work.
- `tests/`: local unit tests for configs, COLMAP parsing, geometry shapes,
  mask fusion, latent mask mapping, and DiffSplat compatibility shims.

The CLI exposes staged commands so each heavy step can be run independently:

```bash
python3 -m latent_void validate-config --config configs/zaratan_inpaint360gs_bag.yaml
python3 -m latent_void discover-dataset --config configs/zaratan_inpaint360gs_bag.yaml
python3 -m latent_void prepare-geometry --config configs/zaratan_inpaint360gs_bag.yaml
python3 -m latent_void reconstruct --config configs/zaratan_inpaint360gs_bag.yaml
python3 -m latent_void segment --config configs/zaratan_inpaint360gs_bag.yaml
python3 -m latent_void fuse --config configs/zaratan_inpaint360gs_bag.yaml
python3 -m latent_void inpaint --config configs/zaratan_inpaint360gs_bag.yaml
python3 -m latent_void render --config configs/zaratan_inpaint360gs_bag.yaml
```

On Zaratan, heavy stages are run interactively with `srun` on H100s rather than
`sbatch`, which makes debugging and credential-sensitive model access easier.

## 5. Engineering Issues Found and Fixed

Several simple-looking runtime failures were caused by mismatches between
research code assumptions and the Zaratan execution environment:

- GSRecon expected geometry channels, but Inpaint360GS provides RGB and poses.
  We added Marigold depth/normals and coordinate-map preprocessing.
- H100 jobs stalled or failed when trying to download Hugging Face models inside
  the compute allocation. We added login-node snapshot download scripts and
  config paths for offline compute-node execution.
- DiffSplat imports older Transformers symbols that moved or disappeared in the
  installed package version. We added compatibility aliases.
- DiffSplat imports `wandb` in inference paths. We added a no-op optional import
  shim so inference does not require live Weights & Biases setup.
- DiffSplat's auxiliary SDXL VAE paths were hardcoded as remote repository IDs.
  We patched them to local checkpoint snapshots.
- The installed Gaussian rasterizer returned an older output shape and did not
  accept one newer keyword argument. We added a renderer compatibility shim.
- Render diagnostics sometimes returned `raw_depth` rather than `depth`. The
  renderer adapter now handles that key.
- The installed RaDe-GS rasterizer extension was initially compiled only for
  H100 (`sm_90`), causing A100 render-loss jobs to fail. The setup script now
  builds for `8.0;9.0` so A100 and H100 debugging both work.
- Full-scene DiffSplat inputs produced weak renders because of object-centric
  training/domain mismatch. The project pivoted to object-scale scene-local
  patch manifests and canonicalization.

These fixes are part of the project contribution because they convert the idea
into a runnable H100 pipeline.

## 6. Experiments and Results

### 6.1 Local Tests

Local verification on May 5, 2026 after fast-forwarding to GitHub/Zaratan
commit `c3e4f9a`:

```text
Ran 27 tests in 0.377s
OK (skipped=2)
```

The two skipped tests depend on optional image libraries or local image assets
and are expected in the lightweight local environment.

### 6.2 H100 MVP Run

The first end-to-end MVP run completed on Zaratan H100 using an Inpaint360GS
`bag` scene.

| Item | Result |
| --- | --- |
| Slurm allocation | `19186674` |
| Node | `gpu-a6-4` |
| Output directory | `/home/gnanesh/scratch.msml612pcs3/latent_void/runs/inpaint360gs_bag_srun_h100` |
| Dataset scene | Inpaint360GS `bag` |
| Selected views | 16 |
| Object prompt | `bag` |
| Gaussian count | 262,144 |
| SAM 3 masks | 16 |
| SAM 3 backend | Transformers |
| SAM 3 score range | 0.921960 to 0.973890 |
| SAM 3 mean score | 0.964469 |
| Mask shape | 256 x 256 |
| Deleted Gaussians | 1,701 |
| Deleted fraction | 0.6489 percent |
| Latent void mask shape | `[4, 32, 32]` |
| Inpaint method | fallback latent fill |
| Render diagnostics | `ok: true` |
| Rendered outputs | 8 before views and 8 after views, with RGB/image, alpha, raw depth, raw normal, coordinate, and point-cloud diagnostics |

Generated artifacts include:

- `geometry/geometry_manifest.json`
- `gsrecon/gaussians.npz`
- `gsrecon/gs_grid.npy`
- `gsrecon/latent.npy`
- `masks/sam3_results.json`
- `void/gaussian_deletion_mask.npy`
- `void/latent_void_mask.npy`
- `inpaint/latent_inpainted.npy`
- `renders/render_status.json`
- before/after RGB, alpha, and depth images

### 6.3 Scene-Local Training Results

The project then moved beyond the first MVP into scene-local patch adaptation.
This was necessary because direct full-scene DiffSplat reconstruction from real
Inpaint360GS scenes did not produce readable enough local renders.

The current H100/A100 experiments show:

| Experiment | Result |
| --- | --- |
| In-domain GObjaverse sanity check | Official object example rendered recognizably through direct GS grid and GSVAE reconstruction, confirming DiffSplat checkpoints/renderer are healthy. |
| H100 patch dataset smoke | `1` `bag` patch sample, `0` failures. |
| H100 teacher targets smoke | `1` sample, `4` held-out target pairs, `0` failures. |
| H100 recon adapter smoke | `1000` CUDA steps; loss improved from `0.2850546837` to `0.0466557033`. |
| H100 masked latent denoiser smoke | `1000` CUDA steps; masked loss improved from `1.9817237854` to `0.0206833798`; final context error `0.0`. |
| Long H100 denoiser | `2048` synthetic-mask samples, `20000` steps; loss improved from `1.9393244982` to `0.0003723427`; context error stayed `0.0`. |
| Held-out denoiser evaluation | `256` new masks; mean masked MSE `0.0003028519`, median `0.0002251347`, max `0.0043224539`, mean context error `0.0`. |
| Multi-scene patch dataset | `11` Inpaint360GS scene-local samples, `0` failures. |
| Multi-scene teacher targets | `11` samples, `44` held-out target pairs, `0` failures. |
| Multi-scene recon adapter | `10000` CUDA steps over `44` pairs; loss improved from `0.3321922123` to `0.0440353006`. |
| Multi-scene recon adapter evaluation | mean weighted loss `0.0480843608`, RGB MSE `0.0165630163`, alpha MSE `0.0400669282`, depth L1 `0.1175397943`. |

These numbers prove that the data generation and training mechanics work. They
do not yet prove final-quality 3D inpainting because the recon adapter is a
lightweight scaffold and the decoded visual renders still show real-scene
domain mismatch.

### 6.4 Visual and Baseline Findings

A separate visual baseline was created on the Inpaint360GS `car` scene using a
vanilla 3DGS source reconstruction trained for `30000` iterations. It produced a
strong source render and useful before/void/target comparison sheets. However,
projection-only Gaussian pruning was not sufficient for clean object removal:
conservative pruning removed `52,216 / 1,688,912` Gaussians and aggressive
pruning removed `83,877 / 1,688,912` Gaussians, but both left dark object-shaped
smears.

The project also integrated an official Inpaint360GS baseline/evaluation path.
For the `bag` scene, the recorded evaluation file contains:

| Metric | Value |
| --- | ---: |
| SSIM masked | 0.9803727 |
| PSNR masked | 26.2941284 |
| LPIPS masked | 0.0149063 |
| SSIM full | 0.7729949 |
| PSNR full | 22.8350258 |
| LPIPS full | 0.2442329 |
| FID | null |

This baseline is useful for comparison, but it is not our native DiffSplat
latent method. The corresponding pipeline summary still marks some late
Inpaint360GS stages as failed in the orchestrator, so these metrics should be
treated as baseline/evaluation artifacts rather than final project success.

### 6.5 Initial Findings

The MVP establishes that the full data path is operational:

- Real scene data can be loaded from Inpaint360GS-style inputs.
- Missing GSRecon geometry channels can be generated from RGB and cameras.
- DiffSplat GSRecon/GSVAE can export a structured Gaussian scene on H100.
- SAM 3 can segment a text-prompted object across calibrated views.
- Multi-view masks can be fused into a Gaussian-level 3D deletion mask.
- The deletion mask can be converted into a latent-space void.
- The edited latent can be decoded/rendered for before/after diagnostics.
- Self-supervised masked latent training can preserve unmasked context exactly
  under hard clamping.
- Multi-scene scene-local patch generation and held-out target supervision are
  executable on Zaratan.

The experiment does not yet establish that the final object removal is visually
artifact-free. The current bottleneck is readable real-scene reconstruction in
the DiffSplat/GSVAE representation, not lack of segmentation, H100 execution, or
basic latent training mechanics.

## 7. Limitations

The main limitation is visual quality of real-scene DiffSplat reconstruction.
The pretrained DiffSplat stack is healthy on in-domain GObjaverse object data,
but real Inpaint360GS scene patches still produce blurry, sparse, or
geometrically unstable renders. Until the local patch reconstruction is readable,
native latent inpainting cannot be judged fairly.

Other limitations:

- Only one scene and one object prompt have been verified end to end.
- Marigold depth and normal estimates are monocular and may not be perfectly
  multi-view consistent.
- The pretrained DiffSplat checkpoints were designed primarily around object
  data, so generalization to unbounded real scenes remains uncertain.
- Teacher-render diagnostics suggest a camera/coordinate/depth rendering
  contract issue in the local patch path; even non-neural teacher Gaussians did
  not re-render one patch target cleanly.
- The current mask fusion is vote-based and lacks depth-aware cleanup,
  connected-component filtering, or shadow/effect removal.
- There are no quantitative quality metrics yet for perceptual quality,
  multi-view consistency, hole detection, or floaters.
- The render outputs have been checked as nonblank diagnostics, but still need
  systematic visual inspection and presentation-ready comparison grids.

## 8. Remaining Work

Before final submission and presentation, the most important work is:

- Fix or replace the scene-local camera/coordinate/depth rendering contract so
  local patch teacher Gaussians can re-render targets cleanly.
- Move from lightweight recon-adapter scaffolds to actual DiffSplat GSRecon and
  possibly GSVAE adaptation on scene-local patches.
- Apply the trained masked latent denoiser to real object voids only after
  reconstruction quality is readable.
- Add a self-supervised fine-tuning path if pretrained latent behavior is
  insufficient. The training data can be intact scene datasets such as DL3DV-10K
  with artificial object-like 3D masks.
- Add render-space losses for RGB consistency, depth consistency, opacity
  cleanup, and artifact suppression.
- Improve Gaussian deletion masks with confidence filtering, connected
  components, depth-aware tests, 2D/3D dilation controls, and optional shadow
  prompts.
- Run more Inpaint360GS scenes and prompts.
- Build qualitative comparison grids and videos for the class presentation.
- Add quantitative or semi-quantitative checks: mask coverage, deletion ratio,
  view consistency, hole/floater diagnostics, and render difference summaries.
- Keep documenting working and failing runs in `project_memory/`.

## 9. Conclusion

This project built a working foundation for native latent 3D object removal in
Gaussian-splat scenes. The current codebase handles real scene loading, geometry
generation, DiffSplat GSRecon/GSVAE export, SAM 3 multi-view segmentation,
Gaussian-level object deletion, latent void creation, masked latent training,
local patch reconstruction adaptation, baseline evaluation, and render
diagnostics on Zaratan GPUs.

The core research claim is not complete yet: high-quality artifact-free native
latent inpainting still depends on stronger real-scene reconstruction in the
DiffSplat/GSVAE domain. The project nevertheless reached a concrete and useful
result: the end-to-end pipeline, training data machinery, and GPU execution path
are working, and the remaining blocker has been narrowed to real-scene
scene-local reconstruction/domain adaptation.

## 10. Reproducibility State

The local report was generated after fast-forwarding the local repository to
GitHub/Zaratan `main` commit `c3e4f9a`. Local tests passed on that commit with
only this report uncommitted.

The Zaratan working tree also contained additional uncommitted experimental
changes and untracked tools at report time. The important committed artifacts
are already on `origin/main`, but the uncommitted Zaratan work should be
reconciled before final code submission. Notable dirty files included updates to
`configs/inpaint360gs_example.yaml`, `project_memory/PLAN.md`,
`project_memory/STATUS.md`, `tests/test_pipeline.py`,
`tools/run_gsrecon_export.py`, and `tools/train_masked_latent_denoiser.py`,
plus untracked denoiser, diagnostics, and run scripts.

## 11. AI Tool Usage Disclosure

OpenAI Codex/ChatGPT was used to brainstorm the project plan, write and revise
code, debug runtime errors, coordinate the local-to-Zaratan workflow, and draft
this report. The team is responsible for verifying the accuracy, completeness,
and quality of all submitted code, results, citations, and presentation
materials.

## References

[1] B. Kerbl, G. Kopanas, T. Leimkuehler, and G. Drettakis. "3D Gaussian
Splatting for Real-Time Radiance Field Rendering." SIGGRAPH 2023.
https://arxiv.org/abs/2308.04079

[2] C. Lin, P. Pan, B. Yang, Z. Li, and Y. Mu. "DiffSplat: Repurposing Image
Diffusion Models for Scalable Gaussian Splat Generation." ICLR 2025.
https://arxiv.org/abs/2501.16764

[3] Meta AI. "SAM 3: Segment Anything with Concepts." 2025.
https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/

[4] S. Wang, S. Zhang, C. Millerdurai, R. Westermann, D. Stricker, and A.
Pagani. "Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian
Splatting for 360-degree Scenes." WACV 2026.
https://arxiv.org/abs/2511.06457

[5] A. Mirzaei, T. Aumentado-Armstrong, K. G. Derpanis, J. Kelly, M. A.
Brubaker, I. Gilitschenski, and A. Levinshtein. "SPIn-NeRF: Multiview
Segmentation and Perceptual Inpainting with Neural Radiance Fields." CVPR 2023.
https://arxiv.org/abs/2211.12254

[6] B. Ke, A. Obukhov, S. Huang, N. Metzger, R. C. Daudt, and K. Schindler.
"Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation."
CVPR 2024.
https://arxiv.org/abs/2312.02145

[7] B. Ke, K. Qu, T. Wang, N. Metzger, S. Huang, B. Li, A. Obukhov, and K.
Schindler. "Marigold: Affordable Adaptation of Diffusion-Based Image Generators
for Image Analysis." 2025.
https://arxiv.org/abs/2505.09358

[8] L. Ling et al. "DL3DV-10K: A Large-Scale Scene Dataset for Deep
Learning-based 3D Vision." CVPR 2024.
https://arxiv.org/abs/2312.16256

[9] J. L. Schoenberger and J.-M. Frahm. "Structure-from-Motion Revisited." CVPR
2016.
https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Schoenberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html

[10] OpenAI. "Codex." 2026.
https://openai.com/codex/
