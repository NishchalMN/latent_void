# Bag Scene Full Technical Report (2026-05-05)

Project: `latent_void`  
Primary baseline: Inpaint360GS  
Research branch: native DiffSplat-style latent void inpainting

## 1) Why this document exists

This is a complete technical report of what was run, what artifacts exist, what is broken, what got fixed, and what is currently trustworthy for evaluation and final reporting.

---

## 2) Current reality (plain truth)

- Baseline Inpaint360GS on `bag` works and produced valid outputs and metrics.
- `iteration_5000` and `iteration_20000` are complete under inpaint render folders.
- `iteration_12000` exists for video frames and point cloud, but not under the same `inpaint/.../iteration_12000` folder branch currently used by 5k/20k.
- Native latent path is wired and executable but still blocked by source reconstruction quality (Gate A/B problem), so final visual quality claims are not yet strong there.

---

## 3) Main folders and what they contain

## A) Inpaint360GS output root

- `output/inpaint360/bag/`

Important subfolders:

- `inpaint/ours_object_inpaint_virtual/iteration_5000/`
- `inpaint/ours_object_inpaint_virtual/iteration_20000/`
- `video/ours__object_inpaint_virtual/iteration_5000/`
- `video/ours__object_inpaint_virtual/iteration_12000/`
- `video/ours__object_inpaint_virtual/iteration_20000/`
- `point_cloud_object_inpaint_virtual/iteration_5000/point_cloud.ply`
- `point_cloud_object_inpaint_virtual/iteration_12000/point_cloud.ply`
- `point_cloud_object_inpaint_virtual/iteration_20000/point_cloud.ply`
- `inpaint_evaluation_results.json`

## B) Native latent / diagnostics roots

- `runs/source_recovery/`
- `runs/scene_patch_training_long/`
- `runs/visual_inspection/`

---

## 4) Concrete image paths for report figures

Use these directly in the written report:

## Baseline test/inpaint evidence

- `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_5000/renders/test_IMG_0186.png`
- `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_20000/renders/test_IMG_0205.png`

## Orbit/video evidence (contains 12000)

- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_5000/00004.png`
- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_12000/00004.png`
- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_12000/00223.png`
- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_12000/00231.png`
- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_20000/00004.png`

## Native branch diagnostics

- `runs/visual_inspection/inpaint360gs_bag_srun_h100_staged_render_diagnostics.png`
- `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_render_diagnostics_sheet.png`
- `runs/visual_inspection/gobjaverse_official_example_render_diagnostics_sheet.png`

---

## 5) Verified quantitative metrics

Tracked values from evaluation outputs/status logs:

## Iteration 20000

- `SSIM_masked = 0.9849212169647217`
- `PSNR_masked = 31.803434371948242`
- `LPIPS_masked = 0.010785897262394428`
- `SSIM_nonmasked = 0.7039921879768372`
- `PSNR_nonmasked = 21.698040008544922`
- `LPIPS_nonmasked = 0.306243896484375`
- `SSIM_full = 0.6881226897239685`
- `PSNR_full = 21.158658981323242`
- `LPIPS_full = 0.3196602463722229`

## Iteration 5000

- `SSIM_masked = 0.9803727269172668`
- `PSNR_masked = 26.29412841796875`
- `LPIPS_masked = 0.014906308613717556`
- `SSIM_nonmasked = 0.7909024357795715`
- `PSNR_nonmasked = 25.670026779174805`
- `LPIPS_nonmasked = 0.2281893640756607`
- `SSIM_full = 0.772994875907898`
- `PSNR_full = 22.835025787353516`
- `LPIPS_full = 0.2442328780889511`

## Metric interpretation

- `20000` is stronger where object was removed (masked region).
- `5000` is stronger for global/background preservation.
- This is a real optimization tradeoff, not a logging artifact.

---

## 6) What exactly happened with iteration_12000

Observed:

- Exists: `video/ours__object_inpaint_virtual/iteration_12000/`
- Exists: `point_cloud_object_inpaint_virtual/iteration_12000/point_cloud.ply`
- Missing currently: `inpaint/ours_object_inpaint_virtual/iteration_12000/`

Meaning:

- The 12k training outputs are not fully absent.
- The artifact layout is incomplete for direct apples-to-apples comparison against 5k/20k inpaint image folders.
- This explains why you cannot find `.../inpaint/.../iteration_12000/` while still seeing 12k video frames.

---

## 7) Important debugging conclusion from render behavior

- In `external/Inpaint360GS/render.py`, classifier fallback loads from:
  - `output/inpaint360/bag/point_cloud/iteration_2000/classifier.pth`
- This fallback is expected if classifier is not saved per inpaint iteration.
- Video path uses orbit camera generation, which can expose worse artifacts than fixed test-camera renders.

Practical consequence:

- A bad-looking orbit does not automatically mean all fixed-view inpaint outputs are bad.

---

## 8) Changes implemented during this cycle (key technical fixes)

1. `tools/inpaint360gs_metrics_fid_masked.py`
   - Fixed output write path to avoid permission error.
2. `tools/run_gsrecon_export.py`
   - Added robust promoted-checkpoint loading path (`model_state_dict` and `state_dict` handling).
3. `external/Inpaint360GS/render.py`
   - Added classifier fallback lookup so rendering does not crash on missing nested path.
4. `tools/masked_latent_denoiser_nn.py`
   - Added shared denoiser module for training and inference.
5. `tools/inpaint_latent_masked_denoiser.py`
   - Added standalone learned-latent inpaint infer CLI.
6. `tools/train_masked_latent_denoiser.py`
   - Refactored to use shared model module.
7. `tests/test_pipeline.py`
   - Added command-format regression test for masked denoiser pipeline path.

---

## 9) What is report-safe vs not report-safe

## Safe and defensible statements

- Baseline inpaint pipeline is operational on `bag`.
- There is clear and measured masked-vs-global quality tradeoff with longer finetuning.
- Native latent branch has executable infrastructure and training diagnostics but still fails high-quality scene reconstruction gates.

## Not safe yet

- Claiming native latent branch final visual superiority.
- Claiming 12k is best final checkpoint without complete consistent output branch and matched comparison set.

---

## 10) Suggested final report structure (technical write-up)

1. Problem definition and constraints.
2. Baseline pipeline and metrics (`5000` vs `20000`).
3. Artifact inconsistency diagnosis (`12000` video exists but missing inpaint folder).
4. Native latent research branch: architecture, diagnostics, blocker.
5. Lessons learned and next steps.

---

## 11) Immediate recovery plan (if continuing)

1. Regenerate complete `iteration_12000` inpaint render branch.
2. Build fixed-camera triplet comparison: `{5000, 12000, 20000}`.
3. Freeze one final baseline iteration for presentation visuals.
4. Keep native branch as forward-looking research contribution with transparent blockers.

---

## 12) Appendix: key JSON/log paths

- `output/inpaint360/bag/inpaint_evaluation_results.json`
- `output/inpaint360/all_scene_evaluation_results.json`
- `output/inpaint360/bag/pipeline_status.json`
- `output/inpaint360/pipeline_summary.json`
- `output/inpaint360/bag/logs/render_only_12000_video.log`

