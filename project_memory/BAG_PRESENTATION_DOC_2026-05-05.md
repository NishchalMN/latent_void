# Bag Scene Presentation Document (20-minute class talk)

Use this as your slide script and speaking notes.  
Goal: be clear, honest, and strong with evidence.

## Slide 1 - Title

Title:

**Real-Scene 3D Object Removal: Baseline Success and Native Latent Inpainting Progress**

Subtitle:

- Scene: `Inpaint360GS bag`
- Baseline: Inpaint360GS
- Research branch: native DiffSplat latent void inpainting

Speaker note:

- We pursued both practical baseline quality and a research-forward native latent method under strict time constraints.

---

## Slide 2 - Problem and Objective

Content:

- Remove target object in 3D scene.
- Preserve non-target scene quality.
- Produce visually convincing renders + quantitative metrics.

Speaker note:

- The key challenge is balancing local hole fill and global scene fidelity.

---

## Slide 3 - Two-track strategy

Content:

- **Track A (baseline):** Inpaint360GS end-to-end inpaint pipeline.
- **Track B (research):** Native latent inpainting over GSRecon/GSVAE outputs.

Speaker note:

- Track A gives deliverable-quality evidence now.
- Track B gives long-term novelty but currently has reconstruction blockers.

---

## Slide 4 - Baseline outputs that exist right now

Show these images:

- `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_5000/renders/test_IMG_0186.png`
- `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_20000/renders/test_IMG_0205.png`

Also mention orbit frames:

- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_5000/00004.png`
- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_12000/00223.png`
- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_20000/00004.png`

Speaker note:

- 5k looks cleaner globally.
- 20k fills removed region better.

---

## Slide 5 - Metrics table (core evidence)

Use this exact summary:

- `iteration_5000`
  - masked: PSNR `26.29`, SSIM `0.9804`, LPIPS `0.0149`
  - non-masked: PSNR `25.67`, SSIM `0.7909`, LPIPS `0.2282`
  - full: PSNR `22.84`, SSIM `0.7730`, LPIPS `0.2442`
- `iteration_20000`
  - masked: PSNR `31.80`, SSIM `0.9849`, LPIPS `0.0108`
  - non-masked: PSNR `21.70`, SSIM `0.7040`, LPIPS `0.3062`
  - full: PSNR `21.16`, SSIM `0.6881`, LPIPS `0.3197`

JSON source:

- `output/inpaint360/bag/inpaint_evaluation_results.json`

Speaker note:

- Strong masked improvement at 20k comes with global-quality degradation.
- This is the central tradeoff result.

---

## Slide 6 - What happened to 12k (clear explanation)

Content:

- `iteration_12000` exists under:
  - `video/ours__object_inpaint_virtual/iteration_12000/`
  - `point_cloud_object_inpaint_virtual/iteration_12000/point_cloud.ply`
- But is currently missing under:
  - `inpaint/ours_object_inpaint_virtual/iteration_12000/`

Speaker note:

- Training artifacts were produced.
- Folder structure for complete inpaint image branch is incomplete for 12k comparison.
- This is why there is confusion when looking for inpaint 12k renders.

---

## Slide 7 - Engineering/debug fixes delivered

Content bullets:

- Metrics write-path permission fix.
- GSRecon promoted checkpoint load fix.
- Render classifier fallback path fix.
- Shared masked latent denoiser module + inference CLI.
- Pipeline/test wiring for denoiser command contract.

Relevant files:

- `tools/inpaint360gs_metrics_fid_masked.py`
- `tools/run_gsrecon_export.py`
- `external/Inpaint360GS/render.py`
- `tools/masked_latent_denoiser_nn.py`
- `tools/inpaint_latent_masked_denoiser.py`
- `tests/test_pipeline.py`

---

## Slide 8 - Native latent branch status (honest research update)

Show diagnostics:

- `runs/visual_inspection/inpaint360gs_bag_srun_h100_staged_render_diagnostics.png`
- `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_render_diagnostics_sheet.png`
- `runs/visual_inspection/gobjaverse_official_example_render_diagnostics_sheet.png`

Speaker note:

- Environment and tooling are functioning.
- Main blocker is real-scene reconstruction domain mismatch and source quality.
- Inpainting quality cannot surpass weak source reconstructions.

---

## Slide 9 - What to claim confidently

Say:

- Baseline delivers usable object-removal results with measurable behavior.
- There is a quantified quality tradeoff across training iterations.
- Native latent method is implemented and testable, but final-quality visuals are still gated by reconstruction quality.

Do not say:

- Native latent is already better than baseline.
- 12k is final best without complete branch and fair fixed-view comparison.

---

## Slide 10 - Final recommendation and closing

Recommendation:

1. Present `5000` and `20000` as final baseline tradeoff story.
2. Use `12000` only as a process artifact unless full inpaint branch is regenerated.
3. Frame native latent work as strong in-progress research with clear next steps.

Closing line:

**“We achieved a working real-scene baseline and established the exact technical bottleneck for native latent 3D inpainting, with concrete evidence and an executable path forward.”**

---

## Backup Slide A - File path checklist

Metrics:

- `output/inpaint360/bag/inpaint_evaluation_results.json`
- `output/inpaint360/all_scene_evaluation_results.json`

Baseline images:

- `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_5000/renders/`
- `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_20000/renders/`

Videos:

- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_5000/`
- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_12000/`
- `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_20000/`

Research diagnostics:

- `runs/visual_inspection/`
- `runs/source_recovery/`

---

## Backup Slide B - Q&A quick answers

Q: Why not just use 2D inpainted views and stop?  
A: Multi-view consistency and novel-view robustness usually require a 3D-consistent representation, otherwise seams/inconsistency appear across views.

Q: Why does longer training hurt global quality?  
A: Over-optimization toward masked region can distort non-target regions; this appears in non-masked/full metrics.

Q: Is the project successful?  
A: Baseline deliverable: yes. Native latent research: partially successful infrastructure, not yet final visual quality.

