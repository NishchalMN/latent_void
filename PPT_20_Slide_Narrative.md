# 20-Slide PPT Narrative: DiffSplat + Inpaint360GS Unified Story

**Purpose:** This document is a slide-by-slide script to generate a ~20-page presentation that combines:
- DiffSplat ablation findings (`DiffSplat_Ablation_Summary.md`)
- Real-scene object-removal engineering track (`report_full.md`)

Use this as direct source content for Gemini/Claude PPT generation.

---

## Slide 1 — Title Slide
**Title:**  
From Native 3D Generation to Real-Scene 3D Object Removal  

**Subtitle:**  
DiffSplat Ablation Insights + Inpaint360GS Engineering Extension

**On-slide bullets:**
- Generative Deep Learning — Group 6
- Team members: Kaustubh Shah, Sree Gnanesh Dommeti, Nishchal Marur Nanjunda Swamy, Haochen Yang
- Repository: `latent_void`

**Visual placeholder:**
- `[PLACEHOLDER: one hero frame from bag final inpaint orbit video]`

**Speaker notes:**
- This talk has two connected parts: first, what we learned from DiffSplat behavior; second, how we built a practical real-scene object-removal pipeline and stabilized it.

---

## Slide 2 — Problem and Motivation
**Title:**  
Why This Project?

**On-slide bullets:**
- 2D inpainting can look good per image but breaks under viewpoint changes.
- Real multi-view editing needs geometric consistency, not just local texture plausibility.
- Goal: remove objects from calibrated real scenes while preserving scene integrity.

**Visual placeholder:**
- `[PLACEHOLDER: side-by-side: single-view plausible edit vs multi-view inconsistency example]`

**Speaker notes:**
- Emphasize that this is a systems problem and a model problem.
- Our target is not one pretty frame; it is a consistent 3D editable scene.

---

## Slide 3 — Background: NeRF to 3DGS
**Title:**  
Representation Shift: NeRF -> 3D Gaussian Splatting

**On-slide bullets:**
- NeRF: high quality but expensive volume rendering and optimization.
- 3DGS: explicit anisotropic gaussians; fast rasterization and real-time view synthesis.
- For editing pipelines, 3DGS improves iteration speed and practical debugging.

**Visual placeholder:**
- `[PLACEHOLDER: simple architecture diagram NeRF (implicit field) vs 3DGS (explicit gaussians)]`

**Speaker notes:**
- Keep this concise: 3DGS is the operational backbone that makes iterative object editing feasible.

---

## Slide 4 — Why 3DGS Is Practical for This Project
**Title:**  
Why We Chose 3DGS as the Editing Substrate

**On-slide bullets:**
- Explicit primitives let us inspect/modify scene geometry directly.
- Compatible with object-aware masking, pruning, and fusion pipelines.
- Scales better for repeated ablations and engineering loops.

**Visual placeholder:**
- `[PLACEHOLDER: screenshot of gaussian point cloud visualization from project outputs]`

**Speaker notes:**
- Mention that many pipeline bugs were diagnosable only because representation is explicit.

---

## Slide 5 — DiffSplat: Core Idea
**Title:**  
DiffSplat Overview

**On-slide bullets:**
- Fine-tunes pretrained 2D diffusion priors to generate 3D Gaussian splat grids.
- Promise: stronger 2D priors should transfer to stronger 3D generation.
- Our first track validates this claim via ablation-style experiments.

**Visual placeholder:**
- `[PLACEHOLDER: DiffSplat architecture figure from paper (or recreated)]`

**Speaker notes:**
- Frame DiffSplat as the proposal-aligned research backbone.

---

## Slide 6 — Ablation Study Design (What We Tested)
**Title:**  
DiffSplat Ablations: Study Matrix

**On-slide bullets:**
- Main prior-strength comparison: SD1.5 vs PixArt-Sigma vs SD3.5M.
- Guidance scale sweep.
- Seed sensitivity (determinism/diversity).
- Prompt category performance.
- Native 3D vs naive 2D multi-view comparison.
- Failure mode catalog by object type.

**Visual placeholder:**
- `[PLACEHOLDER: compact table "Study / Question / Key Finding"]`

**Speaker notes:**
- State that unavailable official ablation checkpoints were replaced by complementary studies.

---

## Slide 7 — Key Ablation Results (High-Level)
**Title:**  
What the Ablations Showed

**On-slide bullets:**
- No monotonic “bigger base model = better 3D” trend.
- PixArt-Sigma achieved the highest mean CLIP among tested variants.
- Guidance robust in mid-range; very low guidance harms quality.
- Outputs were highly deterministic across random seeds.

**Visual placeholder:**
- `[PLACEHOLDER: bar chart of mean CLIP by model + line chart guidance scale]`

**Speaker notes:**
- Emphasize practical takeaway: prior size alone did not dominate performance.

---

## Slide 8 — Prompt Categories and Failure Modes
**Title:**  
Where DiffSplat Succeeds and Fails

**On-slide bullets:**
- Strongest on simple objects and coarse geometry.
- Weak on fine details, thin structures, human anatomy, difficult materials.
- Text/logo prompts appear “semantically good” by CLIP even when text itself is wrong.

**Visual placeholder:**
- `[PLACEHOLDER: grid of best/worst prompt outputs from ablation study]`

**Speaker notes:**
- Clarify CLIP limitations: semantic alignment is not full geometric fidelity.

---

## Slide 9 — Why DiffSplat Alone Was Not Enough for Large Real Scenes
**Title:**  
Gap to Real Multi-View Scene Editing

**On-slide bullets:**
- DiffSplat strengths did not directly translate to robust real-scene object removal.
- Scene scale, occlusion complexity, and mask consistency became dominant bottlenecks.
- Needed a practical scene-level editing stack with stronger engineering controls.

**Visual placeholder:**
- `[PLACEHOLDER: "expected vs observed" table for large-scene behavior]`

**Speaker notes:**
- Transition slide into second track.

---

## Slide 10 — Extension Track: Inpaint360GS-Based Pipeline
**Title:**  
Our Engineering Extension: Real-Scene 3D Object Removal

**On-slide bullets:**
- Built a reproducible stage-wise pipeline around Inpaint360GS.
- Added robust orchestration, logs, status files, and resumability.
- Optimized for real scene operations, not only benchmark object generation.

**Visual placeholder:**
- `[PLACEHOLDER: full pipeline block diagram with stage IDs 1..11]`

**Speaker notes:**
- This is where project impact shifted from pure model eval to end-to-end system reliability.

---

## Slide 11 — 11-Stage Baseline Flow
**Title:**  
Inpaint360GS Pipeline (Baseline)

**On-slide bullets (compact list):**
- 1) 3DGS train  
- 2) SAM segmentation (+2b reduction)  
- 3) 3D mask association  
- 4) label numbering  
- 5) semantic distillation  
- 6) target-ID selection  
- 7) object removal  
- 8) virtual trajectory  
- 9) LaMa color/depth completion  
- 10a) fusion  
- 10b) inpaint finetune  
- 11) evaluation

**Visual placeholder:**
- `[PLACEHOLDER: timeline-style stage strip]`

**Speaker notes:**
- Mention that stages are file-contract-based; each stage writes artifacts for downstream.

---

## Slide 12 — Core Bottleneck We Found
**Title:**  
Mask Consistency and Semantic Remap Instability

**On-slide bullets:**
- Segment-all + remap pipeline produced unstable associated masks in challenging scenes.
- Inconsistent object IDs propagated into removal and inpaint quality degradation.
- This motivated replacing heavy front-half stages with SAM3 target-first logic.

**Visual placeholder:**
- `[PLACEHOLDER: associated_sam_color failure examples across views]`

**Speaker notes:**
- This is the operational failure that drove the SAM3 migration work.

---

## Slide 13 — Critical Fix: Depth Grounding Bridge
**Title:**  
Post-LaMa Depth Fix That Unblocked Fusion

**On-slide bullets:**
- Applied between Stage 9 and Stage 10.
- Step A: affine depth alignment (`z_hole ~= a*z_completed + b`) on ring pixels.
- Step B: planar projection in masked region from local hole-depth ring.
- Outcome: reduced floating/tilted fused patches and improved geometric coherence.

**Visual placeholder:**
- `[PLACEHOLDER: before/after depth maps + fused PLY overlay]`

**Speaker notes:**
- Stress that this fix had higher impact than changing finetune iterations alone.

---

## Slide 14 — SAM3 Strategy: Reduce Heavy Legacy Stages
**Title:**  
SAM3 Integration to Reduce Compute and Failure Points

**On-slide bullets:**
- Legacy path segments everything then remaps IDs globally.
- SAM3 prompt path segments known target object directly.
- Objective: reduce brittle remap/distill stages and simplify target selection.

**Visual placeholder:**
- `[PLACEHOLDER: old flow vs new flow comparison diagram]`

**Speaker notes:**
- Explain two modes:
  - Compatibility SAM3 mode (replace raw mask input, keep downstream).
  - Direct SAM3 mode (bypass stages 2-6 semantics).

---

## Slide 15 — New SAM3 Pipelines Implemented
**Title:**  
What We Implemented (Separate Safe Paths)

**On-slide bullets:**
- `tools/run_sam3_inpaint360gs_pipeline.py` (compatibility path)
  - SAM3 masks -> existing stage 3..11 with fail-fast guards.
- `tools/sam3_direct/run_sam3_direct_pipeline.py` (direct-target path)
  - prompt masks + projection pruning + virtual generation + existing stage 10/11.
- Kept in separate files to protect baseline reproducibility.

**Visual placeholder:**
- `[PLACEHOLDER: code/file tree snapshot showing both runner paths]`

**Speaker notes:**
- Mention engineering guardrails added: summary checks, stale-mask refresh, environment path handling.

---

## Slide 16 — Debugging Learnings (Real Failures and Fixes)
**Title:**  
Failure -> Root Cause -> Fix

**On-slide bullets:**
- **Failure:** stage continued after earlier errors  
  **Fix:** fail-fast by reading pipeline summary stage statuses.
- **Failure:** stale masks reused  
  **Fix:** clear and refresh raw mask outputs, map SAM3 masks to image stems.
- **Failure:** depth/fusion array mismatch  
  **Fix:** shape-contract checks + depth harmonization before fusion.
- **Failure:** module import errors in subprocesses  
  **Fix:** explicit `PYTHONPATH` propagation for Inpaint360GS contexts.

**Visual placeholder:**
- `[PLACEHOLDER: small table with 4 rows Failure / Root Cause / Fix / Status]`

**Speaker notes:**
- This slide demonstrates software engineering depth and operational maturity.

---

## Slide 17 — Quantitative Evaluation (Keep Budget Discussion Light)
**Title:**  
Evaluation Summary (Scene-Level)

**On-slide bullets:**
- Metrics reported per run folder: masked / non-masked / full regions.
- Reporting fixed to avoid misleading pooled averages by default.
- Main interpretation: improved hole fill can trade off with background stability.
- Emphasis shifted to robust geometric plausibility + honest per-run reporting.

**Visual placeholder:**
- `[PLACEHOLDER: concise metrics table, no heavy 5k/8k/20k focus]`

**Speaker notes:**
- Keep this high-confidence; avoid overclaiming unstable budget comparisons.

---

## Slide 18 — Qualitative Results: Bag Scene Storyboard
**Title:**  
Bag Scene End-to-End Visual Narrative

**On-slide bullets:**
- Original scene
- Holed/removal scene
- Inpainted/final scene
- Orbit/video consistency check

**Visual placeholders (required):**
- `[PLACEHOLDER GIF/VIDEO 1: bag original orbit]`
- `[PLACEHOLDER GIF/VIDEO 2: bag holed/removal orbit]`
- `[PLACEHOLDER GIF/VIDEO 3: bag inpainted orbit]`

**Speaker notes:**
- This is the core “demo” slide. Let media dominate; keep text minimal.

---

## Slide 19 — Contribution Summary
**Title:**  
What We Contributed Beyond Baseline Reproduction

**On-slide bullets:**
- Comprehensive DiffSplat ablation program under checkpoint constraints.
- End-to-end real-scene object-removal pipeline hardening.
- Depth grounding bridge that materially improved fusion robustness.
- SAM3 migration paths to reduce brittle front-half processing.
- Reproducibility protocol for external patched code (`upstream_overrides` workflow).

**Visual placeholder:**
- `[PLACEHOLDER: contribution matrix (Research, Engineering, Evaluation, Reproducibility)]`

**Speaker notes:**
- Position as combined research + systems delivery.

---

## Slide 20 — Conclusion and Future Directions
**Title:**  
Conclusion and Next Steps

**On-slide bullets:**
- DiffSplat gives important native-3D priors but has practical limits on complex real scenes.
- Robust real-scene editing required substantial pipeline engineering, not just model swaps.
- SAM3 target-first direction is promising to reduce compute and improve consistency.
- Next: stricter shape contracts, stronger direct-target path, and richer real-scene benchmarks.

**Visual placeholder:**
- `[PLACEHOLDER: roadmap graphic: "Now -> Next -> Target"]`

**Speaker notes:**
- End with balanced message: strong progress, clear remaining gaps, concrete roadmap.

---

## Appendix: PPT Generation Prompt Template (for Gemini/Claude)

Use this if you want automatic deck generation:

> Generate a 20-slide technical presentation using the slide script in this file.  
> Requirements:  
> - Keep slide titles exactly as provided.  
> - Keep each slide to 3-6 concise bullets.  
> - Add speaker notes from the provided speaker notes section.  
> - Preserve all `[PLACEHOLDER ...]` blocks exactly so we can manually insert media later.  
> - Style: clean academic + engineering narrative, minimal clutter, dark text on light background, consistent iconography.  
> - Include one summary visual/table/chart on slides 6, 7, 16, and 19.  
> - Do not fabricate new metrics beyond what's listed; if uncertain, keep placeholders.

