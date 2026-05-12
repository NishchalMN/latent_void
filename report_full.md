# Native Latent 3D Object Removal in Real Multi-View Scenes

## Full Technical Report (Expanded)

**Course:** Generative Deep Learning  
**Team members:** TODO  
**Date:** May 7, 2026  
**Repository:** `latent_void` (`NishchalMN/latent_void`)

---

## Abstract

This project builds and validates a practical pipeline for object removal in real calibrated multi-view scenes while preserving the original research objective: **native latent 3D inpainting** rather than per-view 2D image editing. The implementation uses a staged adapter architecture around Inpaint360GS and DiffSplat-compatible components, enabling fast local dry-runs and full GPU execution on Zaratan.

Two parallel tracks are maintained:

1. **Production baseline track (Inpaint360GS-compatible):** reproducible scene-level object removal/inpainting with run control, checkpoints, and metrics.
2. **Research track (native latent):** reconstruction + mask + latent contracts toward true shared-3D latent editing.

The strongest validated engineering improvement is a **post-LaMa depth grounding bridge** (affine depth alignment + planar projection in masked regions) that materially improves fusion stability and reduces floating/tilted patch artifacts.

---

## 1. Problem Statement

Single-image inpainting can look realistic but fails in multi-view consistency and geometry. Real-scene 3D object removal requires:

- view-consistent hole filling under camera motion,
- geometric plausibility (no floating sheets / depth offsets),
- minimal background regression outside masked regions,
- and clear, honest metrics by run budget.

Primary question:

> Can a real-scene object be removed by editing a shared 3D latent scene representation, rather than inpainting each 2D view independently and reprojecting?

---

## 2. Architecture and System Design

### 2.1 Adapter-first orchestration

`latent_void` does not vendor heavyweight model code; each stage is a thin adapter that:

1. resolves paths under scene run directories,
2. renders external command templates from config,
3. dispatches commands,
4. persists command/manifests/status for resumability.

This keeps the project robust to external repo churn and cluster restarts.

### 2.2 Why external repos are mirrored by overrides

`external/` is gitignored by design. To keep modifications reproducible:

- tracked mirror copies live under `upstream_overrides/inpaint360gs/`,
- install script copies tracked overrides into live external repo,
- sync script copies live changes back into tracked overrides before commit.

This mechanism prevents “works on my machine” drift in patched external scripts.

### 2.3 Pipeline stages (baseline path)

1. vanilla 3DGS training  
2. raw SAM segmentation  
2b. segment reduction  
3. 3D mask association  
4. label numbering  
5. semantic distillation  
6. target ID/config generation  
7. object removal  
8. virtual trajectory  
9. LaMa color/depth completion  
10a. PLY fusion  
10b. 3DGS inpaint finetune  
11. metrics (masked/non-masked/full)

---

## 3. Detailed Stage Semantics

### 3.1 Stage 1: vanilla 3DGS training

- Inputs: `images_<res>`, camera calibration.
- Typical budget: 30k iterations.
- Output anchor: `point_cloud/iteration_30000/point_cloud.ply`.
- Role: defines the base scene geometry and appearance used by all later stages.

### 3.2 Stage 2/2b: raw SAM + class reduction

- Raw SAM provides view-local object proposals, often over-segmented.
- Reduction merges tiny labels to keep class count manageable.
- Risk: excessive reduction can suppress fine target regions.

### 3.3 Stage 3: 3D mask association

- Projects gaussians into each view and remaps per-view mask IDs into global scene IDs.
- Output: `associated_sam/` and `associated_sam_color/`.
- Important interpretation: many objects colored is expected; this is not target-only visualization.

### 3.4 Stage 4/5: label numbering + semantic distillation

- Numbering creates normalized indexed object maps.
- Distillation trains object-aware logits in gaussian space from associated labels.
- This supports semantic object selection/removal in stage 7.

### 3.5 Stage 6/7/8: target selection, removal, virtual viewpoints

- Stage 6 selects target IDs (unseen-mask overlap or fallback heuristic).
- Stage 7 removes target gaussians and creates removal artifacts.
- Stage 8 generates virtual camera path around edited region.

### 3.6 Stage 9: LaMa color/depth completion

- Prepares inpaint inputs for virtual views.
- Produces completed color and completed depth (`depth_completed`) used in fusion.

### 3.7 Stage 10: fusion + inpaint

- 10a fuses color/depth priors into support/hole PLYs.
- 10b finetunes inpainted gaussian scene.
- Supports checkpoint videos at chosen iteration milestones (e.g., 5k/8k while targeting 12k).

### 3.8 Stage 11: evaluation

- Metrics: SSIM/PSNR/LPIPS (masked, non-masked, full), optional FID.
- Updated to support per-folder/per-budget evaluation without misleading pooled defaults.

---

## 4. Major Technical Improvements Implemented

### 4.1 Virtual depth grounding bridge (critical)

Observed failure:
- fused patch aligns in one view but drifts/tilts in others.

Fix (between stage 9 and stage 10):

1. **Affine depth alignment**  
   `inpaint360_align_completed_depth.py`  
   Fits ring-based `z_hole ~= a * z_completed + b` and updates masked pixels.

2. **Planar hole projection**  
   `inpaint360_project_completed_to_hole_plane.py`  
   Fits local hole-depth plane from ring pixels and overwrites masked depth accordingly.

Result:
- materially improved geometric consistency for fusion inputs.

### 4.2 Mid-run inpaint checkpoints + videos

Patched inpaint supports:
- CLI `--checkpoint-video-iters`
- env `INPAINT360_CHECKPOINT_VIDEO_ITERS`

This enables one 12k run to emit interpretable snapshots (e.g., 5k/8k) without restarting training.

### 4.3 Fusion pose convention hardening

Patched fusion defaults to renderer-consistent transform path (`world_view_transform`) with legacy fallback option retained for A/B diagnostics.

### 4.4 Metrics correctness fix

Previous issue:
- mixed `iteration_*` folders could be blended, obscuring true budget tradeoffs.

Current behavior:
- per-folder metrics saved explicitly,
- aggregate across runs is opt-in,
- single-subdir evaluation supported.

### 4.5 Reproducibility hardening for external patches

Added:
- override sync tool,
- override install tool,
- tracked mirror layout under `upstream_overrides/`.

This is now required workflow whenever external scripts are modified.

---

## 5. Failure Analyses and Root Causes

### 5.1 Mask instability and semantic drift

Symptoms:
- inconsistent `associated_sam_color`,
- object missing in some views,
- unstable removal/inpaint behavior.

Root contributors:
- over-fragmented raw SAM masks,
- ID remap instability across views,
- noisy distillation supervision from inconsistent labels.

### 5.2 Depth/fusion shape mismatch crash

Observed:
- `depth_completed` shape differed from hole depth/masks.
- depth-fix scripts skipped all frames.
- fusion crashed with boolean index length mismatch.

Root cause:
- resolution contract mismatch in intermediate depth outputs.

Mitigation:
- enforce shape consistency before fusion and fail-fast on skipped update counts.

### 5.3 Wrapper robustness issue (fixed)

Observed:
- stage failures could still print “done” at wrapper level.

Fix:
- summary-based fail-fast checks after phase A and phase B in SAM3 runner.

---

## 6. SAM3 Migration Strategy

Two SAM3-driven paths were added:

### 6.1 Compatibility path
`tools/run_sam3_inpaint360gs_pipeline.py`

- Replaces raw SAM generation with SAM3 prompt masks,
- keeps downstream stages 3..11 for compatibility,
- includes fail-fast and optional depth-fix bridge.

### 6.2 Direct-target path (separate files)
`tools/sam3_direct/run_sam3_direct_pipeline.py`

- bypasses legacy stages 2..6 semantics,
- uses prompt masks + projection-prune for target removal setup,
- preserves downstream contracts for LaMa/fusion/inpaint/eval reuse.

Why two paths:
- compatibility path stabilizes quickly with minimal risk,
- direct path explores compute/robustness gains by removing fragile remap/distill stages.

---

## 7. Experimental Findings (Current)

### 7.1 Validated outcomes

- End-to-end runs complete on real scenes with resumable orchestration.
- Depth-fix bridge improves fusion alignment versus pre-fix behavior.
- Mid-run checkpoints enable practical quality-vs-iteration inspection.
- Metrics reporting now supports honest per-budget interpretation.

### 7.2 Budget tradeoff

Empirically:
- longer finetune tends to improve masked hole metrics,
- but may regress non-masked/full regions via over-editing.

Practical operating point:
- 12k with intermediate snapshots, or 5k for conservative background preservation depending on scene.

---

## 8. Reproducible Runbook

### 8.1 Install tracked external overrides

```bash
cd /scratch/zt1/project/msml612pcs3/user/gnanesh/latent_void
python3 tools/sync_inpaint360gs_upstream_overrides.py
bash scripts/install_inpaint360gs_overrides.sh
```

### 8.2 Baseline full run (single scene)

```bash
bash scripts/run_inpaint360gs_scene_full.sh car
```

### 8.3 Full run with depth-fix bridge

```bash
bash scripts/run_inpaint360gs_scene_virtual_depth_full.sh car
```

### 8.4 SAM3 compatibility runner

```bash
SAM3_DEVICE=cuda python tools/run_sam3_inpaint360gs_pipeline.py \
  --scene car \
  --prompt "car" \
  --sam3-root external/sam3 \
  --sam3-checkpoint checkpoints/sam3 \
  --resolution 2 \
  --finetune-iters 5000 \
  --checkpoint-video-iters "" \
  --run-depth-fix
```

### 8.5 SAM3 direct-target runner

```bash
python tools/sam3_direct/run_sam3_direct_pipeline.py \
  --scene car \
  --prompt "car" \
  --sam3-root external/sam3 \
  --sam3-checkpoint checkpoints/sam3 \
  --resolution 2 \
  --finetune-iters 5000 \
  --checkpoint-video-iters "" \
  --run-depth-fix
```

### 8.6 Per-budget evaluation without blended averages

```bash
export PYTHONPATH=external/Inpaint360GS:external/Inpaint360GS/gaussian_splatting:$PYTHONPATH
python tools/inpaint360gs_metrics_fid_masked.py \
  -m output/inpaint360/car \
  --eval-only-subdir iteration_5000 \
  --skip-fid
```

---

## 9. Quality Assurance Checklist

### 9.1 Visual checkpoints in order

1. base 3DGS render quality (`3dgs_output`),
2. raw/associated masks consistency,
3. removal virtual renders (`virtual/ours_object_removal/iteration_2000/renders`),
4. virtual mask quality (`inpaint_2d_unseen_mask_virtual`),
5. depth vs depth_completed sanity,
6. fused PLY alignment (`fused_mask_col_dep_ply` vs `fused_hole_col_dep_ply`),
7. checkpoint videos (5k/8k/12k),
8. final inpaint renders + metrics per iteration folder.

### 9.2 Runtime guards

- fail if stage summary reports non-ok stage,
- fail if depth-fix updates 0 frames unexpectedly,
- fail if expected folder contracts are missing before fusion.

---

## 10. Limitations

1. Real-scene reconstruction quality still bounds final native-latent quality.
2. Legacy Inpaint360GS stage coupling is rigid; contract mismatches propagate quickly.
3. SAM prompt quality can vary across viewpoint/occlusion regimes.
4. Offline/FID and cluster process limits require operational workarounds.
5. Some wrappers still require stronger shape and contract validation by default.

---

## 11. Next Work Plan

### 11.1 Immediate engineering

- Add strict shape harmonization before fusion (depth/mask/color contracts).
- Add fail-fast checks in all runners (including direct path) for stage summaries.
- Add stage artifacts dashboard script (one command to inspect all checkpoints).

### 11.2 Model-level

- Improve direct-target SAM3 stability with temporal/multi-view consistency heuristics.
- Add stronger inpaint regularization for background preservation.
- Explore replacing semantic-distillation dependence entirely in direct mode.

### 11.3 Reporting

- Standardize per-scene/per-budget report templates:
  - masked/non-masked/full metrics,
  - key frames,
  - orbit videos,
  - failure notes and fix status.

---

## 12. Interview / Impact Summary

This project demonstrates end-to-end ownership of a complex 3D vision + GenAI system:

- architecture design under external dependency constraints,
- production-style reproducibility for research code,
- root-cause debugging across segmentation/geometry/fusion/inpaint,
- metric integrity fixes,
- and practical quality gains through targeted pipeline interventions.

Key impact narrative:
- moved from fragile, opaque multi-stage behavior to traceable, reproducible, and diagnosable runs with controlled quality/compute tradeoffs.

---

## References

[1] B. Kerbl, G. Kopanas, T. Leimkuehler, G. Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH 2023.  
[2] C. Lin, P. Pan, B. Yang, Z. Li, Y. Mu. DiffSplat: Repurposing Image Diffusion Models for Scalable Gaussian Splat Generation. ICLR 2025.  
[3] Meta AI. SAM 3: Segment Anything with Concepts. 2025.  
[4] S. Wang et al. Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360-degree Scenes. WACV 2026.  
[5] A. Mirzaei et al. SPIn-NeRF: Multiview Segmentation and Perceptual Inpainting with Neural Radiance Fields. CVPR 2023.
