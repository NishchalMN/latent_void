# Technical Report: Multi-View 3D Object Removal and Inpainting  
## Inpaint360GS Pipeline Reproduction and Multi-Scene Evaluation Plan

**Course:** Generative Deep Learning (Group 6)  
**Repository:** `latent_void` (orchestration, metrics, adapters) + `external/Inpaint360GS` (baseline implementation)  
**Report status:** Draft — **Scene `bag`** is documented with **verified** paths and metrics. **Scenes `car` and `cube`** follow the **same eleven-stage protocol**; quantitative cells and some paths are **explicit placeholders** until your scheduled **5k / 12k / 20k** finetune runs finish. **Two additional scenes** appear only as **extension placeholders** for future work.

**Integrity note:** All numbers in §7.1 for `bag` are taken from recorded evaluation logs / `project_memory` audit. Cells marked **TBD** must be filled from `output/inpaint360/<scene>/inpaint_evaluation_results.json` after each run — do not invent values.

---

## Abstract

We study **3D-consistent object removal and inpainting** on real multi-view captures using a pipeline aligned with **Inpaint360GS**: train or load a 3D Gaussian Splatting (3DGS) scene representation, obtain object masks (e.g. SAM), associate masks in 3D, remove target Gaussians, inpaint missing evidence with 2D inpainting, and **finetune** a specialized 3DGS stage to fill geometric and appearance holes. We integrate this stack with the **`latent_void`** project for configuration, dataset discovery, and evaluation tooling.

**Primary completed evidence** is on the **`bag`** scene: full artifact trees, masked and unmasked image metrics, and a documented tradeoff between **shorter** finetuning (better global / background fidelity) and **longer** finetuning (stronger inpainted-region metrics). **Scenes `car` and `cube`** are executed with the **same eleven-stage contract**; this report reserves **tables and figure slots** for **three finetune budgets** (5000, 12000, 20000 iterations) per scene. **Two further scenes** are listed as **placeholders** for extended evaluation only.

A **parallel research thread** (native latent editing via DiffSplat-style GSRecon / GSVAE) is **not** claimed as production-quality; reconstruction quality on real scans remains the binding limitation, consistent with diagnostics in `runs/visual_inspection/`.

**Original proposal alignment:** An initial direction toward **full DiffSplat paper reproduction** (training-scale ablations, T3Bench-scale metrics) was **deprioritized**: upstream DiffSplat **training** assumes large-scale curated data layouts and compute; **inference** checkpoints exist, but **course-scale reproduction** of the full training and benchmark protocol is **out of scope** relative to time and cluster constraints. This report therefore centers on **reproducible Inpaint360GS-style** results with **on-disk artifacts**.

---

## 1. Introduction

### 1.1 Motivation

Single-image inpainting produces plausible RGB fill but often **breaks multi-view consistency**. Object removal in a **calibrated multi-view** capture requires either optimizing a 3D representation to agree with many cameras or accepting visible seams. **Inpaint360GS-class** methods combine **2D inpainting** (fast, detailed texture) with **3D Gaussian splatting finetuning** to obtain a **single 3D model** that can be rendered from novel viewpoints.

### 1.2 Project goals

1. **Reproduce** an **eleven-stage** Inpaint360GS-style pipeline end-to-end on **multiple** Inpaint360 dataset scenes.  
2. **Quantify** quality using **masked**, **non-masked**, and **full-frame** metrics where implemented.  
3. **Document failures** (partial artifact trees, classifier checkpoints, orbit vs test cameras) honestly.  
4. **Reserve** headroom for **five-scene** reporting: **three** scenes in the main body (`bag`, `car`, `cube`) and **two** named placeholders for extension.

### 1.3 Contributions (what we actually deliver)

- **Operational pipeline** with logs and outputs under `output/inpaint360/`.  
- **Fixed evaluation tooling** (e.g. metrics JSON written under each scene directory).  
- **Empirical tradeoff analysis** for finetune length on **`bag`**.  
- **Explicit runbook** for **`car`** and **`cube`** at **5k / 12k / 20k** finetune checkpoints.  
- **Research appendix** on native latent path limitations (optional for course; strengthens “why 3DGS baseline”).

---

## 2. Related Work

- **3D Gaussian Splatting (3DGS):** explicit Gaussian primitives + differentiable rasterization for novel-view synthesis.  
- **Inpaint360GS / Gaussian grouping line:** semantic object association, removal, and inpainting-focused 3DGS optimization.  
- **DiffSplat (ICLR 2025):** feed-forward / diffusion-based **generative** Gaussian splat grids; motivating for **native latent** editing but **not** the empirical core of this report.

References should cite Kerbl et al. (3DGS), the Inpaint360GS paper, and DiffSplat as appropriate in the PDF version.

---

## 3. System Overview

### 3.1 Repository layout

| Component | Role |
|-----------|------|
| `external/Inpaint360GS/` | Upstream training, removal, 2D inpaint, 3D finetune, render scripts |
| `latent_void/` | Config validation, dataset discovery, optional native-latent adapters, metric scripts |
| `data/inpaint360/<scene>/` | Per-scene COLMAP + images + inpaint assets |
| `output/inpaint360/<scene>/` | Per-scene outputs: point clouds, renders, videos, metrics |

### 3.2 Eleven-stage pipeline (conceptual)

The following **logical stages** match the **group’s “11-stage” narrative**. Exact script names may vary by invocation (shell pipeline vs manual); the **artifact contract** is what matters for grading.

| Stage | Description | Typical outputs (per scene) |
|------|-------------|-----------------------------|
| 1 | Dataset ingest / COLMAP / camera calibration | `sparse/`, images under `data/inpaint360/<scene>/` |
| 2 | Base 3DGS training (or load checkpoint) | `3dgs_output/` or `point_cloud/iteration_*` |
| 3 | Multi-view segmentation (e.g. SAM) | mask directories under dataset / output |
| 4 | 3D mask association / object IDs | labels usable by removal stage |
| 5 | Object removal (Gaussian deletion, hull, etc.) | `point_cloud_object_removal/` or equivalent |
| 6 | 2D inpainting (e.g. LaMa) on inpaint / virtual views | fused color+depth inputs for 3D stage |
| 7 | Fusion / point cloud or buffer preparation | `virtual/.../fused_*` style paths |
| 8 | **3D inpaint finetuning** (`edit_object_inpaint.py` lineage) | `point_cloud_object_inpaint_virtual/iteration_*` |
| 9 | Rendering (train / test / inpaint / video) | `inpaint/.../renders/`, `video/.../` |
| 10 | Evaluation (PSNR, SSIM, LPIPS, masked variants) | `inpaint_evaluation_results.json` |
| 11 | Manifest / summary for reproducibility | `pipeline_status.json`, `pipeline_summary.json` |

### 3.3 Finetune checkpoints: 5k, 12k, 20k

**Goal for `car` and `cube`:** retain **three** finetuned models corresponding to **`finetune_iteration` ∈ {5000, 12000, 20000}**.

**Operational pattern:**

1. For each scene, maintain a JSON config under  
   `external/Inpaint360GS/config/object_inpaint/inpaint360/<scene>.json`  
   (today only `bag.json` is checked in; **create `car.json` and `cube.json`** by copying `bag.json` and adjusting scene-specific fields: `select_obj_id`, `target_id`, `circle_radius`, `target_object_radius`, paths if needed).

2. Run the object-inpaint stage **three times** (or use a driver script) with:
   - `finetune_iteration: 5000` → produces `.../iteration_5000/`  
   - `finetune_iteration: 12000` → produces `.../iteration_12000/`  
   - `finetune_iteration: 20000` → produces `.../iteration_20000/`

3. After each run, invoke **render + eval** so each scene has comparable `inpaint/.../renders/` and metrics rows.

**Note on `bag`:** Historical runs already include **5k** and **20k** under `inpaint/ours_object_inpaint_virtual/`. **12k** exists as **video** and **point cloud** but **not** as a full `inpaint/.../iteration_12000/` tree; §7 documents this. When you unify all three scenes, **re-render the full inpaint branch for 12k** where missing so tables are fair.

---

## 4. Experimental Setup

### 4.1 Hardware and software

- **GPU:** cluster node (e.g. A100 / H100 — record exact model in final PDF).  
- **Python:** project venv e.g. `.venvs/latent_void_py310` with PyTorch, LPIPS, Inpaint360GS dependencies.  
- **Key fix:** evaluation script writes metrics **inside** the scene output directory (avoid writing to `/scratch` root without permission).

### 4.2 Scenes

| Scene | Role in this report | Status |
|-------|---------------------|--------|
| `bag` | Primary **verified** results | **Complete** (metrics + artifacts documented) |
| `car` | Second scene; **5k / 12k / 20k** plan | **Run pending** — tables TBD |
| `cube` | Third scene; **5k / 12k / 20k** plan | **Run pending** — tables TBD |
| `[Placeholder scene D]` | Extension | **Not run** — replace name when available |
| `[Placeholder scene E]` | Extension | **Not run** — replace name when available |

### 4.3 Hyperparameters (reference)

Scene-specific JSON controls removal and finetuning. Example **`bag`** (current repo file):

**File:** `external/Inpaint360GS/config/object_inpaint/inpaint360/bag.json`

```json
{
  "removal_thresh": 0.02,
  "select_obj_id": [182],
  "images": "images_inpaint_unseen_virtual",
  "object_path": "inpaint_2d_unseen_mask_virtual",
  "lambda_dssim": 0.8,
  "opacity_init": 0.1,
  "lambda_lpips": 0.0012,
  "finetune_iteration": 12000,
  "target_id": [182],
  "surrounding_ids": [],
  "target_object_radius": 0.585,
  "circle_radius": 0.3178
}
```

For **`car`** and **`cube`**, duplicate this file, adjust **`select_obj_id`**, **`target_id`**, and radii per scene documentation, and set **`finetune_iteration`** to **5000**, then **12000**, then **20000** for the three saved runs.

---

## 5. Implementation Notes

### 5.1 Metrics

Evaluation uses (at minimum) **PSNR**, **SSIM**, **LPIPS** with **masked**, **non-masked**, and **full** variants where applicable. Implementation: `tools/inpaint360gs_metrics_fid_masked.py` (FID optional / skipped offline).

### 5.2 Classifier checkpoint during render

`external/Inpaint360GS/render.py` may load **`point_cloud/iteration_2000/classifier.pth`** when no classifier is shipped next to object-inpaint iterations. **RGB renders** come from the **finetuned** Gaussian cloud; the **classifier** mainly affects **semantic side channels** in some visualizations. State this in the report to avoid misinterpretation.

### 5.3 Orbit / video vs test stills

Video trajectories often use **synthetic orbit** poses. **Held-out test** PNGs are usually **more stable** for reporting; use **test** renders for fair scene-to-scene comparison, and orbit as **supplementary**.

### 5.4 Native latent branch (optional section)

Diagnostics exemplars (not claimed as final quality):

- `runs/visual_inspection/inpaint360gs_bag_srun_h100_staged_render_diagnostics.png`  
- `runs/visual_inspection/inpaint360gs_bag_srun_h100_local_patch_render_diagnostics_sheet.png`  
- In-domain sanity: `runs/visual_inspection/gobjaverse_official_example_render_diagnostics_sheet.png`

**Takeaway:** native latent editing is **plausible** when reconstruction is strong; **real-scene** GSRecon/GSVAE quality was the bottleneck for this project timeline.

---

## 6. Results: Qualitative

### 6.1 Scene `bag` — fixed test views

| Comparison | Path |
|------------|------|
| Finetune 5k (example) | `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_5000/renders/test_IMG_0186.png` |
| Finetune 20k (example) | `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_20000/renders/test_IMG_0205.png` |

**Recommendation for the PDF:** pick **one** test index (e.g. `test_IMG_0186`) and show **5k | 12k | 20k** in a row once **12k** full inpaint folder exists.

### 6.2 Scene `bag` — orbit / video frames (supplementary)

| Iteration | Example path |
|-----------|----------------|
| 5000 | `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_5000/00004.png` |
| 12000 | `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_12000/00004.png` |
| 20000 | `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_20000/00004.png` |

Caption in slides: **“Orbit trajectory — may show more artifacts than held-out test cameras.”**

### 6.3 Scenes `car` and `cube` — figure placeholders

After runs complete, use **parallel paths** (replace `<scene>`):

- `output/inpaint360/<scene>/inpaint/ours_object_inpaint_virtual/iteration_5000/renders/`  
- `output/inpaint360/<scene>/inpaint/ours_object_inpaint_virtual/iteration_12000/renders/`  
- `output/inpaint360/<scene>/inpaint/ours_object_inpaint_virtual/iteration_20000/renders/`  

**Placeholder figure caption:**  
*Figure X (TBD): `car` test view [VIEW_ID] — finetune 5k / 12k / 20k.*

### 6.4 Placeholder scenes D and E

**Figure Y (TBD):** Reserved for extended evaluation — no artifacts yet.

---

## 7. Results: Quantitative

### 7.1 Scene `bag` — verified metrics

Values below are from the **project audit** (evaluation over inpaint views; two finetune checkpoints compared in one aggregate run — **per-checkpoint** rows should match your `inpaint_evaluation_results.json` structure).

**Iteration 20 000**

| Metric | Value |
|--------|--------|
| SSIM (masked) | 0.9849212169647217 |
| PSNR (masked) | 31.803434371948242 |
| LPIPS (masked) | 0.010785897262394428 |
| SSIM (non-masked) | 0.7039921879768372 |
| PSNR (non-masked) | 21.698040008544922 |
| LPIPS (non-masked) | 0.306243896484375 |
| SSIM (full) | 0.6881226897239685 |
| PSNR (full) | 21.158658981323242 |
| LPIPS (full) | 0.3196602463722229 |

**Iteration 5 000**

| Metric | Value |
|--------|--------|
| SSIM (masked) | 0.9803727269172668 |
| PSNR (masked) | 26.29412841796875 |
| LPIPS (masked) | 0.014906308613717556 |
| SSIM (non-masked) | 0.7909024357795715 |
| PSNR (non-masked) | 25.670026779174805 |
| LPIPS (non-masked) | 0.2281893640756607 |
| SSIM (full) | 0.772994875907898 |
| PSNR (full) | 22.835025787353516 |
| LPIPS (full) | 0.2442328780889511 |

**Interpretation:** Longer finetuning **improves masked-region metrics** on `bag` but **hurts non-masked and full-frame** scores — consistent with **over-optimization** toward the inpainted hole. This is a **substantive** result for the report.

**Iteration 12 000 (`bag`):** run eval when `inpaint/.../iteration_12000/` is fully populated; until then leave row:

| Metric | Value |
|--------|--------|
| All | **TBD** (pending full render + eval for 12k) |

### 7.2 Scene `car` — metrics placeholders (5k / 12k / 20k)

**Source after runs:** `output/inpaint360/car/inpaint_evaluation_results.json` (or per-run JSON if you split).

| Metric | 5k | 12k | 20k |
|--------|-----|-----|-----|
| SSIM (masked) | **TBD** | **TBD** | **TBD** |
| PSNR (masked) | **TBD** | **TBD** | **TBD** |
| LPIPS (masked) | **TBD** | **TBD** | **TBD** |
| SSIM (non-masked) | **TBD** | **TBD** | **TBD** |
| PSNR (non-masked) | **TBD** | **TBD** | **TBD** |
| LPIPS (non-masked) | **TBD** | **TBD** | **TBD** |
| SSIM (full) | **TBD** | **TBD** | **TBD** |
| PSNR (full) | **TBD** | **TBD** | **TBD** |
| LPIPS (full) | **TBD** | **TBD** | **TBD** |

### 7.3 Scene `cube` — metrics placeholders (5k / 12k / 20k)

| Metric | 5k | 12k | 20k |
|--------|-----|-----|-----|
| SSIM (masked) | **TBD** | **TBD** | **TBD** |
| PSNR (masked) | **TBD** | **TBD** | **TBD** |
| LPIPS (masked) | **TBD** | **TBD** | **TBD** |
| SSIM (non-masked) | **TBD** | **TBD** | **TBD** |
| PSNR (non-masked) | **TBD** | **TBD** | **TBD** |
| LPIPS (non-masked) | **TBD** | **TBD** | **TBD** |
| SSIM (full) | **TBD** | **TBD** | **TBD** |
| PSNR (full) | **TBD** | **TBD** | **TBD** |
| LPIPS (full) | **TBD** | **TBD** | **TBD** |

### 7.4 Cross-scene summary (fill after `car` / `cube` complete)

| Scene | Best masked PSNR (iter) | Best full PSNR (iter) | Notes |
|-------|-------------------------|------------------------|-------|
| `bag` | 31.80 (20k) | 22.84 (5k) | Tradeoff clearly visible |
| `car` | **TBD** | **TBD** | |
| `cube` | **TBD** | **TBD** | |
| Placeholder D | — | — | |
| Placeholder E | — | — | |

---

## 8. Failure Modes and Limitations

1. **Partial 12k tree on `bag`:** video and `point_cloud_object_inpaint_virtual/iteration_12000` exist; full **`inpaint/.../iteration_12000`** may be missing unless explicitly rendered — document and fix for fair 5k/12k/20k grids.  
2. **Classifier path:** semantic visualization may use an **older** `classifier.pth`; RGB still reflects finetuned Gaussians.  
3. **Orbit vs test:** orbit videos can look worse than test stills; prefer test for metrics and main figures.  
4. **Scope:** three-scene body + two placeholders; not a full Inpaint360 benchmark sweep.  
5. **DiffSplat proposal:** full training-scale DiffSplat reproduction **not** completed; inference-only story is separate and **out of scope** for this document’s empirical claims.

---

## 9. Conclusion

We presented a **reproducible eleven-stage Inpaint360GS-style pipeline** with **strong empirical evidence on `bag`** and a **clear finetuning tradeoff** between inpainted-region metrics and global fidelity. **Scenes `car` and `cube`** are aligned to the **same protocol** with **reserved 5k / 12k / 20k** checkpoint and metric slots. **Two additional scenes** are placeholders for optional extension. A **native latent** research thread is documented as **blocked by reconstruction quality** on real data, not by tooling alone.

---

## 10. Appendix A — Artifact index (`bag`)

| Artifact | Path |
|----------|------|
| Scene root | `output/inpaint360/bag/` |
| Inpaint renders 5k | `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_5000/renders/` |
| Inpaint renders 20k | `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_20000/renders/` |
| Video 5k / 12k / 20k | `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_{5000,12000,20000}/` |
| Point clouds | `output/inpaint360/bag/point_cloud_object_inpaint_virtual/iteration_*/point_cloud.ply` |
| Metrics | `output/inpaint360/bag/inpaint_evaluation_results.json` |
| Global summary | `output/inpaint360/all_scene_evaluation_results.json` |
| Pipeline status | `output/inpaint360/bag/pipeline_status.json`, `output/inpaint360/pipeline_summary.json` |

## 11. Appendix B — Artifact index placeholders (`car`, `cube`)

| Scene | Expected root |
|-------|----------------|
| `car` | `output/inpaint360/car/` |
| `cube` | `output/inpaint360/cube/` |

After each finetune run, confirm:

- `inpaint/ours_object_inpaint_virtual/iteration_{5000,12000,20000}/renders/`  
- `point_cloud_object_inpaint_virtual/iteration_{5000,12000,20000}/point_cloud.ply`  
- `inpaint_evaluation_results.json` updated or merged per your eval driver  

## 12. Appendix C — Config checklist for new scenes

- [ ] Create `external/Inpaint360GS/config/object_inpaint/inpaint360/car.json`  
- [ ] Create `external/Inpaint360GS/config/object_inpaint/inpaint360/cube.json`  
- [ ] Verify `data/inpaint360/car` and `data/inpaint360/cube` exist with required inpaint / mask folders  
- [ ] Run pipeline stages 1–11 per group runbook  
- [ ] Run three finetune budgets: **5000**, **12000**, **20000**  
- [ ] Run metric script for each scene and paste into §7.2–7.4  

---

*End of draft. Convert to PDF with your course template; replace **TBD** and placeholder scene names when runs complete.*
