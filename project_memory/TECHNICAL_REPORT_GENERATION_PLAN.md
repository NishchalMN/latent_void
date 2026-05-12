# Plan: Detailed Technical Report (honest scope)

**Audience:** course graders  
**Primary contribution:** end-to-end execution of the **Inpaint360GS-style multi-stage pipeline** with **real artifacts, logs, and metrics** on at least scene **`bag`**.  
**Secondary thread:** investigation toward native latent / DiffSplat-adjacent tooling inside `latent_void`; **not** claimed as completed paper reproduction.

**Tone rule:** every quantitative claim ties to a **file path** or **command log**; every limitation is explicit.

---

## 1) Executive framing (write this first, 1 page)

**Problem:** 3D-consistent object removal and inpainting on real multi-view captures.

**What we delivered:**

- Full orchestration of **Inpaint360GS-compatible stages** (train/remove/2D inpaint/3D finetune/render/eval) with outputs under `output/inpaint360/`.
- Quantitative evaluation where implemented (masked vs non-masked / full metrics).
- Transparent discussion of **optimization tradeoffs** (e.g. finetune iterations: hole fill vs background).

**What we did not deliver (and why):**

- **Original DiffSplat-centric proposal at full paper fidelity:** upstream DiffSplat **training** assumes dataset packaging and infrastructure described in the official repo (internal-scale data layout). Complete **benchmark-scale reproduction** and **training-time ablations** were **out of scope for available time** on this project timeline—not a substitute for claiming fake numbers.
- **Native latent inpainting at publication quality:** blocked upstream by reconstruction / domain alignment; documented as future work.

Avoid overstating “missing weights” unless you verify a specific checkpoint gap; safer wording: **“Inference-focused reproduction and paper-scale training/eval were deferred due to dataset/training pipeline and time constraints.”**

---

## 2) Report skeleton (recommended section order)

### Abstract (150–250 words)

- Task, method family (Inpaint360GS pipeline), primary scene(s), headline quantitative outcome (real JSON), one limitation sentence.

### 1. Introduction

- Motivation: why 3D consistency matters vs pure 2D inpainting.
- Project evolution: proposed DiffSplat study vs pragmatic pivot to reproducible Inpaint360GS baseline.

### 2. Background & related work

- 3D Gaussian Splatting (cite Kerbl et al.).
- Inpaint360GS / Gaussian Grouping–style object association (cite their paper/repo).
- Optional one paragraph on latent/generative splat methods (DiffSplat as motivation only).

### 3. System overview

- Diagram: data → stages → artifacts.
- **Repository roles:**
  - `external/Inpaint360GS`: baseline implementation.
  - `latent_void`: adapters, configs, metrics fixes, optional native-latent experiments.

### 4. Method: eleven-stage pipeline (name stages explicitly)

For each stage, use a **mini-subsection** with:

| Stage | Purpose | Main inputs | Main outputs | Tool/script |
|------|---------|-------------|--------------|-------------|

Typical Inpaint360GS-style narrative (adjust labels to match **your** exact orchestration script):

1. Dataset / COLMAP / camera poses  
2. Train base 3DGS  
3. SAM / segmentation  
4. 3D mask association  
5. Object removal (Gaussian deletion / classifier)  
6. LaMa (or configured) 2D inpainting on inpaint views  
7. Fusion / point preparation  
8. **Object inpaint finetuning** (`edit_object_inpaint.py` lineage)  
9. Render train/test/inpaint/video  
10. Metrics (PSNR, SSIM, LPIPS; masked variants)  
11. Packaging (`pipeline_status.json`, summaries)

**Critical:** paste **actual commands** or pointer to `pipeline_summary.json` / tmux logs.

### 5. Experimental setup

- Hardware (GPU model), OS module notes if relevant.
- Python env path (e.g. `.venvs/latent_void_py310`).
- Key hyperparameters: `finetune_iteration`, `lambda_lpips`, `lambda_dssim`, removal thresholds—cite **`external/Inpaint360GS/config/object_inpaint/inpaint360/bag.json`** (or whichever config you used).

### 6. Results

**6.1 Qualitative**

- Fixed-camera comparisons: `iteration_5000` vs `iteration_20000` renders (paths below).
- Optional orbit/video frames with caveat that orbit is **harder** than held-out test poses.

**6.2 Quantitative**

- Table from **`output/inpaint360/bag/inpaint_evaluation_results.json`** (and/or aggregate JSON).
- Interpret masked vs non-masked divergence as **documented tradeoff**, not failure.

**6.3 Ablations / sensitivities (honest mini-ablation)**

- Effect of finetune length (5k vs 20k)—you already have numbers.
- Optional: classifier fallback behavior (`point_cloud/iteration_2000/classifier.pth`) — clarify it affects **semantic-side outputs**, not necessarily RGB.

### 7. Failure modes & debugging notes

- Fork/`EAGAIN` on cluster; mitigation (clean sessions, limits).
- Partial artifact trees (e.g. **12k**: video + ply exist; full `inpaint/.../iteration_12000/` branch absent)—explain **why** (which render mode was run).
- Native latent branch: reconstruction gate failed; cite diagnostic PNG paths under `runs/visual_inspection/`.

### 8. Ethics & limitations

- Dataset licensing / attribution for Inpaint360GS assets.
- Scope: single-scene depth vs multi-scene benchmark.

### 9. Conclusion & future work

- Bullet future work: classifier checkpoints per inpaint iteration; complete 12k export; second scene; latent denoiser after reconstruction improves.

### Appendix A: Artifact index

- Bullet list of **every path** a grader can open cold.

### Appendix B: Command transcript

- Trimmed log excerpts or `tee` logs under `output/inpaint360/bag/logs/`.

---

## 3) Figures & tables checklist (generate these deliberately)

**Must-have figures**

1. Pipeline diagram (boxes + arrows).
2. Side-by-side RGB: **same** `test_IMG_*` for **5k vs 20k**.
3. Optional third panel: GT or “before removal” if you have it in-tree.
4. One plot or table: masked vs non-masked metrics.

**Nice-to-have**

5. Zoom crop on object region for hole-fill story.
6. One orbit strip with caption “novel trajectory—artifacts expected.”

**Primary paths already documented** (reuse from `BAG_FULL_TECHNICAL_REPORT_2026-05-05.md`):

- `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_5000/renders/test_IMG_0186.png`
- `output/inpaint360/bag/inpaint/ours_object_inpaint_virtual/iteration_20000/renders/test_IMG_0205.png`
- Video frames under `output/inpaint360/bag/video/ours__object_inpaint_virtual/iteration_*/`

---

## 4) Writing workflow (efficient)

### Pass 1 — Evidence audit (2–4 hours)

- Open `output/inpaint360/bag/pipeline_status.json`, `pipeline_summary.json`, `inpaint_evaluation_results.json`.
- List stages with status `ok` / missing.
- Screenshot or embed **file paths** into a draft Appendix.

### Pass 2 — Narrative draft (4–8 hours)

- Write Sections 3–6 using the stage table template.
- Insert figure placeholders with **paths**, not screenshots yet.

### Pass 3 — Polish & honesty pass (2–4 hours)

- Remove any claim without a path/citation.
- Unify notation (iteration naming, folder naming).
- Add limitation paragraphs where evidence is partial.

### Pass 4 — Group consistency

- One student owns numbers table; one owns diagrams; one owns related work.

---

## 5) DiffSplat paragraph template (accurate, non-dismissive)

Use something like:

> Our initial project direction aligned with reproducing DiffSplat-class feed-forward Gaussian generation. The official DiffSplat release emphasizes **inference** with HuggingFace checkpoints, while **full training-scale reproduction** depends on dataset packaging and compute described upstream (including notes that internal dataset layouts may require adaptation). Given course time constraints, we prioritized an alternative reproducible objective: **running the full Inpaint360GS removal/inpainting pipeline on released scene data**, producing stored artifacts and quantitative metrics. DiffSplat remains motivating related work and motivated portions of our `latent_void` scaffolding.

Adjust only if you personally verified a specific missing artifact.

---

## 6) Deliverables bundle for graders

Create `reports/submission_readme.pdf` or top-level `SUBMISSION.md` pointing to:

- This technical report PDF.
- `output/inpaint360/bag/` subtree listing (or zip excluding massive binaries if allowed).
- `external/Inpaint360GS` commit hash submodule note.

---

## 7) Optional next doc to generate from this plan

- `project_memory/FINAL_TECHNICAL_REPORT_DRAFT.md` — paste sections here first; export to PDF when frozen.
