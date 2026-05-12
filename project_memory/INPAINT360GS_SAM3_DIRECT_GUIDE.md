# SAM3-direct Inpaint360GS path (step-by-step)

This document explains the **prompted SAM3 → prune → virtual orbit → LaMa → fusion → 3D inpaint** workflow used when you **do not** want the full legacy pipeline (segment-all, association, distillation, auto target ID).

Canonical repo tools:

- `tools/run_sam3_multiview.py` — text-prompt masks per view
- `tools/prune_3dgs_with_inpaint360gs_masks.py` — CPU projection voting, writes pruned PLY
- `tools/sam3_direct/run_sam3_direct_pipeline.py` — end-to-end driver (optional)
- `tools/render_inpaint360_virtual_orbits.py` — **virtual RGB+depth orbits** + copy to `output/inpaint360/<scene>/virtual/…` (for step-by-step runs)

---

## Your understanding (yes)

After pruning, we place **virtual cameras** on a short **orbit** around the scene. We do this twice:

1. **Full model** → `virtual/ours_2000/` (car still visible in RGB; reference orbit)
2. **Pruned model** → `virtual/ours_object_removal/iteration_2000/` (car removed; “hole” region)

LaMa then fills **color + depth** on the **pruned** branch using masks derived from SAM3 on the **full** branch’s virtual RGB (same frame indices, aligned cameras).

---

## Q: Why virtual cameras? Why not reuse train/test views?

**Short:** The pipeline needs **paired** views with **identical** extrinsics: full vs pruned. Training cameras are fixed; an **orbit** gives a compact set of **novel** viewpoints that all share the same path, so frame `00005.png` means the **same** pose for both models.

**Longer:**

- **Pairing:** Fusion and LaMa prep assume matching `00000`…`00029` names under `ours_2000` (what *should* be there) vs `ours_object_removal/iteration_2000` (hole render). Train/test image names don’t line up as a synchronized orbit.
- **Coverage:** An orbit looks **around** the object and exposes the region you removed from angles useful for 3D inpainting; not every train view sees the hole equally well.
- **Contract:** `prepare_lama_data.py --inpaint2lama` and fusion code are wired to the **virtual** folder layout used in Inpaint360GS.

You *could* engineer a variant that only uses train views; it would not match the stock Inpaint360GS data layout without extra glue.

---

## Q: What is `iteration_2000`?

It is **not** your vanilla 3DGS training step (e.g. 30000). In the **original** Inpaint360GS removal stage, the edited scene is often exported under a removal checkpoint labeled `iteration_2000`. Downstream scripts **hard-code or default** paths like:

`virtual/ours_object_removal/iteration_2000/renders`  
`virtual/ours_object_removal/iteration_2000/depth`  
`virtual/ours_object_removal/iteration_2000/depth_completed` (after LaMa)

In the **SAM3-direct** path you **skip** full `edit_object_removal` training, but you still **reuse that folder name** by rendering the **pruned** PLY into the same directory layout so fusion/LaMa don’t need rewrites.

---

## Q: Where do renders land vs where SAM3/LaMa look?

`_render_virtual_views` writes under **`<3dgs_output>/virtual/...`** (e.g. `output/inpaint360/car/3dgs_output/virtual/ours_2000`).

The SAM3 virtual manifest in `run_sam3_direct_pipeline.py` expects:

`output/inpaint360/<scene>/virtual/ours_2000/renders`

So we **copy** `ours_2000` (and the pruned removal tree) into `output/inpaint360/<scene>/virtual/...`.  
`tools/render_inpaint360_virtual_orbits.py` and the fixed `run_sam3_direct_pipeline.py` perform that copy.

---

## Step order (after vanilla 3DGS + SAM3 train masks + prune)

1. **Virtual orbits** (this doc’s next practical step):

   ```bash
   export PYTHONPATH=$PWD/external/Inpaint360GS:$PWD/external/Inpaint360GS/gaussian_splatting:$PYTHONPATH

   python tools/render_inpaint360_virtual_orbits.py \
     --scene car \
     --base-model output/inpaint360/car/3dgs_output \
     --pruned-model output/inpaint360/car_sam3_direct_model/3dgs_output
   ```

2. **SAM3 on virtual RGB** (full-model orbit only): manifest → `run_sam3_multiview.py` → masks as PNG in `data/inpaint360/<scene>/inpaint_2d_unseen_mask_virtual/`.  
   Virtual frames are named `00000.png` … — mask filenames must match; `_npy_masks_to_png_255` in `run_sam3_direct_pipeline.py` uses `sam3_results.json` so stems stay aligned.

   Example (`scene=car`, repo root):

   ```bash
   python - <<'PY'
   import json, os
   scene = "car"
   renders = os.path.join("output", "inpaint360", scene, "virtual", "ours_2000", "renders")
   out = os.path.join("output", "inpaint360", scene, "sam3_direct_virtual_manifest.json")
   views = []
   for name in sorted(os.listdir(renders)):
       if not name.lower().endswith(".png"):
           continue
       views.append({
           "view_id": "virtual_%04d" % len(views),
           "image_path": os.path.abspath(os.path.join(renders, name)),
       })
   os.makedirs(os.path.dirname(out), exist_ok=True)
   json.dump({"views": views}, open(out, "w"), indent=2)
   print(len(views), "views ->", out)
   PY

   python tools/run_sam3_multiview.py \
     --sam3-root external/sam3 \
     --checkpoint-path checkpoints/sam3 \
     --manifest output/inpaint360/car/sam3_direct_virtual_manifest.json \
     --prompt "car" \
     --output-dir output/inpaint360/car/sam3_direct_virtual_masks_npy \
     --backend transformers \
     --device cuda

   python - <<'PY'
   import importlib.util
   spec = importlib.util.spec_from_file_location(
       "sd", "tools/sam3_direct/run_sam3_direct_pipeline.py")
   m = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(m)
   m._npy_masks_to_png_255(
       "output/inpaint360/car/sam3_direct_virtual_masks_npy",
       "data/inpaint360/car/inpaint_2d_unseen_mask_virtual",
   )
   print("masks -> data/inpaint360/car/inpaint_2d_unseen_mask_virtual/")
   PY
   ```

### Hole hardening (recommended if pruned virtual RGB looks “smeared”)

Pruned **Gaussian renders** often show **soft, muddy holes** at object boundaries (semi-transparent edge Gaussians). **Centered** views look cleaner; **grazing / near-image-border** views get **streaky RGB fringing** and pavement “pulled” into the void — the **SAM mask can still be correct**, but **splatting leaves junk outside** a tight mask. LaMa then sees ambiguous input and under-fills.

**Do not rely only on uniform `--dilate`:** it expands the **entire** mask and can mark **good cobblestone** for inpainting.

**Preferred:** `tools/harden_pruned_virtual_renders.py` with **`--dark-fringe-luma`**: starting from the SAM hole, repeatedly add **only** 4-neighbors whose **luma** is below a threshold (the smear is darker / muddier than typical pavement). Then hard-fill the combined hole to black.

```bash
# Restore originals if you already hardened once:
#   rm -rf output/inpaint360/car/virtual/ours_object_removal/iteration_2000/renders
#   cp -a .../renders_backup .../renders

python tools/harden_pruned_virtual_renders.py --scene car --backup \
  --dark-fringe-luma 72 --dark-fringe-iters 14 \
  --update-masks --backup-masks
```

**Important:** `--update-masks` rewrites `data/inpaint360/<scene>/inpaint_2d_unseen_mask_virtual/*.png`
to match the **expanded** hole (SAM + dilate + dark-fringe). Without this, pruned renders can be filled
outside the SAM outline but LaMa still uses the **old tight** `*_mask.png`, so fringe stays unpainted/black.

Tune: if fringe remains, try **`--dark-fringe-luma 78`** (slightly more aggressive) or **`--dark-fringe-iters 18`**. If **pavement** gets eaten, **lower** luma (e.g. 65) or **fewer** iters.

Optional small uniform dilate **after** you trust dark-fringe: `--dilate 2`.

**Orbit mitigation:** smear worsens when the object hugs the frame — re-run `tools/render_inpaint360_virtual_orbits.py` with a **smaller** `--circle-radius` (e.g. `0.85`) so the car stays more centered (trade-off: less parallax).

This edits `output/inpaint360/<scene>/virtual/ours_object_removal/iteration_<tag>/renders/*.png` in place (with `renders_backup/` if `--backup`). Use `--iteration-tag 2000` if your virtual folder name differs.

3. **LaMa** (from `external/Inpaint360GS`, scene output = `output/inpaint360/<scene>`):

   ```bash
   export PYTHONPATH=$PWD/external/Inpaint360GS:$PWD/external/Inpaint360GS/gaussian_splatting:$PYTHONPATH

   python external/Inpaint360GS/tools/prepare_lama_data.py \
     -s data/inpaint360/car -m output/inpaint360/car -r 2 --inpaint2lama

   ( cd external/Inpaint360GS/LaMa && \
     TORCH_HOME=$PWD PYTHONPATH=$PWD python bin/predict_color.py --data_name 360_car_virtual )
   ( cd external/Inpaint360GS/LaMa && \
     TORCH_HOME=$PWD PYTHONPATH=$PWD python bin/predict_depth.py --data_name 360_car_virtual )

   python external/Inpaint360GS/tools/prepare_lama_data.py \
     -s data/inpaint360/car -m output/inpaint360/car -r 2
   ```

4. **Optional (recommended if fusion looked bad before):** `tools/inpaint360_align_completed_depth.py` + `tools/inpaint360_project_completed_to_hole_plane.py` on `depth_completed` vs `depth` with masks in `data/inpaint360/car/inpaint_2d_unseen_mask_virtual` (see bag recipe).

5. **`run_inpaint360gs_full.py --start-stage 10`** — fusion + 3D inpaint + eval.

---

## Pruning script note

`prune_3dgs_with_inpaint360gs_masks.py` is **CPU-only** (NumPy projection). It does **not** use GPU VRAM.

---

## Related

- Full legacy 11-stage driver: `tools/run_inpaint360gs_full.py`
- Virtual depth recipe (bag): `project_memory/INPAINT360GS_BAG_VIRTUAL_DEPTH_FUSION_RECIPE.md`
