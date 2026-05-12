# SAM3 Direct-Target Pipeline

This folder contains a separate workflow that bypasses legacy Inpaint360GS
segment-all stages and uses a known SAM3 text prompt target directly.

## Script

- `run_sam3_direct_pipeline.py`

## What it bypasses

- legacy stages 2..6: raw SAM segment-all, reduce/remap association, semantic
  distillation, auto target-ID selection

## What it still reuses

- LaMa prep/run/postprocess
- optional depth-fix tools
- stage 10/11 fusion + inpaint + eval from `tools/run_inpaint360gs_full.py`

## Example

```bash
cd /scratch/zt1/project/msml612pcs3/user/gnanesh/latent_void
source .venvs/latent_void_py310/bin/activate

python tools/sam3_direct/run_sam3_direct_pipeline.py \
  --scene car \
  --prompt "car" \
  --sam3-root external/sam3 \
  --sam3-checkpoint checkpoints/sam3 \
  --resolution 2 \
  --finetune-iters 12000 \
  --checkpoint-video-iters "5000 8000" \
  --run-depth-fix
```

## Notes

- The script writes direct-path intermediates under `output/inpaint360/<scene>/`
  (and temporary pruned model under `output/inpaint360/<scene>_sam3_direct_model/`).
- It preserves expected downstream folder contracts under
  `virtual/ours_2000/` and `virtual/ours_object_removal/iteration_2000/`.
