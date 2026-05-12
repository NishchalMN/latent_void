#!/usr/bin/env python3
"""Run SAM3 prompted segmentation for Inpaint360GS pipeline.

Replaces stages 2-4 (raw_sam, mask_associate, add_labels) with a single
SAM3 text-prompt segmentation pass. Writes masks in the format expected
by stage 5 (distillation) and stage 6 (target identification).

Usage (on GPU node with venv activated):
    cd /scratch/zt1/project/msml612pcs3/user/gnanesh/latent_void
    python tools/run_sam3_seg_for_inpaint360gs.py \
        --scene car --prompt "car" --data-root data/inpaint360
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--prompt", required=True, help="Text prompt for SAM3 (e.g. 'car')")
    p.add_argument("--data-root", default="data/inpaint360")
    p.add_argument("--resolution", type=int, default=2)
    p.add_argument("--sam3-root", default="external/sam3")
    p.add_argument("--object-id", type=int, default=1, help="Object label ID to assign")
    p.add_argument("--device", default="cuda")
    p.add_argument("--score-threshold", type=float, default=0.1)
    p.add_argument("--dilate-px", type=int, default=5, help="Dilate mask by N pixels")
    p.add_argument("--backup", action="store_true", help="Backup existing associated_sam/")
    return p.parse_args()


def load_sam3(sam3_root, device):
    """Try transformers backend first, then repo backend."""
    # Try transformers
    try:
        import torch
        from transformers import AutoModelForMaskGeneration, AutoProcessor
        model_id = "facebook/sam2.1-hiera-large"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForMaskGeneration.from_pretrained(model_id).to(device)
        model.eval()
        print(f"  Loaded SAM via transformers (AutoModel): {model_id}")
        return torch, model, processor, "transformers_auto"
    except Exception as e1:
        print(f"  transformers AutoModel failed: {e1}")

    # Try repo backend
    try:
        import torch
        sys.path.insert(0, os.path.abspath(sam3_root))
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(device=device, load_from_HF=True)
        model.eval()
        processor = Sam3Processor(model, device=device, confidence_threshold=0.1)
        print(f"  Loaded SAM3 via repo backend")
        return torch, model, processor, "repo"
    except Exception as e2:
        print(f"  SAM3 repo backend failed: {e2}")
        raise RuntimeError(f"Could not load SAM3: transformers={e1}, repo={e2}")


def segment_image_transformers(image, prompt, model, processor, device, score_threshold):
    """Segment using transformers AutoModel pipeline."""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    import torch
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"],
    )
    if len(masks) == 0 or masks[0].shape[0] == 0:
        return None, 0.0
    scores = outputs.iou_scores[0].cpu().numpy().flatten()
    best = int(scores.argmax())
    if scores[best] < score_threshold:
        return None, float(scores[best])
    mask = masks[0][best].cpu().numpy().squeeze()
    return mask > 0.5, float(scores[best])


def segment_image_repo(image, prompt, model, processor, score_threshold):
    """Segment using SAM3 repo backend."""
    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=prompt)
    masks = output.get("masks")
    scores = output.get("scores")
    if masks is None or len(masks) == 0:
        return None, 0.0
    import torch
    if hasattr(scores, "detach"):
        score_values = scores.detach().float().cpu().numpy()
    else:
        score_values = np.asarray(scores, dtype=np.float32)
    best = int(score_values.argmax())
    if score_values[best] < score_threshold:
        return None, float(score_values[best])
    mask = masks[best]
    if hasattr(mask, "detach"):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    return mask > 0.5, float(score_values[best])


def dilate_mask(mask, px):
    """Dilate binary mask by px pixels."""
    if px <= 0:
        return mask
    import cv2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def main():
    args = parse_args()
    scene_data = os.path.join(args.data_root, args.scene)
    img_dir = os.path.join(scene_data, f"images_{args.resolution}")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(scene_data, "images")

    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and not f.startswith("test_")
    ])
    print(f"Found {len(image_files)} training images in {img_dir}")

    # Output directories
    assoc_dir = os.path.join(scene_data, "associated_sam")
    assoc_color_dir = os.path.join(scene_data, "associated_sam_color")
    num_dir = os.path.join(scene_data, f"images_{args.resolution}_num")
    sam3_out = os.path.join(scene_data, "sam3_masks")

    # Backup existing
    if args.backup:
        for d in [assoc_dir, assoc_color_dir, num_dir]:
            bak = d + "_rawsam_bak"
            if os.path.isdir(d) and not os.path.isdir(bak):
                print(f"  Backing up {d} -> {bak}")
                shutil.copytree(d, bak)

    for d in [assoc_dir, assoc_color_dir, num_dir, sam3_out]:
        os.makedirs(d, exist_ok=True)

    # Load SAM3
    torch, model, processor, backend = load_sam3(args.sam3_root, args.device)

    results = []
    t0 = time.time()

    with torch.inference_mode():
        for i, fname in enumerate(image_files):
            img_path = os.path.join(img_dir, fname)
            image = Image.open(img_path).convert("RGB")
            stem = os.path.splitext(fname)[0]

            if backend == "transformers_auto":
                mask, score = segment_image_transformers(
                    image, args.prompt, model, processor, args.device, args.score_threshold
                )
            else:
                mask, score = segment_image_repo(
                    image, args.prompt, model, processor, args.score_threshold
                )

            if mask is None:
                mask = np.zeros((image.height, image.width), dtype=bool)
                score = 0.0
                print(f"  [{i+1}/{len(image_files)}] {fname}: NO MASK (score={score:.3f})")
            else:
                # Resize mask to match image if needed
                if mask.shape != (image.height, image.width):
                    mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                        (image.width, image.height), Image.NEAREST)) > 127
                mask = dilate_mask(mask, args.dilate_px)
                print(f"  [{i+1}/{len(image_files)}] {fname}: score={score:.3f}, "
                      f"mask_pixels={mask.sum()}/{mask.size} ({100*mask.sum()/mask.size:.1f}%)")

            # Save SAM3 mask (full resolution npy)
            np.save(os.path.join(sam3_out, f"{stem}.npy"), mask.astype(np.uint8))

            # Save as associated_sam format (uint8, object_id where mask=True)
            label_img = np.zeros((image.height, image.width), dtype=np.uint8)
            label_img[mask] = args.object_id
            Image.fromarray(label_img).save(os.path.join(assoc_dir, f"{stem}.png"))

            # Save color visualization
            color_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            color_img[mask] = [255, 0, 0]  # Red for car
            Image.fromarray(color_img).save(os.path.join(assoc_color_dir, f"{stem}.png"))

            # Save as images_N_num format (same as associated_sam for our case)
            # This replaces stage 4 (add_label_num)
            Image.fromarray(label_img).save(os.path.join(num_dir, fname))

            results.append({
                "image": fname, "score": score,
                "mask_pixels": int(mask.sum()), "total_pixels": int(mask.size),
            })

    elapsed = time.time() - t0

    # Write scene.json (required by distillation + downstream)
    scene_json = {
        "num_classes": args.object_id + 1,  # bg(0) + car(1) = 2 classes
        "object_ids": [args.object_id],
        "prompt": args.prompt,
        "backend": backend,
    }
    with open(os.path.join(assoc_dir, "scene.json"), "w") as f:
        json.dump(scene_json, f, indent=2)

    # Write summary
    scores = [r["score"] for r in results if r["score"] > 0]
    summary = {
        "scene": args.scene,
        "prompt": args.prompt,
        "backend": backend,
        "num_views": len(image_files),
        "num_valid_masks": len(scores),
        "mean_score": float(np.mean(scores)) if scores else 0,
        "min_score": float(np.min(scores)) if scores else 0,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    summary_path = os.path.join(sam3_out, "sam3_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  SAM3 segmentation complete for '{args.scene}'")
    print(f"  Valid masks: {len(scores)}/{len(image_files)}")
    print(f"  Mean score: {np.mean(scores):.3f}" if scores else "  No valid masks!")
    print(f"  Time: {elapsed:.0f}s")
    print(f"  Masks written to: {assoc_dir}")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
