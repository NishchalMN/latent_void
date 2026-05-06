"""Generate per-view instance masks using SAM 3 for the Inpaint360GS pipeline.

Replaces Inpaint360GS's raw_mask_sam.py by using the already-downloaded SAM 3
checkpoint instead of requiring original SAM ViT-H + CropFormer/Detectron2.

Output format: one PNG per image in <scene>/raw_sam/, where pixel value = object
ID (0 = background, 1..N = detected objects). This is exactly what
mask_associate.py expects with --mask_generator sam.

Usage (on GPU node):
    python tools/generate_sam3_instance_masks.py \
        --checkpoint checkpoints/sam3 \
        --data-root data/inpaint360 \
        --scenes car bag doppelherz \
        --resolution 2
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def load_sam3_model(checkpoint_path, device):
    """Load SAM 3 model and processor from local checkpoint."""
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor

    print(f"Loading SAM 3 from {checkpoint_path}...")
    model = Sam3VideoModel.from_pretrained(
        checkpoint_path, local_files_only=True
    ).to(device)
    model.eval()
    processor = Sam3VideoProcessor.from_pretrained(
        checkpoint_path, local_files_only=True
    )
    print(f"SAM 3 loaded on {device}")
    return model, processor


def generate_instance_masks_for_image(model, processor, image, device):
    """Run automatic instance segmentation on a single image.

    Returns an (H, W) uint8 array with pixel values = object IDs.
    """
    import torch

    inputs = processor(images=image, return_tensors="pt").to(device)
    original_size = [image.size[::-1]]  # (H, W)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=original_size, threshold=0.3
    )

    if not results or len(results) == 0:
        return np.zeros(original_size[0], dtype=np.uint8)

    result = results[0]
    seg_map = result["segmentation"]

    if hasattr(seg_map, "cpu"):
        seg_map = seg_map.cpu().numpy()

    seg_map = seg_map.astype(np.uint8)
    return seg_map


def generate_masks_grid_fallback(model, processor, image, device):
    """Fallback: use grid point prompts if automatic detection doesn't work.

    Generates a grid of points, runs SAM 3 on each, and combines into
    an instance mask.
    """
    import torch

    w, h = image.size
    mask_id = np.zeros((h, w), dtype=np.uint8)
    obj_count = 0

    points_per_side = 16
    xs = np.linspace(0, w - 1, points_per_side + 2)[1:-1].astype(int)
    ys = np.linspace(0, h - 1, points_per_side + 2)[1:-1].astype(int)
    grid = [(int(x), int(y)) for y in ys for x in xs]

    batch_size = 32
    for batch_start in range(0, len(grid), batch_size):
        batch_points = grid[batch_start:batch_start + batch_size]
        input_points = [[[p[0], p[1]] for p in batch_points]]

        try:
            inputs = processor(
                images=image,
                input_points=input_points,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            masks = processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"],
            )
            if masks and len(masks) > 0:
                for m in masks[0]:
                    m_np = m.cpu().numpy().squeeze()
                    if m_np.ndim == 2 and m_np.sum() > 100:
                        overlap = (mask_id > 0) & (m_np > 0.5)
                        if overlap.sum() < m_np.sum() * 0.5:
                            obj_count += 1
                            if obj_count >= 255:
                                return mask_id
                            mask_id[m_np > 0.5] = obj_count
        except Exception as e:
            print(f"    Grid batch failed: {e}")
            continue

    return mask_id


def process_scene(model, processor, data_root, scene_name, resolution, device):
    """Generate instance masks for all images in a scene."""
    scene_dir = os.path.join(data_root, scene_name)
    image_dir = os.path.join(scene_dir, f"images_{resolution}")
    output_dir = os.path.join(scene_dir, "raw_sam")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isdir(image_dir):
        print(f"  ERROR: Image directory not found: {image_dir}")
        return False

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".JPG"))
    ])
    print(f"  Processing {len(image_files)} images from {image_dir}")

    for i, fname in enumerate(image_files):
        img_path = os.path.join(image_dir, fname)
        image = Image.open(img_path).convert("RGB")

        try:
            seg_map = generate_instance_masks_for_image(model, processor, image, device)
        except Exception as e:
            print(f"    Automatic detection failed for {fname}: {e}")
            print(f"    Trying grid-point fallback...")
            try:
                seg_map = generate_masks_grid_fallback(model, processor, image, device)
            except Exception as e2:
                print(f"    Grid fallback also failed: {e2}")
                seg_map = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)

        n_objects = len(np.unique(seg_map)) - (1 if 0 in seg_map else 0)
        out_name = os.path.splitext(fname)[0] + ".png"
        out_path = os.path.join(output_dir, out_name)
        Image.fromarray(seg_map).save(out_path)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"    [{i+1}/{len(image_files)}] {fname}: {n_objects} objects")

    print(f"  Saved masks to {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate SAM 3 instance masks for Inpaint360GS"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to SAM 3 checkpoint directory")
    parser.add_argument("--data-root", required=True,
                        help="Root of dataset (e.g. data/inpaint360)")
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--resolution", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model, processor = load_sam3_model(args.checkpoint, args.device)

    for scene in args.scenes:
        print(f"\n{'='*50}")
        print(f"  Scene: {scene}")
        print(f"{'='*50}")
        ok = process_scene(
            model, processor, args.data_root, scene, args.resolution, args.device
        )
        if not ok:
            print(f"  FAILED: {scene}")

    print("\nDone.")


if __name__ == "__main__":
    main()
