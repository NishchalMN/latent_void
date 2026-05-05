#!/usr/bin/env python3
"""Reduce SAM automatic segmentation masks to fewer classes.

SAM's automatic mask generator produces 100-255+ tiny segments per image.
Inpaint360GS's semantic distillation expects ~20-30 meaningful classes.
This script merges the smallest segments until we have at most --max-classes.

Usage:
    python tools/reduce_sam_segments.py \
        --mask-dir data/inpaint360/bag/raw_sam \
        --max-classes 30
"""
import argparse
import os
import sys
import numpy as np
from PIL import Image
from collections import Counter


def reduce_masks_in_dir(mask_dir, max_classes):
    """Reduce all mask images in a directory to at most max_classes."""
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    if not files:
        print(f"No .png files in {mask_dir}")
        return

    # Pass 1: count global label frequencies across all images
    global_counts = Counter()
    for fname in files:
        mask = np.array(Image.open(os.path.join(mask_dir, fname)))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        for val, cnt in zip(*np.unique(mask, return_counts=True)):
            if val == 0:
                continue
            global_counts[int(val)] += int(cnt)

    total_labels = len(global_counts)
    print(f"Found {total_labels} unique labels across {len(files)} images")

    if total_labels <= max_classes:
        print(f"Already <= {max_classes} classes, nothing to do")
        return

    # Keep top max_classes labels by frequency; merge the rest into nearest kept label
    sorted_labels = sorted(global_counts.items(), key=lambda x: -x[1])
    keep_labels = set(lbl for lbl, _ in sorted_labels[:max_classes])
    # Build a remap: kept labels get sequential IDs 1..max_classes, rest → 0 (background)
    remap = {0: 0}
    new_id = 1
    for lbl, _ in sorted_labels[:max_classes]:
        remap[lbl] = new_id
        new_id += 1

    print(f"Keeping top {max_classes} labels, merging {total_labels - max_classes} into background")

    # Pass 2: apply remap to all images
    for fname in files:
        path = os.path.join(mask_dir, fname)
        mask = np.array(Image.open(path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        new_mask = np.zeros_like(mask)
        for old_id, new_id in remap.items():
            new_mask[mask == old_id] = new_id

        Image.fromarray(new_mask.astype(np.uint8)).save(path)

    print(f"Done. Reduced to {max_classes} classes + background")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--max-classes", type=int, default=30)
    args = parser.parse_args()
    reduce_masks_in_dir(args.mask_dir, args.max_classes)


if __name__ == "__main__":
    main()
