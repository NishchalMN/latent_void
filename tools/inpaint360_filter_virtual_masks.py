#!/usr/bin/env python3
"""
Clean virtual inpaint masks before ply fusion.

This reduces fusion spikes by:
1) Keeping only the largest connected component.
2) Removing tiny speckles.
3) Optional erosion to shrink aggressive masks.
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mask-dir", required=True, help="Directory with virtual mask PNGs")
    p.add_argument("--backup-dir", default=None, help="If set, copy originals here first")
    p.add_argument("--min-area", type=int, default=800, help="Drop connected components smaller than this")
    p.add_argument("--open-kernel", type=int, default=3, help="Opening kernel size (0 disables)")
    p.add_argument("--erode-kernel", type=int, default=2, help="Erode kernel size (0 disables)")
    p.add_argument("--erode-iters", type=int, default=1, help="Erode iterations")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def largest_component(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask_u8
    comp_ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not comp_ids:
        return np.zeros_like(mask_u8)
    best = max(comp_ids, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    return ((labels == best).astype(np.uint8) * 255)


def main() -> int:
    args = parse_args()
    mask_dir = os.path.abspath(args.mask_dir)
    if not os.path.isdir(mask_dir):
        raise SystemExit(f"mask dir not found: {mask_dir}")

    files = sorted(f for f in os.listdir(mask_dir) if f.lower().endswith(".png"))
    if not files:
        raise SystemExit(f"no png masks in: {mask_dir}")

    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    changed = 0
    for fname in files:
        path = os.path.join(mask_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"skip bad image: {fname}")
            continue

        mask = (img > 127).astype(np.uint8) * 255
        orig_sum = int(mask.sum())

        if args.open_kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.open_kernel, args.open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        mask = largest_component(mask, args.min_area)

        if args.erode_kernel > 0 and args.erode_iters > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erode_kernel, args.erode_kernel))
            mask = cv2.erode(mask, k, iterations=args.erode_iters)

        new_sum = int(mask.sum())
        if new_sum != orig_sum:
            changed += 1
            if args.backup_dir and not args.dry_run:
                shutil.copy2(path, os.path.join(args.backup_dir, fname))
            if not args.dry_run:
                cv2.imwrite(path, mask)
        print(f"{fname}: {orig_sum} -> {new_sum}")

    print(f"done. changed {changed}/{len(files)} masks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

