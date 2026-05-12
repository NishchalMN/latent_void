#!/usr/bin/env python3
"""
Repair LaMa virtual-view depth before color–depth fusion.

LaMa depth completion often hallucinates a smooth surface at the wrong distance,
so fused points float above/below the real table. OpenCV inpainting propagates
depth from the mask boundary (table ring) into the hole.

Run AFTER LaMa depth prediction, BEFORE edit_object_removal_plyfusion.py
(or rerun fusion + 3D inpaint only).

Default fill is ``plane_ring`` (fit a plane to a border band outside the mask).
Use ``--fill inpaint`` only if you want the old OpenCV behavior (often bumpier).

Example (bag):
  python tools/inpaint360_repair_virtual_depth.py \\
    --depth-dir output/inpaint360/bag/virtual/ours_object_removal/iteration_2000/depth_completed \\
    --mask-dir  data/inpaint360/bag/inpaint_2d_unseen_mask_virtual \\
    --backup --fill plane_ring --ring-width 5
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--depth-dir", required=True, help="Directory with 00000.npy ... depth maps")
    p.add_argument("--mask-dir", required=True, help="Directory with matching 00000.png masks")
    p.add_argument("--backup", action="store_true", help="Copy .npy to .npy.bak before overwrite")
    p.add_argument(
        "--fill",
        choices=["inpaint", "median_ring", "plane_ring"],
        default="plane_ring",
        help="inpaint=cv2 (can add bumps); median_ring=constant depth from mask border; "
        "plane_ring=least-squares plane fit on border (better for tilted tables)",
    )
    p.add_argument("--inpaint-radius", type=int, default=8, help="cv2.inpaint radius (pixels); --fill inpaint only")
    p.add_argument(
        "--ring-width",
        type=int,
        default=5,
        help="Border band thickness (pixels) for median_ring / plane_ring",
    )
    p.add_argument(
        "--method",
        choices=["telea", "ns"],
        default="telea",
        help="INPAINT_TELEA (fast) vs INPAINT_NS; --fill inpaint only",
    )
    return p.parse_args()


def _morphology_ring(hole_u8: np.ndarray, width: int) -> np.ndarray:
    """Band just outside the hole: dilate(hole) - hole. width controls ring thickness."""
    k = 2 * max(width, 1) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(hole_u8, kernel)
    ring = ((dil > 0) & (hole_u8 == 0)).astype(np.uint8) * 255
    return ring


def _fill_median_ring(depth: np.ndarray, hole: np.ndarray, ring_width: int) -> np.ndarray:
    ring = _morphology_ring(hole, ring_width)
    samples = depth[ring > 0]
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        return depth
    med = float(np.median(samples))
    out = depth.copy()
    out[hole > 0] = med
    return out


def _fill_plane_ring(depth: np.ndarray, hole: np.ndarray, ring_width: int) -> np.ndarray:
    ring = _morphology_ring(hole, ring_width)
    ys, xs = np.where(ring > 0)
    z = depth[ys, xs].astype(np.float64)
    keep = np.isfinite(z)
    xs, ys, z = xs[keep].astype(np.float64), ys[keep].astype(np.float64), z[keep]
    if z.size < 12:
        return _fill_median_ring(depth, hole, ring_width)
    # z = c0*x + c1*y + c2
    a = np.column_stack((xs, ys, np.ones_like(xs)))
    coeff, _, _, _ = np.linalg.lstsq(a, z, rcond=None)
    out = depth.copy()
    hy, hx = np.where(hole > 0)
    out[hy, hx] = coeff[0] * hx.astype(np.float64) + coeff[1] * hy.astype(np.float64) + coeff[2]
    return out


def main():
    args = parse_args()
    depth_dir = os.path.abspath(args.depth_dir)
    mask_dir = os.path.abspath(args.mask_dir)
    flag = cv2.INPAINT_TELEA if args.method == "telea" else cv2.INPAINT_NS

    files = sorted(f for f in os.listdir(depth_dir) if f.endswith(".npy"))
    if not files:
        raise SystemExit("no .npy files in %s" % depth_dir)

    for fname in files:
        stem = os.path.splitext(fname)[0]
        mask_path = os.path.join(mask_dir, stem + ".png")
        if not os.path.isfile(mask_path):
            print("skip %s (no mask %s)" % (fname, mask_path))
            continue

        depth_path = os.path.join(depth_dir, fname)
        depth = np.load(depth_path)
        if depth.ndim != 2:
            depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise SystemExit("expected HxW depth, got %s" % (depth.shape,))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("skip %s (bad mask)" % fname)
            continue
        if mask.shape[:2] != depth.shape[:2]:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        hole = (mask > 127).astype(np.uint8) * 255
        if not hole.any():
            continue

        if args.fill == "inpaint":
            d_min, d_max = float(np.nanmin(depth)), float(np.nanmax(depth))
            if d_max <= d_min:
                continue
            d_norm = ((depth - d_min) / (d_max - d_min)).astype(np.float32)
            repaired = cv2.inpaint(d_norm, hole, args.inpaint_radius, flag)
            depth_out = repaired * (d_max - d_min) + d_min
        elif args.fill == "median_ring":
            depth_out = _fill_median_ring(depth, hole, args.ring_width)
        else:
            depth_out = _fill_plane_ring(depth, hole, args.ring_width)

        if args.backup:
            bak = depth_path + ".bak"
            if not os.path.isfile(bak):
                shutil.copy2(depth_path, bak)

        np.save(depth_path, depth_out.astype(depth.dtype))
        print("repaired %s" % fname)

    print("done. Re-run PLY fusion + 3D inpaint for this scene.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
