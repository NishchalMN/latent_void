#!/usr/bin/env python3
"""
Force masked depth_completed pixels onto a plane fitted from depth_hole ring pixels.

This is stronger than a*z+b alignment and helps when the patch looks correct from
only one angle (sheet-like tilt across views).
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--completed-dir", required=True, help=".../depth_completed")
    p.add_argument("--hole-dir", required=True, help=".../depth")
    p.add_argument("--mask-dir", required=True, help=".../inpaint_2d_unseen_mask_virtual")
    p.add_argument("--backup", action="store_true")
    p.add_argument("--ring-width", type=int, default=5)
    p.add_argument("--min-samples", type=int, default=200)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def make_ring(hole: np.ndarray, width: int) -> np.ndarray:
    k = 2 * max(1, width) + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(hole.astype(np.uint8), ker) > 0
    return dil & (~hole)


def fit_plane_xy_z(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    # z = a*x + b*y + c
    A = np.column_stack([xs, ys, np.ones_like(xs)])
    coef, _, _, _ = np.linalg.lstsq(A, zs, rcond=None)
    return coef


def main() -> int:
    args = parse_args()
    cdir = os.path.abspath(args.completed_dir)
    hdir = os.path.abspath(args.hole_dir)
    mdir = os.path.abspath(args.mask_dir)

    files = sorted(f for f in os.listdir(cdir) if f.endswith(".npy"))
    if not files:
        raise SystemExit(f"no npy files in {cdir}")

    updated, skipped = 0, 0
    for fname in files:
        stem = os.path.splitext(fname)[0]
        cp = os.path.join(cdir, fname)
        hp = os.path.join(hdir, fname)
        mp = os.path.join(mdir, stem + ".png")
        if not (os.path.isfile(hp) and os.path.isfile(mp)):
            skipped += 1
            continue

        zc = np.load(cp).astype(np.float32)
        zh = np.load(hp).astype(np.float32)
        if zc.shape != zh.shape:
            skipped += 1
            continue

        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped += 1
            continue
        if mask.shape[:2] != zc.shape[:2]:
            mask = cv2.resize(mask, (zc.shape[1], zc.shape[0]), interpolation=cv2.INTER_NEAREST)
        hole = mask > 127
        if hole.sum() < 30:
            skipped += 1
            continue

        ring = make_ring(hole, args.ring_width)
        valid = ring & np.isfinite(zh) & (zh > 1e-8)
        if int(valid.sum()) < args.min_samples:
            skipped += 1
            continue

        ys, xs = np.where(valid)
        zs = zh[valid].astype(np.float64)
        coef = fit_plane_xy_z(xs.astype(np.float64), ys.astype(np.float64), zs)

        hy, hx = np.where(hole)
        z_plane = coef[0] * hx.astype(np.float64) + coef[1] * hy.astype(np.float64) + coef[2]

        out = zc.copy()
        out[hy, hx] = z_plane.astype(out.dtype, copy=False)

        if args.backup:
            bak = cp + ".bak"
            if not os.path.isfile(bak) and not args.dry_run:
                shutil.copy2(cp, bak)
        if not args.dry_run:
            np.save(cp, out)

        updated += 1
        print(f"{stem}: plane z={coef[0]:.6f}*x + {coef[1]:.6f}*y + {coef[2]:.6f}")

    print(f"done. updated={updated}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
