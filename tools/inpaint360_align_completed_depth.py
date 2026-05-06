#!/usr/bin/env python3
"""
Align depth_completed to depth_hole per frame using a ring around the virtual mask.

For each frame:
  z_aligned = a * z_completed + b
where (a, b) are fit on ring pixels outside the mask to match depth_hole.
"""

from __future__ import annotations

import argparse
import os
import shutil

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--completed-dir", required=True, help="Path to depth_completed/*.npy")
    p.add_argument("--hole-dir", required=True, help="Path to depth/*.npy from removal render")
    p.add_argument("--mask-dir", required=True, help="Path to inpaint_2d_unseen_mask_virtual/*.png")
    p.add_argument("--backup", action="store_true", help="Backup completed depth as *.npy.bak")
    p.add_argument("--ring-width", type=int, default=5, help="Ring width in pixels outside mask")
    p.add_argument("--min-samples", type=int, default=300, help="Minimum ring pixels to fit")
    p.add_argument("--clamp-a-min", type=float, default=0.5, help="Lower clamp for scale a")
    p.add_argument("--clamp-a-max", type=float, default=2.0, help="Upper clamp for scale a")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def make_ring(hole: np.ndarray, width: int) -> np.ndarray:
    k = 2 * max(1, width) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(hole.astype(np.uint8), kernel) > 0
    return dil & (~hole)


def robust_fit_ab(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Initial least squares
    A = np.column_stack([x, np.ones_like(x)])
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(sol[0]), float(sol[1])

    # One IRLS-like refinement with Huber weights
    r = y - (a * x + b)
    s = np.median(np.abs(r - np.median(r))) + 1e-8
    k = 1.345 * s
    w = np.ones_like(r)
    big = np.abs(r) > k
    w[big] = k / (np.abs(r[big]) + 1e-8)
    Aw = A * w[:, None]
    yw = y * w
    sol2, _, _, _ = np.linalg.lstsq(Aw, yw, rcond=None)
    return float(sol2[0]), float(sol2[1])


def main() -> int:
    args = parse_args()
    cdir = os.path.abspath(args.completed_dir)
    hdir = os.path.abspath(args.hole_dir)
    mdir = os.path.abspath(args.mask_dir)

    files = sorted(f for f in os.listdir(cdir) if f.endswith(".npy"))
    if not files:
        raise SystemExit(f"no .npy files in {cdir}")

    updated = 0
    skipped = 0
    for fname in files:
        stem = os.path.splitext(fname)[0]
        cp = os.path.join(cdir, fname)
        hp = os.path.join(hdir, fname)
        mp = os.path.join(mdir, stem + ".png")
        if not (os.path.isfile(hp) and os.path.isfile(mp)):
            skipped += 1
            print(f"skip {stem}: missing hole or mask")
            continue

        zc = np.load(cp).astype(np.float32)
        zh = np.load(hp).astype(np.float32)
        if zc.shape != zh.shape:
            skipped += 1
            print(f"skip {stem}: shape mismatch completed {zc.shape} hole {zh.shape}")
            continue

        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            skipped += 1
            print(f"skip {stem}: bad mask")
            continue
        if mask.shape[:2] != zc.shape[:2]:
            mask = cv2.resize(mask, (zc.shape[1], zc.shape[0]), interpolation=cv2.INTER_NEAREST)
        hole = mask > 127
        if hole.sum() < 50:
            skipped += 1
            print(f"skip {stem}: tiny mask")
            continue

        ring = make_ring(hole, args.ring_width)
        valid = ring & np.isfinite(zc) & np.isfinite(zh)
        if int(valid.sum()) < args.min_samples:
            skipped += 1
            print(f"skip {stem}: not enough ring samples ({int(valid.sum())})")
            continue

        x = zc[valid].astype(np.float64)
        y = zh[valid].astype(np.float64)
        a, b = robust_fit_ab(x, y)
        a = float(np.clip(a, args.clamp_a_min, args.clamp_a_max))

        zc_aligned = (a * zc + b).astype(zc.dtype, copy=False)

        # Keep untouched outside mask to minimize side effects.
        z_out = zc.copy()
        z_out[hole] = zc_aligned[hole]

        if args.backup:
            bak = cp + ".bak"
            if not os.path.isfile(bak) and not args.dry_run:
                shutil.copy2(cp, bak)

        if not args.dry_run:
            np.save(cp, z_out)
        updated += 1
        print(f"{stem}: a={a:.5f}, b={b:.5f}, samples={int(valid.sum())}")

    print(f"done. updated={updated}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
