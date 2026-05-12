#!/usr/bin/env python3
"""Harden pruned-scene virtual RGB for LaMa by enforcing a clean hole inside SAM masks.

Gaussian pruning + rasterization often leaves **semi-transparent / smeared** boundaries
around removed objects (worse near frame borders and at grazing angles). SAM3 masks
are binary and correct, but the RGB hole looks muddy — LaMa then fails to fill.

This script composites each pruned virtual render with its SAM mask:
  - pixels inside the mask (inpaint region) -> solid fill (default: black)
  - pixels outside -> unchanged pruned render

Optional extensions (see guide — **blind dilate alone is often wrong**):

- **``--dark-fringe-luma``** — grow the hole only into **4-neighbors** that stay
  darker than a luma threshold for a few iterations. This targets the **smeared,
  near-black splat fringe** (often *outside* a tight SAM outline on border
  frames) without uniformly eating bright cobblestone like a fat dilate.
- **``--dilate``** — small uniform morphological dilate on the SAM mask only if
  you still need a thin ring.
- **``--diff-vs-full``** — union pixels where **pruned RGB** differs from
  **full-model** ``virtual/ours_2000/renders`` (same file name). Prefer
  **``--diff-edge-band``** (narrow ring outside SAM) over a huge
  **``--diff-near-dilate``** disk: a large disk marks distant pavement, cones,
  and alignment noise on “good” views (e.g. wide tails on frame 00027).
- **``--diff-max-pruned-luma``** — only count diff pixels where pruned luma is
  below L (smear is dark; bright cobblestone diffs are ignored).

Typical use **before** ``prepare_lama_data.py --inpaint2lama``::

    python tools/harden_pruned_virtual_renders.py --scene car --backup \\
        --diff-vs-full --diff-mean-thresh 14 --diff-edge-band 55 \\
        --diff-max-pruned-luma 155 \\
        --dark-fringe-luma 105 --dark-fringe-iters 20 \\
        --update-masks --backup-masks

Always pass ``--update-masks`` — otherwise expanded black regions stay **outside**
the SAM PNG and ``*_mask.png`` does not tell LaMa to repaint them.

See project_memory/INPAINT360GS_SAM3_DIRECT_GUIDE.md (hole hardening).
"""

from __future__ import annotations

import argparse
import os
import shutil

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True)
    p.add_argument(
        "--repo-root",
        default="",
        help="Repo root (default: parent of tools/)",
    )
    p.add_argument(
        "--iteration-tag",
        default="2000",
        help="virtual/ours_object_removal/iteration_<tag>/renders",
    )
    p.add_argument(
        "--fill",
        choices=("black", "gray128"),
        default="black",
        help="Fill value inside mask before LaMa",
    )
    p.add_argument(
        "--dilate",
        type=int,
        default=0,
        help="Uniform dilation radius on SAM mask (pixels); can eat real pavement — prefer dark-fringe",
    )
    p.add_argument(
        "--dark-fringe-luma",
        type=float,
        default=None,
        metavar="L",
        help="If set, grow hole into neighbors with luma < L (0–255). Catches splat smear outside SAM.",
    )
    p.add_argument(
        "--dark-fringe-iters",
        type=int,
        default=12,
        help="Max geodesic steps for dark-fringe growth (each step = 1px shell)",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="Copy original renders to renders_backup/ next to renders/",
    )
    p.add_argument(
        "--update-masks",
        action="store_true",
        help="Write expanded hole mask (SAM + dilate + dark-fringe) to "
        "data/inpaint360/<scene>/inpaint_2d_unseen_mask_virtual/ so LaMa *_mask.png matches RGB.",
    )
    p.add_argument(
        "--backup-masks",
        action="store_true",
        help="With --update-masks, copy mask dir to inpaint_2d_unseen_mask_virtual_sam_backup/ once",
    )
    p.add_argument(
        "--diff-vs-full",
        action="store_true",
        help="Union |pruned-full| mean RGB diff above thresh inside dilated SAM (needs ours_2000 renders)",
    )
    p.add_argument(
        "--diff-mean-thresh",
        type=float,
        default=14.0,
        help="Mean abs RGB diff (0-255) for diff-vs-full",
    )
    p.add_argument(
        "--diff-near-dilate",
        type=int,
        default=120,
        help="If diff-edge-band is 0: dilate SAM (px radius) for diff-vs-full neighborhood",
    )
    p.add_argument(
        "--diff-edge-band",
        type=int,
        default=0,
        help="If >0: diff-vs-full only on pixels outside SAM within this px "
        "distance of the hole (Euclidean); ignores diff-near-dilate. "
        "Use ~45–80 to kill far-field pavement/cone false positives.",
    )
    p.add_argument(
        "--diff-max-pruned-luma",
        type=float,
        default=None,
        metavar="L",
        help="If set, diff-vs-full only where pruned luma < L (targets dark smear)",
    )
    return p.parse_args()


def _root(args) -> str:
    if args.repo_root:
        return os.path.abspath(args.repo_root)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _dilate_mask_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    try:
        import cv2

        k = 2 * radius + 1
        kernel = np.ones((k, k), np.uint8)
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except ImportError:
        # Fallback: slow-ish scipy-less box dilate via max filter
        from PIL import ImageFilter

        u8 = (mask.astype(np.uint8)) * 255
        im = Image.fromarray(u8)
        for _ in range(radius):
            im = im.filter(ImageFilter.MaxFilter(size=3))
        return np.array(im) > 127


def _edt_outside_to_sam(sam_core: np.ndarray) -> np.ndarray:
    """Euclidean distance from each pixel to the nearest True pixel in ``sam_core``.

    Inside the mask the distance is 0. Requires OpenCV or SciPy.
    """
    try:
        import cv2

        src = np.where(sam_core, 0, 255).astype(np.uint8)
        return cv2.distanceTransform(src, cv2.DIST_L2, 5)
    except ImportError:
        pass
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt((~sam_core).astype(np.uint8))
    except ImportError as e:
        raise RuntimeError(
            "--diff-edge-band needs opencv-python or scipy (pip install scipy)"
        ) from e


def _luma(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _grow_dark_fringe(
    hole: np.ndarray, gray: np.ndarray, luma_thresh: float, max_iters: int
) -> np.ndarray:
    """Expand ``hole`` by OR-ing 4-neighbors that are darker than ``luma_thresh``."""
    try:
        import cv2

        dark = (gray < luma_thresh).astype(np.uint8) * 255
        h = hole.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(max_iters):
            dil = cv2.dilate(h, kernel, iterations=1)
            new = cv2.bitwise_or(h, cv2.bitwise_and(dil, dark))
            if int(new.sum()) == int(h.sum()):
                break
            h = new
        return h > 127
    except ImportError:
        h = hole.astype(bool)
        dark = gray < luma_thresh
        h_, w_ = h.shape
        for _ in range(max_iters):
            grown = h.copy()
            for y in range(h_):
                for x in range(w_):
                    if not h[y, x]:
                        continue
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h_ and 0 <= nx < w_ and dark[ny, nx]:
                            grown[ny, nx] = True
            if grown.sum() == h.sum():
                break
            h = grown
        return h


def main() -> int:
    args = parse_args()
    root = _root(args)
    virt = os.path.join(
        root,
        "output",
        "inpaint360",
        args.scene,
        "virtual",
        "ours_object_removal",
        f"iteration_{args.iteration_tag}",
    )
    rgb_dir = os.path.join(virt, "renders")
    mask_dir = os.path.join(root, "data", "inpaint360", args.scene, "inpaint_2d_unseen_mask_virtual")

    if not os.path.isdir(rgb_dir):
        raise SystemExit(f"Missing renders: {rgb_dir}")
    if not os.path.isdir(mask_dir):
        raise SystemExit(f"Missing masks: {mask_dir}")

    mask_backup = mask_dir + "_sam_backup"
    if args.update_masks and args.backup_masks and not os.path.isdir(mask_backup):
        shutil.copytree(mask_dir, mask_backup)
        print(f"[OK] SAM mask backup -> {mask_backup}")

    if args.backup:
        bak = rgb_dir + "_backup"
        if not os.path.isdir(bak):
            shutil.copytree(rgb_dir, bak)
            print(f"[OK] backup -> {bak}")

    fill_rgb = (0, 0, 0) if args.fill == "black" else (128, 128, 128)
    n_ok = 0
    n_skip = 0

    for name in sorted(os.listdir(rgb_dir)):
        if not name.lower().endswith(".png"):
            continue
        rp = os.path.join(rgb_dir, name)
        mp = os.path.join(mask_dir, name)
        if not os.path.isfile(mp):
            print(f"[SKIP] no mask for {name}")
            n_skip += 1
            continue

        rgb = np.array(Image.open(rp).convert("RGB"))
        m = np.array(Image.open(mp).convert("L"))
        if m.shape[:2] != rgb.shape[:2]:
            m = np.array(
                Image.fromarray(m).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)
            )
        sam_core = m > 127
        mask = sam_core.copy()
        mask = _dilate_mask_binary(mask, args.dilate)

        gray = _luma(rgb)
        if args.diff_vs_full:
            full_dir = os.path.join(
                root,
                "output",
                "inpaint360",
                args.scene,
                "virtual",
                "ours_2000",
                "renders",
            )
            full_path = os.path.join(full_dir, name)
            if os.path.isfile(full_path):
                full_rgb = np.array(Image.open(full_path).convert("RGB"))
                if full_rgb.shape[:2] != rgb.shape[:2]:
                    full_rgb = np.array(
                        Image.fromarray(full_rgb).resize(
                            (rgb.shape[1], rgb.shape[0]), Image.BILINEAR
                        )
                    )
                diff = np.mean(
                    np.abs(rgb.astype(np.float32) - full_rgb.astype(np.float32)),
                    axis=2,
                )
                diff_ok = diff >= float(args.diff_mean_thresh)
                if args.diff_max_pruned_luma is not None:
                    diff_ok = diff_ok & (gray < float(args.diff_max_pruned_luma))
                eb = int(args.diff_edge_band)
                if eb > 0:
                    dmap = _edt_outside_to_sam(sam_core)
                    diff_ok = diff_ok & (dmap > 0) & (dmap <= float(eb))
                else:
                    near = _dilate_mask_binary(sam_core, int(args.diff_near_dilate))
                    diff_ok = diff_ok & near
                mask = mask | diff_ok
            else:
                print(f"[WARN] diff-vs-full: missing {full_path}", flush=True)
        if args.dark_fringe_luma is not None:
            mask = _grow_dark_fringe(
                mask,
                gray,
                float(args.dark_fringe_luma),
                int(args.dark_fringe_iters),
            )

        out = rgb.copy()
        for c in range(3):
            out[:, :, c] = np.where(mask, fill_rgb[c], out[:, :, c])
        Image.fromarray(out.astype(np.uint8)).save(rp)
        if args.update_masks:
            mask_u8 = (mask.astype(np.uint8)) * 255
            Image.fromarray(mask_u8).save(mp)
        n_ok += 1

    print(f"[OK] hardened {n_ok} frames in {rgb_dir} (skipped {n_skip})")
    if args.dilate:
        print(f"     dilate={args.dilate}px")
    if args.dark_fringe_luma is not None:
        print(
            f"     dark-fringe luma<{args.dark_fringe_luma} iters={args.dark_fringe_iters}"
        )
    if args.update_masks:
        print(f"     updated masks -> {mask_dir}")
    if args.diff_vs_full:
        if int(args.diff_edge_band) > 0:
            extra = f"edge_band={args.diff_edge_band}px"
        else:
            extra = f"near_dilate={args.diff_near_dilate}px"
        if args.diff_max_pruned_luma is not None:
            extra += f" pruned_luma<{args.diff_max_pruned_luma}"
        print(f"     diff-vs-full mean>={args.diff_mean_thresh} {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
