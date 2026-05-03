#!/usr/bin/env python3
"""Generate self-supervised masked latent samples from local patch artifacts."""

import argparse
import json
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.io import ensure_dir, write_json
from latent_void.latent import expand_mask_to_latent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--patch-manifest", default="")
    parser.add_argument("--gaussian-npz", default="")
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mask-mode", choices=["mixed", "patch_mask", "ellipse", "rectangle"], default="mixed")
    parser.add_argument("--min-radius-frac", type=float, default=0.08)
    parser.add_argument("--max-radius-frac", type=float, default=0.28)
    parser.add_argument("--context-fill", choices=["zero", "mean"], default="mean")
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _latent_spatial_shape(latent):
    if latent.ndim < 3:
        raise ValueError("latent must have at least 3 dimensions")
    return tuple(latent.shape[-2:])


def _mask_batch_shape(latent):
    if latent.ndim == 3:
        return ()
    return tuple(latent.shape[:-3])


def _resize_mask(mask, size):
    from PIL import Image

    mask = np.asarray(mask).astype(np.uint8)
    image = Image.fromarray(mask * 255)
    return np.asarray(image.resize((size[1], size[0]), Image.NEAREST)) > 0


def _patch_masks(manifest_path, latent):
    if not manifest_path:
        return []
    manifest = _load_json(manifest_path)
    height, width = _latent_spatial_shape(latent)
    masks = []
    for view in manifest.get("views", []):
        path = view.get("mask_npy")
        if path and os.path.exists(path):
            masks.append(_resize_mask(np.load(path).astype(bool), (height, width)))
    return masks


def _ellipse_mask(rng, shape, min_radius_frac, max_radius_frac):
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    cx = rng.uniform(0.25 * width, 0.75 * width)
    cy = rng.uniform(0.25 * height, 0.75 * height)
    rx = rng.uniform(min_radius_frac * width, max_radius_frac * width)
    ry = rng.uniform(min_radius_frac * height, max_radius_frac * height)
    angle = rng.uniform(0.0, np.pi)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x = xx - cx
    y = yy - cy
    xr = x * cos_a + y * sin_a
    yr = -x * sin_a + y * cos_a
    return ((xr / max(rx, 1e-6)) ** 2 + (yr / max(ry, 1e-6)) ** 2) <= 1.0


def _rectangle_mask(rng, shape, min_radius_frac, max_radius_frac):
    height, width = shape
    box_w = int(rng.uniform(min_radius_frac, max_radius_frac) * width * 2.0)
    box_h = int(rng.uniform(min_radius_frac, max_radius_frac) * height * 2.0)
    box_w = max(1, min(width, box_w))
    box_h = max(1, min(height, box_h))
    x0 = int(rng.integers(0, max(1, width - box_w + 1)))
    y0 = int(rng.integers(0, max(1, height - box_h + 1)))
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y0 + box_h, x0:x0 + box_w] = True
    return mask


def _choose_mask(rng, args, latent, patch_masks):
    shape = _latent_spatial_shape(latent)
    mode = args.mask_mode
    if mode == "mixed":
        choices = ["ellipse", "rectangle"]
        if patch_masks:
            choices.append("patch_mask")
        mode = str(rng.choice(choices))
    if mode == "patch_mask" and patch_masks:
        return np.array(patch_masks[int(rng.integers(0, len(patch_masks)))], copy=True), mode
    if mode == "rectangle":
        return _rectangle_mask(rng, shape, args.min_radius_frac, args.max_radius_frac), mode
    return _ellipse_mask(rng, shape, args.min_radius_frac, args.max_radius_frac), "ellipse"


def _mask_for_latent(base_mask, latent):
    batch_shape = _mask_batch_shape(latent)
    if not batch_shape:
        return base_mask.astype(bool)
    return np.broadcast_to(base_mask.astype(bool), batch_shape + base_mask.shape).copy()


def _masked_latent(latent, mask, context_fill):
    expanded = expand_mask_to_latent(mask, latent)
    output = np.array(latent, copy=True)
    channels_first = np.moveaxis(output, -3, 0)
    source_channels = np.moveaxis(latent, -3, 0)
    for channel_idx in range(channels_first.shape[0]):
        channel = channels_first[channel_idx]
        source = source_channels[channel_idx]
        if context_fill == "mean" and (~expanded).any():
            fill = float(source[~expanded].mean())
        else:
            fill = 0.0
        channel[expanded] = fill
    return output


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    latent = np.load(args.latent_path).astype(np.float32)
    patch_masks = _patch_masks(args.patch_manifest, latent)
    ensure_dir(args.output_dir)
    samples = []
    for idx in range(int(args.num_samples)):
        base_mask, mode = _choose_mask(rng, args, latent, patch_masks)
        latent_mask = _mask_for_latent(base_mask, latent)
        masked_latent = _masked_latent(latent, latent_mask, args.context_fill)
        sample_dir = ensure_dir(os.path.join(args.output_dir, "samples", "%06d" % idx))
        source_path = os.path.join(sample_dir, "source_latent.npy")
        masked_path = os.path.join(sample_dir, "masked_latent.npy")
        mask_path = os.path.join(sample_dir, "latent_mask.npy")
        np.save(source_path, latent)
        np.save(masked_path, masked_latent.astype(np.float32))
        np.save(mask_path, latent_mask.astype(np.uint8))
        meta = {
            "sample_id": "%06d" % idx,
            "mask_mode": mode,
            "source_latent": source_path,
            "masked_latent": masked_path,
            "latent_mask": mask_path,
            "latent_shape": list(latent.shape),
            "mask_shape": list(latent_mask.shape),
            "masked_cells": int(latent_mask.sum()),
        }
        write_json(os.path.join(sample_dir, "metadata.json"), meta)
        samples.append(meta)
    if args.patch_manifest:
        shutil.copyfile(args.patch_manifest, os.path.join(args.output_dir, "source_patch_manifest.json"))
    summary = {
        "ok": True,
        "latent_path": args.latent_path,
        "patch_manifest": args.patch_manifest,
        "gaussian_npz": args.gaussian_npz,
        "num_samples": len(samples),
        "seed": args.seed,
        "mask_mode": args.mask_mode,
        "context_fill": args.context_fill,
        "samples": samples,
    }
    if args.gaussian_npz:
        summary["gaussian_npz_exists"] = os.path.exists(args.gaussian_npz)
    write_json(os.path.join(args.output_dir, "dataset_manifest.json"), summary)
    print(json.dumps({"ok": True, "dataset_manifest": os.path.join(args.output_dir, "dataset_manifest.json"), "num_samples": len(samples)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
