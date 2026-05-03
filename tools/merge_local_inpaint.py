#!/usr/bin/env python3
"""Merge decoded local inpainted Gaussians back into a full scene."""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.gaussians import load_gaussian_npz, save_gaussian_npz
from latent_void.io import ensure_dir, write_json


COUNT_KEYS = ("opacity", "opacities", "features", "means", "xyz", "positions", "scales", "rotations")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-gaussian-npz", required=True)
    parser.add_argument("--local-gaussian-npz", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--deletion-mask", default="")
    parser.add_argument("--local-keep-mask", default="")
    parser.add_argument("--min-local-opacity", type=float, default=0.0)
    parser.add_argument("--min-visible-views", type=int, default=0)
    parser.add_argument("--remove-deleted-full", action="store_true")
    parser.add_argument("--manifest-out", default="")
    return parser.parse_args()


def _gaussian_count(arrays):
    for key in COUNT_KEYS:
        if key in arrays and np.asarray(arrays[key]).ndim >= 1:
            return int(np.asarray(arrays[key]).shape[0])
    for value in arrays.values():
        array = np.asarray(value)
        if array.ndim >= 2:
            return int(array.shape[0])
    raise ValueError("could not infer Gaussian count")


def _load_mask(path, count, default_value):
    if not path:
        return np.full((count,), bool(default_value), dtype=bool)
    mask = np.load(path).astype(bool).reshape(-1)
    if mask.shape[0] != count:
        raise ValueError("mask count %d does not match Gaussian count %d" % (mask.shape[0], count))
    return mask


def _opacity_keep(arrays, count, threshold):
    if threshold <= 0.0:
        return np.ones((count,), dtype=bool)
    for key in ("opacity", "opacities"):
        if key in arrays:
            opacity = np.asarray(arrays[key]).reshape(count, -1).mean(axis=1)
            return opacity >= float(threshold)
    return np.ones((count,), dtype=bool)


def _visibility_keep(arrays, count, min_visible):
    if min_visible <= 0 or "visibility" not in arrays:
        return np.ones((count,), dtype=bool)
    visibility = np.asarray(arrays["visibility"]).astype(bool)
    if visibility.ndim == 2 and visibility.shape[1] == count:
        return visibility.sum(axis=0) >= int(min_visible)
    if visibility.ndim == 1 and visibility.shape[0] == count:
        return visibility.astype(bool)
    return np.ones((count,), dtype=bool)


def _finite_keep(arrays, count):
    keep = np.ones((count,), dtype=bool)
    for key in ("means", "xyz", "positions", "features", "opacity", "opacities"):
        if key in arrays and np.asarray(arrays[key]).shape[0] == count:
            keep &= np.isfinite(np.asarray(arrays[key]).reshape(count, -1)).all(axis=1)
    return keep


def _filter_arrays(arrays, keep):
    count = keep.shape[0]
    result = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == count:
            result[key] = array[keep]
        elif key == "uvs" and array.ndim == 3 and array.shape[1] == count:
            result[key] = array[:, keep, :]
        elif key == "visibility" and array.ndim == 2 and array.shape[1] == count:
            result[key] = array[..., keep]
        else:
            result[key] = array
    return result


def _merge_arrays(full, local):
    merged = {}
    for key, full_value in full.items():
        if key in local:
            full_array = np.asarray(full_value)
            local_array = np.asarray(local[key])
            if full_array.ndim >= 1 and local_array.ndim == full_array.ndim and full_array.shape[1:] == local_array.shape[1:]:
                merged[key] = np.concatenate([full_array, local_array], axis=0)
            else:
                merged[key] = full_array
        else:
            merged[key] = np.asarray(full_value)
    for key, local_value in local.items():
        if key not in merged:
            merged["local_" + key] = np.asarray(local_value)
    return merged


def main():
    args = parse_args()
    full = load_gaussian_npz(args.full_gaussian_npz)
    local = load_gaussian_npz(args.local_gaussian_npz)
    full_count = _gaussian_count(full)
    local_count = _gaussian_count(local)
    delete_mask = _load_mask(args.deletion_mask, full_count, False)
    full_keep = ~delete_mask if args.remove_deleted_full else np.ones((full_count,), dtype=bool)
    local_keep = _load_mask(args.local_keep_mask, local_count, True)
    local_keep &= _opacity_keep(local, local_count, args.min_local_opacity)
    local_keep &= _visibility_keep(local, local_count, args.min_visible_views)
    local_keep &= _finite_keep(local, local_count)
    filtered_full = _filter_arrays(full, full_keep)
    filtered_local = _filter_arrays(local, local_keep)
    merged = _merge_arrays(filtered_full, filtered_local)
    ensure_dir(os.path.dirname(args.output_npz) or ".")
    save_gaussian_npz(args.output_npz, merged)
    manifest = {
        "ok": True,
        "full_gaussian_npz": args.full_gaussian_npz,
        "local_gaussian_npz": args.local_gaussian_npz,
        "output_npz": args.output_npz,
        "full_count": full_count,
        "local_count": local_count,
        "deleted_full_count": int(delete_mask.sum()),
        "kept_full_count": int(full_keep.sum()),
        "kept_local_count": int(local_keep.sum()),
        "remove_deleted_full": bool(args.remove_deleted_full),
        "min_local_opacity": float(args.min_local_opacity),
        "min_visible_views": int(args.min_visible_views),
        "merged_keys": sorted(merged.keys()),
    }
    manifest_out = args.manifest_out or os.path.splitext(args.output_npz)[0] + "_merge_manifest.json"
    write_json(manifest_out, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
