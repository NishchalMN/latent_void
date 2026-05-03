#!/usr/bin/env python3
"""Extract local DiffSplat-style crop inputs around the target mask."""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.geometry import (
    apply_world_transform_to_cameras,
    apply_world_transform_to_points,
    encode_coordinate_maps,
    local_canonical_transform,
)
from latent_void.io import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-manifest", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crop-scale", type=float, default=1.75)
    parser.add_argument("--min-crop-size", type=int, default=32)
    parser.add_argument("--max-views", type=int, default=0)
    parser.add_argument("--skip-empty", action="store_true")
    parser.add_argument("--canonicalize-3d", dest="canonicalize_3d", action="store_true", default=True)
    parser.add_argument("--no-canonicalize-3d", dest="canonicalize_3d", action="store_false")
    parser.add_argument("--canonical-camera-radius", type=float, default=1.4)
    parser.add_argument("--canonical-reference-index", type=int, default=0)
    parser.add_argument("--canonical-min-points", type=int, default=128)
    parser.add_argument("--canonical-mode", choices=["first_view", "object_centered"], default="first_view")
    parser.add_argument("--white-background-outside-mask", dest="white_background_outside_mask", action="store_true", default=True)
    parser.add_argument("--keep-background", dest="white_background_outside_mask", action="store_false")
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _mask_paths(mask_dir, views):
    results_path = os.path.join(mask_dir, "sam3_results.json")
    if os.path.exists(results_path):
        results = _load_json(results_path).get("results", [])
        return [item.get("mask_path") for item in results]
    return [os.path.join(mask_dir, "%04d.npy" % idx) for idx in range(len(views))]


def _load_mask(path, shape):
    if not path or not os.path.exists(path):
        return np.zeros(shape, dtype=bool)
    mask = np.load(path).astype(bool)
    if mask.shape != shape:
        image = Image.fromarray(mask.astype(np.uint8) * 255)
        mask = np.asarray(image.resize((shape[1], shape[0]), Image.NEAREST)) > 0
    return mask


def _mask_crop_box(mask, crop_scale, min_crop_size):
    ys, xs = np.where(mask)
    height, width = mask.shape
    if ys.size == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0, float(min_crop_size)) * float(crop_scale)
    side = min(side, float(max(height, width)))
    x0 = int(round(cx - side * 0.5))
    y0 = int(round(cy - side * 0.5))
    x1 = int(round(cx + side * 0.5))
    y1 = int(round(cy + side * 0.5))
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [int(x0), int(y0), int(x1), int(y1)]


def _resize_channel(channel, size, resample):
    image = Image.fromarray(np.asarray(channel, dtype=np.float32), mode="F")
    return np.asarray(image.resize((size, size), resample), dtype=np.float32)


def _crop_resize_chw(array, box, size, nearest=False):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("expected CHW array, got %s" % (array.shape,))
    x0, y0, x1, y1 = box
    cropped = array[:, y0:y1, x0:x1]
    resample = Image.NEAREST if nearest else Image.BICUBIC
    resized = [_resize_channel(channel, size, resample) for channel in cropped]
    return np.stack(resized, axis=0).astype(np.float32)


def _save_array(path, array):
    ensure_dir(os.path.dirname(path))
    np.save(path, np.asarray(array, dtype=np.float32))
    return path


def _save_mask(path, mask):
    ensure_dir(os.path.dirname(path))
    np.save(path, np.asarray(mask, dtype=np.uint8))
    return path


def _chw_to_hwc(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("expected CHW array, got %s" % (array.shape,))
    return np.transpose(array, (1, 2, 0))


def _hwc_to_chw(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("expected HWC array, got %s" % (array.shape,))
    return np.transpose(array, (2, 0, 1)).astype(np.float32)


def _collect_mask_points(patches):
    points = []
    for item in patches:
        coord = _chw_to_hwc(item["coord_raw_crop"])
        mask = item["mask_crop"]
        valid = mask & np.isfinite(coord).all(axis=-1)
        if valid.any():
            points.append(coord[valid])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(points, axis=0).astype(np.float32)


def _adjust_intrinsics(entry, box, crop_size, height, width):
    intr = entry.get("scaled_intrinsics", {}).get("fxfycxcy_normalized")
    if not intr:
        raise RuntimeError("view is missing scaled_intrinsics.fxfycxcy_normalized")
    fx = float(intr[0]) * float(width)
    fy = float(intr[1]) * float(height)
    cx = float(intr[2]) * float(width)
    cy = float(intr[3]) * float(height)
    x0, y0, x1, y1 = box
    sx = float(crop_size) / max(float(x1 - x0), 1.0)
    sy = float(crop_size) / max(float(y1 - y0), 1.0)
    fx = fx * sx
    fy = fy * sy
    cx = (cx - float(x0)) * sx
    cy = (cy - float(y0)) * sy
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fxfycxcy": [fx, fy, cx, cy],
        "fxfycxcy_normalized": [
            fx / float(crop_size),
            fy / float(crop_size),
            cx / float(crop_size),
            cy / float(crop_size),
        ],
    }


def _apply_white_background(chw, mask):
    array = np.asarray(chw, dtype=np.float32).copy()
    array[:, ~mask] = 1.0
    return array


def main():
    args = parse_args()
    manifest = _load_json(args.geometry_manifest)
    views = manifest.get("views", [])
    if args.max_views > 0:
        views = views[:args.max_views]
    mask_paths = _mask_paths(args.mask_dir, views)
    ensure_dir(args.output_dir)
    patch_items = []
    skipped = []
    for idx, entry in enumerate(views):
        rgb = np.load(entry["rgb_npy"]).astype(np.float32)
        height, width = rgb.shape[1:]
        mask = _load_mask(mask_paths[idx] if idx < len(mask_paths) else "", (height, width))
        box = _mask_crop_box(mask, args.crop_scale, args.min_crop_size)
        if box is None:
            skipped.append({"view_id": entry.get("view_id"), "reason": "empty_mask"})
            if args.skip_empty:
                continue
            box = [0, 0, width, height]
        stem = "%04d_%s" % (idx, entry.get("view_id", "view"))
        rgb_crop = _crop_resize_chw(rgb, box, args.crop_size)
        normal_crop = _crop_resize_chw(np.load(entry["normal_npy"]), box, args.crop_size)
        coord_crop = _crop_resize_chw(np.load(entry["coord_npy"]), box, args.crop_size)
        coord_raw_path = entry.get("coord_raw_npy")
        if not coord_raw_path:
            coord_raw_path = entry.get("coord_npy")
        coord_raw_crop = _crop_resize_chw(np.load(coord_raw_path), box, args.crop_size)
        depth_crop = _crop_resize_chw(np.load(entry["depth_npy"]), box, args.crop_size)
        mask_crop = _crop_resize_chw(mask[None, ...].astype(np.float32), box, args.crop_size, nearest=True)[0] > 0.5
        patch_items.append({
            "idx": idx,
            "entry": entry,
            "stem": stem,
            "box": box,
            "height": height,
            "width": width,
            "rgb_crop": rgb_crop,
            "normal_crop": normal_crop,
            "coord_crop": coord_crop,
            "coord_raw_crop": coord_raw_crop,
            "depth_crop": depth_crop,
            "mask_crop": mask_crop,
        })

    canonicalization = {"enabled": False, "reason": "disabled by config"}
    transform = np.eye(4, dtype=np.float32)
    if args.canonicalize_3d and patch_items:
        reference_index = min(max(int(args.canonical_reference_index), 0), len(patch_items) - 1)
        points = _collect_mask_points(patch_items)
        if points.shape[0] >= int(args.canonical_min_points):
            transform, canonicalization = local_canonical_transform(
                points,
                patch_items[reference_index]["entry"]["camera"]["c2w"],
                camera_radius=args.canonical_camera_radius,
                mode=args.canonical_mode,
            )
            canonicalization["reference_index"] = reference_index
            canonicalization["num_mask_points"] = int(points.shape[0])
        else:
            canonicalization = {
                "enabled": False,
                "reason": "insufficient finite masked coordinate points",
                "num_mask_points": int(points.shape[0]),
                "min_points": int(args.canonical_min_points),
            }

    patch_views = []
    raw_coord_maps = []
    for item in patch_items:
        idx = item["idx"]
        entry = item["entry"]
        stem = item["stem"]
        box = item["box"]
        height = item["height"]
        width = item["width"]
        rgb_crop = item["rgb_crop"]
        normal_crop = item["normal_crop"]
        depth_crop = item["depth_crop"]
        mask_crop = item["mask_crop"]
        coord_raw_hwc = _chw_to_hwc(item["coord_raw_crop"])
        if canonicalization.get("enabled"):
            coord_raw_hwc = apply_world_transform_to_points(coord_raw_hwc, transform)
        raw_coord_maps.append(coord_raw_hwc)
        coord_crop = _hwc_to_chw(encode_coordinate_maps([coord_raw_hwc], mode="diffsplat")[0][0])
        if args.white_background_outside_mask:
            rgb_crop = _apply_white_background(rgb_crop, mask_crop)
            normal_crop = _apply_white_background(normal_crop, mask_crop)
            coord_crop = _apply_white_background(coord_crop, mask_crop)
        transformed_camera = apply_world_transform_to_cameras([entry["camera"]], transform)[0] if canonicalization.get("enabled") else entry["camera"]
        patch_entry = dict(entry)
        patch_entry.update({
            "source_view_id": entry.get("view_id"),
            "view_id": "patch_" + str(entry.get("view_id")),
            "crop_box_xyxy": box,
            "crop_size": args.crop_size,
            "rgb_npy": _save_array(os.path.join(args.output_dir, stem + "_rgb.npy"), rgb_crop),
            "normal_npy": _save_array(os.path.join(args.output_dir, stem + "_normal.npy"), np.clip(normal_crop, 0.0, 1.0)),
            "coord_npy": _save_array(os.path.join(args.output_dir, stem + "_coord.npy"), np.clip(coord_crop, 0.0, 1.0)),
            "coord_raw_npy": _save_array(os.path.join(args.output_dir, stem + "_coord_raw.npy"), _hwc_to_chw(coord_raw_hwc)),
            "depth_npy": _save_array(os.path.join(args.output_dir, stem + "_depth.npy"), depth_crop),
            "mask_npy": _save_mask(os.path.join(args.output_dir, stem + "_mask.npy"), mask_crop),
            "camera": transformed_camera,
            "scaled_intrinsics": _adjust_intrinsics(entry, box, args.crop_size, height, width),
        })
        patch_views.append(patch_entry)
    _, coord_stats = encode_coordinate_maps(raw_coord_maps, mode="diffsplat")
    patch_manifest = {
        "ok": True,
        "source_geometry_manifest": os.path.abspath(args.geometry_manifest),
        "source_mask_dir": os.path.abspath(args.mask_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "input_res": args.crop_size,
        "crop_scale": args.crop_scale,
        "min_crop_size": args.min_crop_size,
        "camera_normalization": {
            "source": manifest.get("camera_normalization", {}),
            "local_canonicalization": canonicalization,
        },
        "coord_normalization": coord_stats,
        "preprocessing": {
            "profile": "local_patch_diffsplat",
            "white_background_outside_mask": args.white_background_outside_mask,
            "contract": [
                "RGB, normal, coord, and depth are cropped identically around the target mask",
                "RGB, normal, and encoded coord channels can be composited to white outside the object mask to match GObjaverse",
                "intrinsics are translated/scaled into crop coordinates",
                "when enabled, camera extrinsics and raw coordinate maps are transformed into a local mask-centered canonical frame",
            ],
        },
        "skipped": skipped,
        "views": patch_views,
    }
    path = os.path.join(args.output_dir, "local_patch_manifest.json")
    write_json(path, patch_manifest)
    print(json.dumps({"ok": True, "manifest": path, "num_views": len(patch_views), "skipped": len(skipped)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
