#!/usr/bin/env python3
"""Prepare an official GObjaverse example folder for the GSRecon adapter."""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.diffsplat_compat import patch_optional_imports, patch_transformers_compat
from latent_void.io import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gobjaverse-dir", required=True, help="Path to a campos_512_v4-style object directory.")
    parser.add_argument("--diffsplat-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-res", type=int, default=256)
    parser.add_argument("--max-views", type=int, default=40)
    parser.add_argument("--norm-radius", type=float, default=1.4)
    parser.add_argument("--fxfy", type=float, default=1422.222 / 1024.0)
    return parser.parse_args()


def _load_camera_from_json(path):
    with open(path, "r") as handle:
        data = json.load(handle)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = np.asarray(data["x"], dtype=np.float32)
    c2w[:3, 1] = np.asarray(data["y"], dtype=np.float32)
    c2w[:3, 2] = np.asarray(data["z"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(data["origin"], dtype=np.float32)
    return c2w


def _to_diffsplat_original_c2w(c2w):
    c2w = np.asarray(c2w, dtype=np.float32).copy()
    c2w[1] *= -1
    c2w[[1, 2]] = c2w[[2, 1]]
    c2w[:3, 1:3] *= -1
    return c2w


def _load_rgba_white(path):
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    rgb = rgba[..., :3]
    alpha = rgba[..., 3:4]
    return (rgb * alpha + (1.0 - alpha)).astype(np.float32), alpha[..., 0] > 0.5


def _load_normal_depth(exr_path):
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import cv2

    nd = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    if nd is None:
        raise RuntimeError("failed to read EXR normal/depth: %s" % exr_path)
    nd = nd.astype(np.float32)
    normal = nd[..., :3][..., ::-1]
    depth = nd[..., 3]
    return normal, depth


def _resize_chw(torch, tensor, size):
    import torch.nn.functional as F

    return F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def _save_array(path, array):
    ensure_dir(os.path.dirname(path))
    np.save(path, np.asarray(array, dtype=np.float32))
    return path


def main():
    args = parse_args()
    sys.path.insert(0, args.diffsplat_root)
    import torch
    from kiui.cam import orbit_camera, undo_orbit_camera
    patch_transformers_compat()
    patch_optional_imports()
    from src.utils.geo_util import normalize_normals, unproject_depth

    view_dirs = [path for path in sorted(os.listdir(args.gobjaverse_dir)) if os.path.isdir(os.path.join(args.gobjaverse_dir, path))]
    view_dirs = view_dirs[:int(args.max_views)]
    if len(view_dirs) < 4:
        raise RuntimeError("need at least 4 GObjaverse views, found %d" % len(view_dirs))

    rgbs, masks, normals, depths, original_c2ws, c2ws = [], [], [], [], [], []
    init_azi = None
    view_ids = []
    for view_id in view_dirs:
        view_dir = os.path.join(args.gobjaverse_dir, view_id)
        stem = os.path.basename(view_id)
        rgb, mask = _load_rgba_white(os.path.join(view_dir, stem + ".png"))
        normal, depth = _load_normal_depth(os.path.join(view_dir, stem + "_nd.exr"))
        normal = np.transpose(normal, (2, 0, 1))
        normal[0, ...] *= -1
        c2w = _to_diffsplat_original_c2w(_load_camera_from_json(os.path.join(view_dir, stem + ".json")))
        ele, azi, dis = undo_orbit_camera(c2w)
        if init_azi is None:
            init_azi = azi
        azi = (azi - init_azi) % 360.0
        ele_sign = ele >= 0
        ele = abs(ele) - 1e-8
        ele = ele * (1.0 if ele_sign else -1.0)
        new_c2w = orbit_camera(ele, azi, dis).astype(np.float32)
        rgbs.append(np.transpose(rgb, (2, 0, 1)))
        masks.append(mask[None, ...].astype(np.float32))
        normals.append(normal)
        depths.append(depth.astype(np.float32))
        original_c2ws.append(c2w)
        c2ws.append(new_c2w)
        view_ids.append(stem)

    rgb_t = torch.from_numpy(np.stack(rgbs, axis=0)).float()
    mask_t = torch.from_numpy(np.stack(masks, axis=0)).float()
    normal_t = torch.from_numpy(np.stack(normals, axis=0)).float()
    depth_t = torch.from_numpy(np.stack(depths, axis=0)).float()
    original_c2w_t = torch.from_numpy(np.stack(original_c2ws, axis=0)).float()
    c2w_t = torch.from_numpy(np.stack(c2ws, axis=0)).float()
    normal_t = normalize_normals(normal_t.unsqueeze(0), original_c2w_t.unsqueeze(0), i=0).squeeze(0)
    normal_t = torch.einsum("rc,vchw->vrhw", c2w_t[0, :3, :3], normal_t)
    normal_t = normal_t * 0.5 + 0.5
    normal_t = normal_t * mask_t + (1.0 - mask_t)

    c2w_t[:, :3, 1:3] *= -1
    scale = float(args.norm_radius) / float(torch.norm(c2w_t[0, :3, 3]).item())
    c2w_t[:, :3, 3] *= scale
    fxfycxcy = torch.tensor([args.fxfy, args.fxfy, 0.5, 0.5], dtype=torch.float32).repeat(len(view_ids), 1)
    coord_t = unproject_depth((depth_t * scale).unsqueeze(0), c2w_t.unsqueeze(0), fxfycxcy.unsqueeze(0)).squeeze(0)
    coord_raw_t = coord_t.clone()
    coord_t = coord_t * 0.5 + 0.5
    coord_t = coord_t * mask_t + (1.0 - mask_t)
    rgb_t = rgb_t * mask_t + (1.0 - mask_t)

    rgb_t = _resize_chw(torch, rgb_t, args.input_res)
    mask_t = _resize_chw(torch, mask_t, args.input_res)
    normal_t = _resize_chw(torch, normal_t, args.input_res)
    coord_t = _resize_chw(torch, coord_t, args.input_res)
    coord_raw_t = _resize_chw(torch, coord_raw_t, args.input_res)
    depth_t = _resize_chw(torch, depth_t.unsqueeze(1), args.input_res)
    normal_t = normal_t * mask_t + (1.0 - mask_t)
    coord_t = coord_t * mask_t + (1.0 - mask_t)

    ensure_dir(args.output_dir)
    entries = []
    for idx, view_id in enumerate(view_ids):
        stem = "%04d_%s" % (idx, view_id)
        entry = {
            "view_id": view_id,
            "image_path": os.path.join(args.gobjaverse_dir, view_id, view_id + ".png"),
            "rgb_npy": _save_array(os.path.join(args.output_dir, stem + "_rgb.npy"), rgb_t[idx].numpy()),
            "normal_npy": _save_array(os.path.join(args.output_dir, stem + "_normal.npy"), np.clip(normal_t[idx].numpy(), 0.0, 1.0)),
            "coord_npy": _save_array(os.path.join(args.output_dir, stem + "_coord.npy"), np.clip(coord_t[idx].numpy(), 0.0, 1.0)),
            "coord_raw_npy": _save_array(os.path.join(args.output_dir, stem + "_coord_raw.npy"), coord_raw_t[idx].numpy()),
            "depth_npy": _save_array(os.path.join(args.output_dir, stem + "_depth.npy"), depth_t[idx].numpy()),
            "camera": {"c2w": c2w_t[idx].numpy().astype(float).tolist()},
            "source_camera": {"c2w": original_c2w_t[idx].numpy().astype(float).tolist()},
            "scaled_intrinsics": {
                "fx": float(args.fxfy * args.input_res),
                "fy": float(args.fxfy * args.input_res),
                "cx": float(0.5 * args.input_res),
                "cy": float(0.5 * args.input_res),
                "fxfycxcy": [float(args.fxfy * args.input_res), float(args.fxfy * args.input_res), float(0.5 * args.input_res), float(0.5 * args.input_res)],
                "fxfycxcy_normalized": [float(args.fxfy), float(args.fxfy), 0.5, 0.5],
            },
        }
        entries.append(entry)

    manifest = {
        "ok": True,
        "source": "gobjaverse_example",
        "source_dir": os.path.abspath(args.gobjaverse_dir),
        "output_dir": os.path.abspath(args.output_dir),
        "input_res": int(args.input_res),
        "num_views": len(entries),
        "camera_normalization": {
            "enabled": True,
            "mode": "diffsplat_gobjaverse_loader",
            "norm_radius": float(args.norm_radius),
            "scale": scale,
            "reference_view": view_ids[0],
        },
        "coord_normalization": {"mode": "diffsplat", "encoding": "coord * 0.5 + 0.5, white outside alpha mask"},
        "preprocessing": {
            "profile": "official_gobjaverse_example",
            "contract": [
                "Mirrors DiffSplat GObjaverse loader camera conversion and fixed intrinsics",
                "Uses EXR normal/depth maps from the official render_data_examples archive",
                "Composites RGB, normals, and coordinates to white outside the alpha mask",
            ],
        },
        "views": entries,
    }
    path = os.path.join(args.output_dir, "geometry_manifest.json")
    write_json(path, manifest)
    print(json.dumps({"ok": True, "manifest": path, "num_views": len(entries), "input_view_ids": view_ids[:4]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
