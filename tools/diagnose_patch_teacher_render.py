"""Render non-neural teacher Gaussians from a local patch manifest.

This diagnostic separates camera/coordinate contract bugs from GSRecon learning
failures. It builds point Gaussians directly from patch RGB + mask + either the
encoded coordinate maps or the depth/camera unprojection, then renders those
Gaussians through DiffSplat's rasterizer.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diffsplat-root", default="external/DiffSplat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--source-views", type=int, default=4)
    parser.add_argument("--mask-threshold", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.006)
    parser.add_argument("--opacity", type=float, default=0.95)
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _load_chw(path):
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("expected CHW/HW array at %s, got %s" % (path, array.shape))
    return array


def _label(image, label):
    image = Image.fromarray(image).convert("RGB")
    pad = 24
    canvas = Image.new("RGB", (image.width, image.height + pad), "white")
    canvas.paste(image, (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), label, fill=(0, 0, 0))
    return canvas


def _to_rgb(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array[:3], 0.0, 1.0)
    return (np.transpose(array, (1, 2, 0)) * 255.0).astype(np.uint8)


def _to_gray_rgb(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array[0], 0.0, 1.0)
    image = (array * 255.0).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


def _write_sheet(path, rows):
    widths = [image.width for row in rows for image in row]
    heights = [image.height for row in rows for image in row]
    cell_w, cell_h = max(widths), max(heights)
    canvas = Image.new("RGB", (cell_w * len(rows[0]), cell_h * len(rows)), "white")
    for row_idx, row in enumerate(rows):
        for col_idx, image in enumerate(row):
            canvas.paste(image, (col_idx * cell_w, row_idx * cell_h))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def _stack_views(views, key, count):
    arrays = [_load_chw(view[key]) for view in views[:count]]
    return np.stack(arrays, axis=0)[None]


def _stack_optional_views(views, key, count):
    if not all(key in view and os.path.exists(view[key]) for view in views[:count]):
        return None
    return _stack_views(views, key, count)


def _stack_cameras(views, count):
    c2w = [np.asarray(view["camera"]["c2w"], dtype=np.float32) for view in views[:count]]
    intr = [
        np.asarray(view["scaled_intrinsics"]["fxfycxcy_normalized"], dtype=np.float32)
        for view in views[:count]
    ]
    return np.stack(c2w, axis=0)[None], np.stack(intr, axis=0)[None]


def _make_pc(torch, GaussianModel, xyz_bvchw, rgb_bvchw, mask_bvchw, scale, opacity):
    xyz = xyz_bvchw[0].permute(0, 2, 3, 1).reshape(-1, 3)
    rgb = rgb_bvchw[0].permute(0, 2, 3, 1).reshape(-1, 3)
    mask = mask_bvchw[0].permute(0, 2, 3, 1).reshape(-1, 1)
    keep = mask[:, 0] > 0.1
    xyz = xyz[keep]
    rgb = rgb[keep].clamp(0.0, 1.0)
    alpha = torch.full((xyz.shape[0], 1), float(opacity), dtype=torch.float32, device=xyz.device)
    scales = torch.full((xyz.shape[0], 3), float(scale), dtype=torch.float32, device=xyz.device)
    rotations = torch.zeros((xyz.shape[0], 4), dtype=torch.float32, device=xyz.device)
    rotations[:, 0] = 1.0
    return GaussianModel().set_data(xyz, rgb, scales, rotations, alpha)


def _render_mode(torch, render, pc, c2w, intr, height, width):
    images, alphas = [], []
    for view_idx in range(c2w.shape[1]):
        result = render(pc, height, width, c2w[0, view_idx], intr[0, view_idx], bg_color=(1.0, 1.0, 1.0))
        images.append(result["image"])
        alphas.append(result["alpha"])
    return torch.stack(images, dim=0)[None], torch.stack(alphas, dim=0)[None]


def _metrics(torch, pred, alpha, target, mask):
    return {
        "image_mse": float(torch.nn.functional.mse_loss(pred, target).detach().cpu()),
        "alpha_mse": float(torch.nn.functional.mse_loss(alpha, mask).detach().cpu()),
        "render_alpha_mean": float(alpha.mean().detach().cpu()),
        "target_alpha_mean": float(mask.mean().detach().cpu()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.abspath(args.diffsplat_root))
    from latent_void.diffsplat_compat import (
        patch_gaussian_rasterizer_compat,
        patch_optional_imports,
        patch_transformers_compat,
    )

    patch_transformers_compat()
    patch_optional_imports()
    patch_gaussian_rasterizer_compat()

    import torch
    from src.models.gs_render.gs_util import GaussianModel, render
    from src.utils.geo_util import unproject_depth

    manifest = _load_json(args.patch_manifest)
    views = manifest.get("views", [])[: int(args.max_views)]
    if len(views) < 1:
        raise RuntimeError("patch manifest has no views: %s" % args.patch_manifest)
    source_count = min(int(args.source_views), len(views))
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    images_np = _stack_views(views, "rgb_npy", len(views))[:, :, :3]
    masks_np = _stack_views(views, "mask_npy", len(views))[:, :, :1]
    coords_np = _stack_views(views, "coord_npy", len(views))[:, :, :3]
    coords_raw_np = _stack_optional_views(views, "coord_raw_npy", len(views))
    if coords_raw_np is not None:
        coords_raw_np = coords_raw_np[:, :, :3]
    depths_np = _stack_views(views, "depth_npy", len(views))[:, :, :1]
    masks_np = np.where(masks_np > 1.0, masks_np / 255.0, masks_np)
    c2w_np, intr_np = _stack_cameras(views, len(views))

    images = torch.from_numpy(images_np).to(device)
    masks = torch.from_numpy(masks_np).to(device)
    coords = torch.from_numpy(coords_np).to(device)
    coords_raw = torch.from_numpy(coords_raw_np).to(device) if coords_raw_np is not None else None
    depths = torch.from_numpy(depths_np).to(device)
    c2w = torch.from_numpy(c2w_np).to(device)
    intr = torch.from_numpy(intr_np).to(device)
    height, width = int(images.shape[-2]), int(images.shape[-1])

    source_images = images[:, :source_count]
    source_masks = masks[:, :source_count]
    source_coords = coords[:, :source_count] * 2.0 - 1.0
    source_depth_xyz = unproject_depth(depths[:, :source_count, 0], c2w[:, :source_count], intr[:, :source_count])

    modes = {
        "coord_teacher": source_coords,
        "depth_teacher": source_depth_xyz,
    }
    if coords_raw is not None:
        modes["coord_raw_teacher"] = coords_raw[:, :source_count]
    status = {
        "ok": True,
        "patch_manifest": args.patch_manifest,
        "num_views": len(views),
        "source_views": source_count,
        "height": height,
        "width": width,
        "scale": args.scale,
        "opacity": args.opacity,
        "modes": {},
    }

    for name, source_xyz in modes.items():
        pc = _make_pc(torch, GaussianModel, source_xyz, source_images, source_masks, args.scale, args.opacity)
        pred, alpha = _render_mode(torch, render, pc, c2w, intr, height, width)
        metrics = _metrics(torch, pred, alpha, images, masks)
        metrics["num_gaussians"] = int(pc.xyz.shape[0])
        status["modes"][name] = metrics

        pred_np = pred.detach().cpu().numpy()[0]
        alpha_np = alpha.detach().cpu().numpy()[0]
        images_np_local = images.detach().cpu().numpy()[0]
        masks_np_local = masks.detach().cpu().numpy()[0]
        rows = []
        for view_idx in range(min(4, len(views))):
            diff = np.abs(pred_np[view_idx] - images_np_local[view_idx])
            rows.append(
                [
                    _label(_to_rgb(pred_np[view_idx]), "%s render %d" % (name, view_idx)),
                    _label(_to_rgb(images_np_local[view_idx]), "target %d" % view_idx),
                    _label(_to_gray_rgb(alpha_np[view_idx]), "alpha %d" % view_idx),
                    _label(_to_rgb(diff), "abs diff %d" % view_idx),
                ]
            )
        sheet_path = os.path.join(args.output_dir, "%s_sheet.png" % name)
        _write_sheet(sheet_path, rows)
        metrics["sheet"] = sheet_path

    status_path = os.path.join(args.output_dir, "teacher_render_status.json")
    with open(status_path, "w") as handle:
        json.dump(status, handle, indent=2)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
