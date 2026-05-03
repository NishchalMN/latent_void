#!/usr/bin/env python3
"""Export DiffSplat GSRecon/GSVAE outputs for a real scene geometry manifest."""

import argparse
import json
import os
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffsplat-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--gsvae-weights", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--geometry-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opt-type", default="gsvae_sdxl_fp16")
    parser.add_argument("--ckpt-iter", type=int, default=-1)
    parser.add_argument("--gsvae-ckpt-iter", type=int, default=-1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _write_status(output_dir, status, returncode):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "gsrecon_export_status.json"), "w") as handle:
        json.dump(status, handle, indent=2)
        handle.write("\n")
    print(json.dumps(status, indent=2), file=sys.stderr if returncode else sys.stdout)
    return returncode


def _checkpoint_dir(path):
    nested = os.path.join(path, "checkpoints")
    return nested if os.path.isdir(nested) else path


def _load_manifest(path):
    with open(path, "r") as handle:
        manifest = json.load(handle)
    if not manifest.get("views"):
        raise RuntimeError("geometry manifest has no views: %s" % path)
    return manifest


def _load_scene_tensors(manifest, torch, device, num_input_views):
    rgbs, normals, coords, masks, c2ws, intrinsics = [], [], [], [], [], []
    selected = manifest["views"][:num_input_views]
    for entry in selected:
        rgb = np.load(entry["rgb_npy"]).astype(np.float32)
        normal = np.load(entry["normal_npy"]).astype(np.float32)
        coord = np.load(entry["coord_npy"]).astype(np.float32)
        if rgb.shape[1:] != normal.shape[1:] or rgb.shape[1:] != coord.shape[1:]:
            raise RuntimeError("RGB/normal/coord shape mismatch for %s" % entry["view_id"])
        rgbs.append(rgb)
        normals.append(normal)
        coords.append(coord)
        masks.append(np.ones((1,) + rgb.shape[1:], dtype=np.float32))
        c2ws.append(np.asarray(entry["camera"]["c2w"], dtype=np.float32))
        intrinsics.append(np.asarray(entry["scaled_intrinsics"]["fxfycxcy_normalized"], dtype=np.float32))
    input_images = np.concatenate([
        np.stack(rgbs, axis=0),
        np.stack(normals, axis=0),
        np.stack(coords, axis=0),
    ], axis=1)
    all_c2ws = np.asarray([entry["camera"]["c2w"] for entry in manifest["views"]], dtype=np.float32)
    all_intrinsics = np.asarray([entry["scaled_intrinsics"]["fxfycxcy_normalized"] for entry in manifest["views"]], dtype=np.float32)
    return {
        "input_images": torch.from_numpy(input_images).unsqueeze(0).to(device),
        "masks": torch.from_numpy(np.stack(masks, axis=0)).unsqueeze(0).to(device),
        "input_C2W": torch.from_numpy(np.stack(c2ws, axis=0)).unsqueeze(0).to(device),
        "input_fxfycxcy": torch.from_numpy(np.stack(intrinsics, axis=0)).unsqueeze(0).to(device),
        "all_C2W_np": all_c2ws,
        "all_fxfycxcy_np": all_intrinsics,
        "input_view_ids": [entry["view_id"] for entry in selected],
        "all_view_ids": [entry["view_id"] for entry in manifest["views"]],
    }


def _patch_transformers_compat():
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return False
    patched = False
    for name in ("apply_chunking_to_forward", "prune_linear_layer", "Conv1D"):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
            patched = True
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        import torch

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - set(already_pruned_heads)
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        patched = True
    return patched


def _project_points(points, c2ws, fxfycxcy, height, width):
    uvs = np.zeros((len(c2ws), points.shape[0], 2), dtype=np.float32)
    visibility = np.zeros((len(c2ws), points.shape[0]), dtype=np.uint8)
    points_h = np.concatenate([points.astype(np.float32), np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    for view_idx, (c2w, intr) in enumerate(zip(c2ws, fxfycxcy)):
        w2c = np.linalg.inv(c2w)
        camera = points_h @ w2c.T
        z = camera[:, 2]
        fx = intr[0] * width
        fy = intr[1] * height
        cx = intr[2] * width
        cy = intr[3] * height
        valid_z = z > 1e-6
        x = np.zeros_like(z, dtype=np.float32)
        y = np.zeros_like(z, dtype=np.float32)
        x[valid_z] = camera[valid_z, 0] / z[valid_z] * fx + cx
        y[valid_z] = camera[valid_z, 1] / z[valid_z] * fy + cy
        uvs[view_idx, :, 0] = x
        uvs[view_idx, :, 1] = y
        visibility[view_idx] = (valid_z & (x >= 0) & (x < width) & (y >= 0) & (y < height)).astype(np.uint8)
    return uvs, visibility


def _run_model(args, manifest):
    sys.path.insert(0, args.diffsplat_root)
    import torch
    from einops import rearrange

    _patch_transformers_compat()
    from src.models import GSRecon, GSAutoencoderKL
    from src.options import opt_dict
    from src.utils import unproject_depth
    from src.utils import util

    opt = opt_dict[args.opt_type]
    opt.chunk_size = 1
    opt.render_type = "default"
    device = torch.device(args.device)
    gsrecon = GSRecon(opt).requires_grad_(False).eval().to(device)
    gsvae = GSAutoencoderKL(opt).requires_grad_(False).eval().to(device)
    gsrecon = util.load_ckpt(_checkpoint_dir(args.weights), args.ckpt_iter, None, gsrecon)
    gsvae = util.load_ckpt(_checkpoint_dir(args.gsvae_weights), args.gsvae_ckpt_iter, None, gsvae)
    tensors = _load_scene_tensors(manifest, torch, device, opt.num_input_views)
    input_images = tensors["input_images"].float()
    input_C2W = tensors["input_C2W"].float()
    input_fxfycxcy = tensors["input_fxfycxcy"].float()

    with torch.inference_mode():
        model_outputs = gsrecon.forward_gaussians(input_images, input_C2W, input_fxfycxcy)
        latents, gs_grid = gsvae.get_gslatents(
            gsrecon,
            input_images,
            input_C2W,
            input_fxfycxcy,
            return_gs=True,
        )
        latent_scaled = gsvae.scaling_factor * (latents - gsvae.shift_factor)
        depth = model_outputs["depth"] + torch.norm(input_C2W[:, :, :3, 3], p=2, dim=2, keepdim=True)[..., None, None]
        xyz = unproject_depth(depth.squeeze(2), input_C2W, input_fxfycxcy)
        rgb = model_outputs["rgb"] * 0.5 + 0.5
        scale_raw = model_outputs["scale"] * 0.5 + 0.5
        scale = opt.scale_min * scale_raw + opt.scale_max * (1.0 - scale_raw)
        opacity = model_outputs["opacity"] * 0.5 + 0.5
        rotation = model_outputs["rotation"]

    batch_size, num_input_views, _, height, width = model_outputs["rgb"].shape
    positions = rearrange(xyz, "b v c h w -> (b v h w) c").detach().cpu().numpy().astype(np.float32)
    uvs, visibility = _project_points(positions, tensors["all_C2W_np"], tensors["all_fxfycxcy_np"], height, width)
    latent_np = latent_scaled.detach().cpu().numpy().astype(np.float32)
    gs_grid_np = gs_grid.detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(args.output_dir, "latent.npy"), latent_np)
    np.save(os.path.join(args.output_dir, "gs_grid.npy"), gs_grid_np)
    np.savez_compressed(
        os.path.join(args.output_dir, "gaussians.npz"),
        positions=positions,
        rgb=rearrange(rgb, "b v c h w -> (b v h w) c").detach().cpu().numpy().astype(np.float32),
        scale=rearrange(scale, "b v c h w -> (b v h w) c").detach().cpu().numpy().astype(np.float32),
        rotation=rearrange(rotation, "b v c h w -> (b v h w) c").detach().cpu().numpy().astype(np.float32),
        opacity=rearrange(opacity, "b v c h w -> (b v h w) c").detach().cpu().numpy().astype(np.float32),
        uvs=uvs,
        visibility=visibility,
        gaussian_grid_shape=np.asarray([batch_size, num_input_views, height, width], dtype=np.int32),
        latent_shape=np.asarray(latent_np.shape, dtype=np.int32),
        gs_grid_shape=np.asarray(gs_grid_np.shape, dtype=np.int32),
        input_C2W=tensors["input_C2W"].detach().cpu().numpy().astype(np.float32),
        input_fxfycxcy=tensors["input_fxfycxcy"].detach().cpu().numpy().astype(np.float32),
        all_C2W=tensors["all_C2W_np"].astype(np.float32),
        all_fxfycxcy=tensors["all_fxfycxcy_np"].astype(np.float32),
        input_view_ids=np.asarray(tensors["input_view_ids"]),
        all_view_ids=np.asarray(tensors["all_view_ids"]),
    )
    return {
        "ok": True,
        "num_gaussians": int(positions.shape[0]),
        "num_projection_views": int(uvs.shape[0]),
        "latent_path": os.path.join(args.output_dir, "latent.npy"),
        "gaussian_npz": os.path.join(args.output_dir, "gaussians.npz"),
        "gs_grid": os.path.join(args.output_dir, "gs_grid.npy"),
        "input_view_ids": tensors["input_view_ids"],
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    preflight = {
        "ok": False,
        "args": vars(args),
        "expected_outputs": [
            "gaussians.npz with positions, uvs [V,N,2], visibility [V,N], and Gaussian attributes",
            "latent.npy with GSVAE splat latents",
            "gs_grid.npy with 12-channel Gaussian grids",
        ],
    }
    try:
        manifest = _load_manifest(args.geometry_manifest)
        missing = [
            path for entry in manifest["views"]
            for path in [entry.get("rgb_npy"), entry.get("normal_npy"), entry.get("coord_npy")]
            if not path or not os.path.exists(path)
        ]
        if missing:
            raise RuntimeError("missing geometry artifacts: " + ", ".join(missing[:6]))
        for label, path in [("diffsplat_root", args.diffsplat_root), ("weights", args.weights), ("gsvae_weights", args.gsvae_weights)]:
            if not os.path.exists(path):
                raise RuntimeError("%s does not exist: %s" % (label, path))
        preflight.update({"ok": True, "num_geometry_views": len(manifest["views"])})
        if args.preflight_only:
            return _write_status(args.output_dir, preflight, 0)
        result = _run_model(args, manifest)
        preflight.update(result)
        return _write_status(args.output_dir, preflight, 0)
    except Exception as exc:
        preflight.update({
            "ok": False,
            "reason": type(exc).__name__ + ": " + str(exc),
            "next_step": "Install DiffSplat GPU dependencies and verify geometry_manifest inputs, then rerun this exporter on an H100 job.",
        })
        return _write_status(args.output_dir, preflight, 2)


if __name__ == "__main__":
    raise SystemExit(main())
