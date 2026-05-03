#!/usr/bin/env python3
"""Decode GSVAE splat latents and render diagnostic views with DiffSplat."""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.diffsplat_compat import (
    patch_diffusers_model_paths,
    patch_gaussian_rasterizer_compat,
    patch_optional_imports,
    patch_transformers_compat,
    resolve_aux_model_paths,
    validate_aux_model_paths,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffsplat-root", required=True)
    parser.add_argument("--gsvae-weights", required=True)
    parser.add_argument("--gaussian-npz", required=True)
    parser.add_argument("--latent-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare-latent-path", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--opt-type", default="gsvae_sdxl_fp16")
    parser.add_argument("--sdxl-vae-path", default="")
    parser.add_argument("--tiny-vae-path", default="")
    parser.add_argument("--gsvae-ckpt-iter", type=int, default=-1)
    parser.add_argument("--max-render-views", type=int, default=8)
    parser.add_argument("--opacity-threshold", type=float, default=0.0)
    parser.add_argument("--latent-is-scaled", dest="latent_is_scaled", action="store_true", default=True)
    parser.add_argument("--latent-is-raw", dest="latent_is_scaled", action="store_false")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def _checkpoint_dir(path):
    nested = os.path.join(path, "checkpoints")
    return nested if os.path.isdir(nested) else path


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _load_camera_arrays(npz_path):
    data = np.load(npz_path)
    required = ["input_C2W", "input_fxfycxcy", "all_C2W", "all_fxfycxcy"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise RuntimeError("gaussian npz is missing render camera arrays: " + ", ".join(missing))
    input_c2w = data["input_C2W"].astype(np.float32)
    input_intr = data["input_fxfycxcy"].astype(np.float32)
    all_c2w = data["all_C2W"].astype(np.float32)
    all_intr = data["all_fxfycxcy"].astype(np.float32)
    if all_c2w.ndim == 3:
        all_c2w = all_c2w[None, ...]
    if all_intr.ndim == 2:
        all_intr = all_intr[None, ...]
    return input_c2w, input_intr, all_c2w, all_intr


def _save_png(path, chw):
    from PIL import Image

    array = np.asarray(chw, dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((array * 255.0).astype(np.uint8))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image.save(path)


def _load_model(args):
    sys.path.insert(0, args.diffsplat_root)
    import torch

    patch_transformers_compat()
    patch_optional_imports()
    patch_gaussian_rasterizer_compat()
    patch_diffusers_model_paths(args.sdxl_vae_path, args.tiny_vae_path)
    from src.models import GSRecon, GSAutoencoderKL
    from src.options import opt_dict
    from src.utils import util

    opt = opt_dict[args.opt_type]
    opt.chunk_size = 1
    opt.render_type = "default"
    device = torch.device(args.device)
    gsrecon = GSRecon(opt).requires_grad_(False).eval().to(device)
    gsvae = GSAutoencoderKL(opt).requires_grad_(False).eval().to(device)
    gsvae = util.load_ckpt(_checkpoint_dir(args.gsvae_weights), args.gsvae_ckpt_iter, None, gsvae)
    return torch, gsrecon, gsvae, device


def _render_latent(torch, gsrecon, gsvae, device, latent_path, input_c2w, input_intr, all_c2w, all_intr, args, label):
    latent = np.load(latent_path).astype(np.float32)
    latents = torch.from_numpy(latent).to(device)
    if args.latent_is_scaled:
        latents = latents / float(gsvae.scaling_factor) + float(gsvae.shift_factor)
    input_c2w_t = torch.from_numpy(input_c2w).to(device)
    input_intr_t = torch.from_numpy(input_intr).to(device)
    view_limit = all_c2w.shape[1] if args.max_render_views <= 0 else min(args.max_render_views, all_c2w.shape[1])
    all_c2w_t = torch.from_numpy(all_c2w[:, :view_limit]).to(device)
    all_intr_t = torch.from_numpy(all_intr[:, :view_limit]).to(device)
    with torch.inference_mode():
        outputs = gsvae.decode_and_render_gslatents(
            gsrecon,
            latents,
            input_c2w_t,
            input_intr_t,
            all_c2w_t,
            all_intr_t,
            opacity_threshold=args.opacity_threshold,
        )
    image = outputs["image"].detach().cpu().numpy()[0]
    alpha_tensor = outputs.get("alpha")
    depth_tensor = outputs["depth"] if "depth" in outputs else outputs.get("raw_depth")
    alpha = alpha_tensor.detach().cpu().numpy()[0] if alpha_tensor is not None else None
    depth = depth_tensor.detach().cpu().numpy()[0] if depth_tensor is not None else None
    render_dir = os.path.join(args.output_dir, label)
    for idx in range(image.shape[0]):
        _save_png(os.path.join(render_dir, "rgb_%04d.png" % idx), image[idx])
        if alpha is not None:
            _save_png(os.path.join(render_dir, "alpha_%04d.png" % idx), alpha[idx])
        if depth is not None:
            np.save(os.path.join(render_dir, "depth_%04d.npy" % idx), depth[idx].astype(np.float32))
    return {
        "label": label,
        "latent_path": latent_path,
        "num_views": int(image.shape[0]),
        "rgb_dir": render_dir,
        "output_keys": sorted(outputs.keys()),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    status = {
        "ok": False,
        "args": vars(args),
        "expected_outputs": ["before/rgb_*.png", "optional after/rgb_*.png", "optional alpha_*.png", "optional depth_*.npy"],
    }
    try:
        args.sdxl_vae_path, args.tiny_vae_path = resolve_aux_model_paths(args.sdxl_vae_path, args.tiny_vae_path)
        for label, path in [
            ("diffsplat_root", args.diffsplat_root),
            ("gsvae_weights", args.gsvae_weights),
            ("gaussian_npz", args.gaussian_npz),
            ("latent_path", args.latent_path),
        ]:
            if not os.path.exists(path):
                raise RuntimeError("%s does not exist: %s" % (label, path))
        validate_aux_model_paths(args.sdxl_vae_path, args.tiny_vae_path)
        if args.compare_latent_path and not os.path.exists(args.compare_latent_path):
            raise RuntimeError("compare_latent_path does not exist: %s" % args.compare_latent_path)
        input_c2w, input_intr, all_c2w, all_intr = _load_camera_arrays(args.gaussian_npz)
        status.update({
            "ok": True,
            "input_views": int(input_c2w.shape[1]),
            "render_views": int(all_c2w.shape[1] if args.max_render_views <= 0 else min(args.max_render_views, all_c2w.shape[1])),
        })
        if args.preflight_only:
            _write_json(os.path.join(args.output_dir, "render_status.json"), status)
            print(json.dumps(status, indent=2))
            return 0
        torch, gsrecon, gsvae, device = _load_model(args)
        renders = [
            _render_latent(torch, gsrecon, gsvae, device, args.latent_path, input_c2w, input_intr, all_c2w, all_intr, args, "before")
        ]
        if args.compare_latent_path:
            renders.append(_render_latent(
                torch,
                gsrecon,
                gsvae,
                device,
                args.compare_latent_path,
                input_c2w,
                input_intr,
                all_c2w,
                all_intr,
                args,
                "after",
            ))
        status["renders"] = renders
        _write_json(os.path.join(args.output_dir, "render_status.json"), status)
        print(json.dumps(status, indent=2))
        return 0
    except Exception as exc:
        status.update({"ok": False, "reason": type(exc).__name__ + ": " + str(exc)})
        _write_json(os.path.join(args.output_dir, "render_status.json"), status)
        print(json.dumps(status, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
