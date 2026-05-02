#!/usr/bin/env python3
"""Generate GSRecon geometry side channels for real RGB scene datasets."""

import argparse
import json
import os

import numpy as np
from PIL import Image

from latent_void.config import get_nested, load_config
from latent_void.datasets import Inpaint360GSDataset
from latent_void.geometry import normalize_coordinate_maps, unproject_depth_to_world
from latent_void.io import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-views", type=int, default=None)
    parser.add_argument("--input-res", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--depth-model", default=None)
    parser.add_argument("--normal-model", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--depth-scale", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_marigold(depth_model, normal_model, device):
    import torch
    from diffusers import MarigoldDepthPipeline, MarigoldNormalsPipeline

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    depth_pipe = MarigoldDepthPipeline.from_pretrained(depth_model, variant="fp16", torch_dtype=dtype).to(device)
    normal_pipe = MarigoldNormalsPipeline.from_pretrained(normal_model, variant="fp16", torch_dtype=dtype).to(device)
    depth_pipe.set_progress_bar_config(disable=True)
    normal_pipe.set_progress_bar_config(disable=True)
    return torch, depth_pipe, normal_pipe


def _prediction_array(prediction):
    if hasattr(prediction, "detach"):
        prediction = prediction.detach().float().cpu().numpy()
    array = np.asarray(prediction, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3 and array.shape[0] == 3:
        array = np.transpose(array, (1, 2, 0))
    return array.astype(np.float32)


def _save_array(path, array):
    ensure_dir(os.path.dirname(path))
    np.save(path, np.asarray(array, dtype=np.float32))
    return path


def _save_rgb(path, image):
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    return _save_array(path, np.transpose(rgb, (2, 0, 1)))


def _normal_to_chw_01(normal):
    normal = np.asarray(normal, dtype=np.float32)
    if normal.ndim != 3 or normal.shape[-1] != 3:
        raise ValueError("normal prediction must have shape [H, W, 3]")
    if normal.min() < -0.01:
        normal = normal * 0.5 + 0.5
    return np.transpose(np.clip(normal, 0.0, 1.0), (2, 0, 1)).astype(np.float32)


def main():
    args = parse_args()
    config = load_config(args.config)
    geometry_config = get_nested(config, "geometry", {})
    max_views = args.max_views if args.max_views is not None else int(get_nested(config, "pipeline.max_views", 4))
    input_res = args.input_res if args.input_res is not None else int(geometry_config.get("input_res", 256))
    device = args.device or geometry_config.get("device") or get_nested(config, "project.device", "cuda")
    depth_model = args.depth_model or geometry_config.get("depth_model", "prs-eth/marigold-depth-v1-1")
    normal_model = args.normal_model or geometry_config.get("normal_model", "prs-eth/marigold-normals-v1-1")
    num_inference_steps = args.num_inference_steps or int(geometry_config.get("num_inference_steps", 4))
    ensemble_size = args.ensemble_size or int(geometry_config.get("ensemble_size", 1))
    depth_scale = args.depth_scale if args.depth_scale is not None else float(geometry_config.get("depth_scale", 5.0))

    dataset = Inpaint360GSDataset(config)
    views = dataset.views(max_views=max_views)
    ensure_dir(args.output_dir)
    if args.dry_run:
        result = {
            "dry_run": True,
            "num_views": len(views),
            "output_dir": args.output_dir,
            "depth_model": depth_model,
            "normal_model": normal_model,
        }
        write_json(os.path.join(args.output_dir, "geometry_command.json"), result)
        print(json.dumps(result, indent=2))
        return 0

    torch, depth_pipe, normal_pipe = _load_marigold(depth_model, normal_model, device)
    raw_coords = []
    entries = []
    with torch.inference_mode():
        for idx, view in enumerate(views):
            image = Image.open(view.image_path).convert("RGB").resize((input_res, input_res), Image.BICUBIC)
            depth_output = depth_pipe(image, num_inference_steps=num_inference_steps, ensemble_size=ensemble_size)
            normal_output = normal_pipe(image, num_inference_steps=num_inference_steps, ensemble_size=ensemble_size)
            depth = _prediction_array(depth_output.prediction)
            normal = _prediction_array(normal_output.prediction)
            if depth.shape != (input_res, input_res):
                depth = np.asarray(Image.fromarray(depth).resize((input_res, input_res), Image.BILINEAR), dtype=np.float32)
            if normal.shape[:2] != (input_res, input_res):
                normal_img = Image.fromarray(((np.clip(normal, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8))
                normal = np.asarray(normal_img.resize((input_res, input_res), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0

            coords, intrinsics = unproject_depth_to_world(depth, view.camera, depth_scale=depth_scale)
            raw_coords.append(coords)
            stem = "%04d_%s" % (idx, view.view_id)
            entry = {
                "view_id": view.view_id,
                "image_path": view.image_path,
                "rgb_npy": _save_rgb(os.path.join(args.output_dir, stem + "_rgb.npy"), image),
                "depth_npy": _save_array(os.path.join(args.output_dir, stem + "_depth.npy"), depth[None, ...]),
                "normal_npy": _save_array(os.path.join(args.output_dir, stem + "_normal.npy"), _normal_to_chw_01(normal)),
                "coord_raw_npy": _save_array(os.path.join(args.output_dir, stem + "_coord_raw.npy"), np.transpose(coords, (2, 0, 1))),
                "camera": view.camera,
                "scaled_intrinsics": intrinsics,
            }
            entries.append(entry)

    normalized_coords, coord_stats = normalize_coordinate_maps(raw_coords)
    for entry, coords in zip(entries, normalized_coords):
        coord_path = entry["coord_raw_npy"].replace("_coord_raw.npy", "_coord.npy")
        entry["coord_npy"] = _save_array(coord_path, np.transpose(coords, (2, 0, 1)))

    manifest = {
        "ok": True,
        "config_path": os.path.abspath(args.config),
        "output_dir": os.path.abspath(args.output_dir),
        "dataset": {"type": get_nested(config, "dataset.type"), "scene": get_nested(config, "dataset.scene")},
        "input_res": input_res,
        "depth_model": depth_model,
        "normal_model": normal_model,
        "num_inference_steps": num_inference_steps,
        "ensemble_size": ensemble_size,
        "depth_scale": depth_scale,
        "coord_normalization": coord_stats,
        "views": entries,
    }
    write_json(os.path.join(args.output_dir, "geometry_manifest.json"), manifest)
    print(json.dumps({"ok": True, "manifest": os.path.join(args.output_dir, "geometry_manifest.json"), "num_views": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
