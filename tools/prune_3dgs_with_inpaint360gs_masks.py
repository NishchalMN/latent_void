"""Prune a 3DGS PLY using Inpaint360GS object masks projected into cameras."""

import argparse
import json
import os
import shutil
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpaint360gs-root", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--input-model-path", required=True)
    parser.add_argument("--output-model-path", required=True)
    parser.add_argument("--sam3-results-json", default="", help="Optional SAM3 results JSON to use instead of source unseen_mask/test views.")
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--max-views", type=int, default=16)
    parser.add_argument("--min-mask-views", type=int, default=2)
    parser.add_argument("--min-mask-fraction", type=float, default=0.25)
    return parser.parse_args()


def _install_paths(root):
    root = os.path.abspath(root)
    for path in (root, os.path.join(root, "gaussian_splatting")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_colmap(source_path):
    from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text
    from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text

    sparse = os.path.join(source_path, "sparse", "0")
    try:
        extrinsics = read_extrinsics_binary(os.path.join(sparse, "images.bin"))
        intrinsics = read_intrinsics_binary(os.path.join(sparse, "cameras.bin"))
    except Exception:
        extrinsics = read_extrinsics_text(os.path.join(sparse, "images.txt"))
        intrinsics = read_intrinsics_text(os.path.join(sparse, "cameras.txt"))
    return extrinsics, intrinsics


def _camera_params(intrinsic):
    if intrinsic.model == "SIMPLE_PINHOLE":
        fx = fy = float(intrinsic.params[0])
        cx = float(intrinsic.params[1])
        cy = float(intrinsic.params[2])
    elif intrinsic.model == "PINHOLE":
        fx = float(intrinsic.params[0])
        fy = float(intrinsic.params[1])
        cx = float(intrinsic.params[2])
        cy = float(intrinsic.params[3])
    else:
        raise ValueError("unsupported camera model: %s" % intrinsic.model)
    return fx, fy, cx, cy


def _mask_path(source_path, image_name):
    stem = os.path.splitext(os.path.basename(image_name))[0]
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(source_path, "unseen_mask", stem + ext)
        if os.path.exists(path):
            return path
    return None


def _sam3_mask_rows(results_json, max_views):
    with open(results_json, "r") as handle:
        results = json.load(handle)
    rows = []
    for item in results.get("results", [])[:max_views]:
        image_path = item.get("image_path", "")
        mask_path = item.get("mask_path", "")
        if not image_path or not mask_path:
            continue
        rows.append((os.path.basename(image_path), mask_path))
    return rows


def _copy_metadata(input_model_path, output_model_path):
    os.makedirs(output_model_path, exist_ok=True)
    for name in ("cfg_args", "cameras.json", "exposure.json", "input.ply"):
        src = os.path.join(input_model_path, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_model_path, name))


def main():
    args = parse_args()
    _install_paths(args.inpaint360gs_root)
    from scene.colmap_loader import qvec2rotmat

    source_path = os.path.abspath(args.source_path)
    input_model_path = os.path.abspath(args.input_model_path)
    output_model_path = os.path.abspath(args.output_model_path)
    input_ply = os.path.join(input_model_path, "point_cloud", "iteration_%d" % args.iteration, "point_cloud.ply")
    output_ply = os.path.join(output_model_path, "point_cloud", "iteration_%d" % args.iteration, "point_cloud.ply")
    os.makedirs(os.path.dirname(output_ply), exist_ok=True)
    _copy_metadata(input_model_path, output_model_path)

    ply = PlyData.read(input_ply)
    vertex = ply["vertex"].data
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    mask_hits = np.zeros((xyz.shape[0],), dtype=np.int32)
    visible_hits = np.zeros((xyz.shape[0],), dtype=np.int32)
    view_rows = []

    extrinsics, intrinsics = _load_colmap(source_path)
    extrinsics_by_name = {os.path.basename(extr.name): extr for extr in extrinsics.values()}
    if args.sam3_results_json:
        view_specs = _sam3_mask_rows(args.sam3_results_json, args.max_views)
    else:
        test_extrinsics = sorted(
            [extr for extr in extrinsics.values() if os.path.basename(extr.name).startswith("test_")],
            key=lambda extr: extr.name,
        )[: args.max_views]
        view_specs = [(os.path.basename(extr.name), _mask_path(source_path, extr.name)) for extr in test_extrinsics]

    for image_name, mask_path in view_specs:
        extr = extrinsics_by_name.get(image_name)
        if extr is None or mask_path is None or not os.path.exists(mask_path):
            view_rows.append({"image_name": image_name, "mask": mask_path, "used": False})
            continue
        if mask_path.endswith(".npy"):
            mask = np.load(mask_path).astype(bool)
        else:
            mask = np.array(Image.open(mask_path).convert("L")) > 127
        mask_h, mask_w = mask.shape[:2]
        intrinsic = intrinsics[extr.camera_id]
        fx, fy, cx, cy = _camera_params(intrinsic)
        scale_x = mask_w / float(intrinsic.width)
        scale_y = mask_h / float(intrinsic.height)

        world_to_cam = qvec2rotmat(extr.qvec)
        cam_xyz = xyz @ world_to_cam.T + np.asarray(extr.tvec, dtype=np.float64)[None, :]
        z = cam_xyz[:, 2]
        valid_z = z > 1e-6
        u = (fx * (cam_xyz[:, 0] / np.maximum(z, 1e-6)) + cx) * scale_x
        v = (fy * (cam_xyz[:, 1] / np.maximum(z, 1e-6)) + cy) * scale_y
        ui = np.rint(u).astype(np.int64)
        vi = np.rint(v).astype(np.int64)
        in_bounds = valid_z & (ui >= 0) & (ui < mask_w) & (vi >= 0) & (vi < mask_h)
        visible_hits[in_bounds] += 1
        in_mask = np.zeros_like(in_bounds)
        idx = np.where(in_bounds)[0]
        in_mask[idx] = mask[vi[idx], ui[idx]]
        mask_hits[in_mask] += 1
        view_rows.append({
            "image_name": image_name,
            "mask": mask_path,
            "used": True,
            "projected": int(in_bounds.sum()),
            "in_mask": int(in_mask.sum()),
        })

    mask_fraction = mask_hits / np.maximum(visible_hits, 1)
    remove = (mask_hits >= args.min_mask_views) & (mask_fraction >= args.min_mask_fraction)
    keep = ~remove
    PlyData([PlyElement.describe(vertex[keep], "vertex")], text=ply.text).write(output_ply)

    status = {
        "ok": True,
        "source_path": source_path,
        "input_model_path": input_model_path,
        "output_model_path": output_model_path,
        "input_ply": input_ply,
        "output_ply": output_ply,
        "iteration": args.iteration,
        "num_input_gaussians": int(len(vertex)),
        "num_removed_gaussians": int(remove.sum()),
        "num_kept_gaussians": int(keep.sum()),
        "min_mask_views": args.min_mask_views,
        "min_mask_fraction": args.min_mask_fraction,
        "views": view_rows,
    }
    status_path = os.path.join(output_model_path, "mask_prune_status.json")
    with open(status_path, "w") as handle:
        json.dump(status, handle, indent=2)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
