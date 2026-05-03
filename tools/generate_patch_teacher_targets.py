#!/usr/bin/env python3
"""Create held-out teacher targets for scene-local reconstruction training."""

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--input-views", type=int, default=4)
    parser.add_argument("--heldout-views", type=int, default=4)
    parser.add_argument("--copy-arrays", action="store_true")
    parser.add_argument("--teacher-gaussian-npz", default="")
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _target_path(output_dir, sample_id, stem, suffix, source_path, copy_arrays):
    if not source_path:
        return ""
    if not copy_arrays:
        return source_path
    target_dir = ensure_dir(os.path.join(output_dir, "targets", sample_id))
    ext = os.path.splitext(source_path)[1] or ".npy"
    target = os.path.join(target_dir, "%s_%s%s" % (stem, suffix, ext))
    shutil.copyfile(source_path, target)
    return target


def _mask_stats(path):
    if not path or not os.path.exists(path):
        return {"exists": False}
    mask = np.load(path).astype(bool)
    return {
        "exists": True,
        "shape": list(mask.shape),
        "foreground_fraction": float(mask.mean()),
    }


def _make_sample_targets(args, sample):
    patch_manifest_path = sample.get("patch_manifest")
    patch_manifest = _load_json(patch_manifest_path)
    views = patch_manifest.get("views", [])
    input_views = views[:int(args.input_views)]
    heldout_views = views[int(args.input_views):int(args.input_views) + int(args.heldout_views)]
    targets = []
    for idx, view in enumerate(heldout_views):
        stem = "%02d_%s" % (idx, view.get("source_view_id") or view.get("view_id") or "view")
        target = {
            "target_id": stem,
            "source_view_id": view.get("source_view_id"),
            "camera": view.get("camera", {}),
            "scaled_intrinsics": view.get("scaled_intrinsics", {}),
            "rgb_npy": _target_path(args.output_dir, sample["sample_id"], stem, "rgb", view.get("rgb_npy"), args.copy_arrays),
            "alpha_npy": _target_path(args.output_dir, sample["sample_id"], stem, "alpha", view.get("mask_npy"), args.copy_arrays),
            "depth_npy": _target_path(args.output_dir, sample["sample_id"], stem, "depth", view.get("depth_npy"), args.copy_arrays),
            "normal_npy": _target_path(args.output_dir, sample["sample_id"], stem, "normal", view.get("normal_npy"), args.copy_arrays),
            "coord_npy": _target_path(args.output_dir, sample["sample_id"], stem, "coord", view.get("coord_npy"), args.copy_arrays),
            "mask_stats": _mask_stats(view.get("mask_npy")),
        }
        targets.append(target)
    return {
        "sample_id": sample["sample_id"],
        "scene_id": sample.get("scene_id"),
        "patch_manifest": patch_manifest_path,
        "num_input_views": len(input_views),
        "num_heldout_targets": len(targets),
        "input_views": [
            {
                "view_id": view.get("view_id"),
                "rgb_npy": view.get("rgb_npy"),
                "mask_npy": view.get("mask_npy"),
                "depth_npy": view.get("depth_npy"),
                "normal_npy": view.get("normal_npy"),
                "coord_npy": view.get("coord_npy"),
                "camera": view.get("camera", {}),
                "scaled_intrinsics": view.get("scaled_intrinsics", {}),
            }
            for view in input_views
        ],
        "targets": targets,
    }


def main():
    args = parse_args()
    dataset = _load_json(args.patch_dataset)
    ensure_dir(args.output_dir)
    samples = []
    failures = []
    for sample in dataset.get("samples", []):
        try:
            samples.append(_make_sample_targets(args, sample))
        except Exception as exc:
            failures.append({"sample_id": sample.get("sample_id"), "reason": str(exc)})
    manifest = {
        "ok": not failures,
        "patch_dataset": args.patch_dataset,
        "output_dir": os.path.abspath(args.output_dir),
        "num_samples": len(samples),
        "num_failures": len(failures),
        "input_views": int(args.input_views),
        "heldout_views": int(args.heldout_views),
        "teacher_gaussian_npz": args.teacher_gaussian_npz,
        "teacher_gaussian_npz_exists": bool(args.teacher_gaussian_npz and os.path.exists(args.teacher_gaussian_npz)),
        "samples": samples,
        "failures": failures,
    }
    path = os.path.join(args.output_dir, "teacher_targets.json")
    write_json(path, manifest)
    print(json.dumps({"ok": manifest["ok"], "teacher_targets": path, "num_samples": len(samples), "num_failures": len(failures)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
