#!/usr/bin/env python3
"""Build a scene-local patch dataset manifest from prepared geometry runs."""

import argparse
import glob
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.io import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--run-glob", action="append", default=[])
    parser.add_argument("--geometry-manifest", action="append", default=[])
    parser.add_argument("--mask-dir", action="append", default=[])
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crop-scale", type=float, default=1.75)
    parser.add_argument("--canonical-mode", choices=["object_centered", "first_view"], default="object_centered")
    parser.add_argument("--canonical-camera-radius", type=float, default=1.4)
    parser.add_argument("--keep-background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _discover_runs(args):
    run_dirs = list(args.run_dir)
    for pattern in args.run_glob:
        run_dirs.extend(glob.glob(pattern))
    pairs = []
    for run_dir in sorted(set(run_dirs)):
        geometry_manifest = os.path.join(run_dir, "geometry", "geometry_manifest.json")
        mask_dir = os.path.join(run_dir, "masks")
        if os.path.exists(geometry_manifest) and os.path.isdir(mask_dir):
            pairs.append((geometry_manifest, mask_dir, os.path.basename(os.path.normpath(run_dir))))
    if args.geometry_manifest:
        if len(args.geometry_manifest) != len(args.mask_dir):
            raise RuntimeError("--geometry-manifest and --mask-dir counts must match")
        for idx, (geometry_manifest, mask_dir) in enumerate(zip(args.geometry_manifest, args.mask_dir)):
            pairs.append((geometry_manifest, mask_dir, "manual_%03d" % idx))
    if args.max_scenes > 0:
        pairs = pairs[:args.max_scenes]
    return pairs


def _run_patch_extractor(args, geometry_manifest, mask_dir, scene_id, output_dir):
    command = [
        sys.executable,
        "tools/extract_local_patch_manifest.py",
        "--geometry-manifest", geometry_manifest,
        "--mask-dir", mask_dir,
        "--output-dir", output_dir,
        "--crop-size", str(args.crop_size),
        "--crop-scale", str(args.crop_scale),
        "--canonicalize-3d",
        "--canonical-camera-radius", str(args.canonical_camera_radius),
        "--canonical-mode", args.canonical_mode,
    ]
    if args.keep_background:
        command.append("--keep-background")
    if args.dry_run:
        return {"dry_run": True, "command": command}
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("patch extraction failed for %s: %s" % (scene_id, result.stderr[-4000:]))
    return {"dry_run": False, "command": command, "stdout": result.stdout}


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    pairs = _discover_runs(args)
    if not pairs:
        raise RuntimeError("no geometry/mask pairs found")
    samples = []
    failures = []
    for idx, (geometry_manifest, mask_dir, scene_id) in enumerate(pairs):
        patch_dir = os.path.join(args.output_dir, "patches", "%04d_%s" % (idx, scene_id))
        try:
            command_result = _run_patch_extractor(args, geometry_manifest, mask_dir, scene_id, patch_dir)
            patch_manifest = os.path.join(patch_dir, "local_patch_manifest.json")
            patch_data = _load_json(patch_manifest) if os.path.exists(patch_manifest) else {}
            views = patch_data.get("views", [])
            samples.append({
                "sample_id": "%04d_%s" % (idx, scene_id),
                "scene_id": scene_id,
                "geometry_manifest": geometry_manifest,
                "mask_dir": mask_dir,
                "patch_manifest": patch_manifest,
                "patch_dir": patch_dir,
                "num_views": len(views),
                "input_view_ids": [view.get("view_id") for view in views[:4]],
                "heldout_view_ids": [view.get("view_id") for view in views[4:8]],
                "command": command_result.get("command"),
            })
        except Exception as exc:
            failures.append({"scene_id": scene_id, "geometry_manifest": geometry_manifest, "reason": str(exc)})
    manifest = {
        "ok": not failures,
        "output_dir": os.path.abspath(args.output_dir),
        "num_samples": len(samples),
        "num_failures": len(failures),
        "crop_size": int(args.crop_size),
        "crop_scale": float(args.crop_scale),
        "canonical_mode": args.canonical_mode,
        "samples": samples,
        "failures": failures,
    }
    path = os.path.join(args.output_dir, "scene_patch_dataset.json")
    write_json(path, manifest)
    print(json.dumps({"ok": manifest["ok"], "dataset_manifest": path, "num_samples": len(samples), "num_failures": len(failures)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
