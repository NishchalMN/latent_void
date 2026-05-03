#!/usr/bin/env python3
"""Evaluate reconstruction gates for scene-local DiffSplat adaptation."""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.io import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic-dir", action="append", default=[])
    parser.add_argument("--teacher-targets", default="")
    parser.add_argument("--recon-train-status", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mse-threshold", type=float, default=0.08)
    parser.add_argument("--loss-threshold", type=float, default=0.08)
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _png_map(root):
    paths = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
    return {os.path.basename(path): path for path in sorted(paths)}


def _image(path):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return array


def _compare_dirs(diagnostic_dir):
    direct_root = os.path.join(diagnostic_dir, "direct_gs_grid")
    latent_root = os.path.join(diagnostic_dir, "latent_reconstruction")
    direct = _png_map(direct_root)
    latent = _png_map(latent_root)
    common = sorted(set(direct) & set(latent))
    comparisons = []
    for name in common:
        a = _image(direct[name])
        b = _image(latent[name])
        if a.shape != b.shape:
            continue
        diff = a - b
        comparisons.append({
            "image": name,
            "direct_path": direct[name],
            "latent_path": latent[name],
            "mse": float(np.mean(diff ** 2)),
            "mae": float(np.mean(np.abs(diff))),
        })
    mse_values = [item["mse"] for item in comparisons]
    return {
        "diagnostic_dir": diagnostic_dir,
        "num_direct_images": len(direct),
        "num_latent_images": len(latent),
        "num_compared": len(comparisons),
        "mean_mse": float(np.mean(mse_values)) if mse_values else None,
        "max_mse": float(np.max(mse_values)) if mse_values else None,
        "comparisons": comparisons,
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    diagnostics = [_compare_dirs(path) for path in args.diagnostic_dir]
    train_status = _load_json(args.recon_train_status) if args.recon_train_status else {}
    teacher_targets = _load_json(args.teacher_targets) if args.teacher_targets else {}
    mse_values = [item["mean_mse"] for item in diagnostics if item.get("mean_mse") is not None]
    mean_mse = float(np.mean(mse_values)) if mse_values else None
    final_loss = train_status.get("final_loss")
    gates = {
        "has_teacher_targets": bool(teacher_targets.get("samples")),
        "has_diagnostic_pairs": bool(mse_values),
        "gsvae_close_to_direct": bool(mse_values and mean_mse <= float(args.mse_threshold)),
        "adapter_loss_pass": bool(final_loss is not None and float(final_loss) <= float(args.loss_threshold)),
    }
    gates["overall_pass"] = gates["has_teacher_targets"] and (gates["gsvae_close_to_direct"] or gates["adapter_loss_pass"])
    report = {
        "ok": True,
        "output_dir": os.path.abspath(args.output_dir),
        "mse_threshold": float(args.mse_threshold),
        "loss_threshold": float(args.loss_threshold),
        "mean_diagnostic_mse": mean_mse,
        "final_recon_adapter_loss": final_loss,
        "gates": gates,
        "diagnostics": diagnostics,
        "teacher_targets": {
            "path": args.teacher_targets,
            "num_samples": teacher_targets.get("num_samples", 0),
        },
        "recon_train_status": args.recon_train_status,
    }
    path = os.path.join(args.output_dir, "recon_gate_report.json")
    write_json(path, report)
    print(json.dumps({"ok": True, "report": path, "gates": gates, "mean_diagnostic_mse": mean_mse, "final_recon_adapter_loss": final_loss}, indent=2))
    return 0 if gates["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
