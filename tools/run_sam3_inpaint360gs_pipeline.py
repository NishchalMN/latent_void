#!/usr/bin/env python3
"""Run Inpaint360GS pipeline using SAM3-generated masks (no raw_mask_sam stage).

This script is a compatibility-first bridge:
1) Generates `raw_sam/*.png` with SAM3 (`tools/generate_sam3_instance_masks.py`)
2) Runs stage 2b (reduce labels) + stages 3..9 using existing driver
3) Applies optional virtual-depth fixes
4) Resumes stages 10..11 (fusion, inpaint, eval)

It intentionally preserves existing folder contracts so downstream Inpaint360GS
scripts continue to work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image


def _run(cmd: list[str], cwd: str) -> None:
    short = " ".join(cmd[:7]) + (" ..." if len(cmd) > 7 else "")
    print(f"\n[RUN] {short}", flush=True)
    code = subprocess.run(cmd, cwd=cwd).returncode
    if code != 0:
        raise SystemExit(code)


def _root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Inpaint360GS with SAM3-generated masks replacing raw SAM stage."
    )
    p.add_argument("--scene", required=True, help="Scene name (e.g. car, cube, bag)")
    p.add_argument(
        "--prompt",
        default="",
        help='SAM3 text prompt for target object (default: same as --scene)',
    )
    p.add_argument(
        "--sam3-root",
        default="external/sam3",
        help="Path to SAM3 repo root",
    )
    p.add_argument(
        "--sam3-checkpoint",
        default="checkpoints/sam3",
        help="Local SAM3 checkpoint dir for run_sam3_multiview.py",
    )
    p.add_argument("--resolution", type=int, default=2)
    p.add_argument("--finetune-iters", type=int, default=12000)
    p.add_argument(
        "--checkpoint-video-iters",
        default="5000 8000",
        help="Space-separated checkpoint video iters; empty disables.",
    )
    p.add_argument(
        "--skip-fid-eval",
        action="store_true",
        default=True,
        help="Skip FID in eval (recommended on offline compute nodes).",
    )
    p.add_argument(
        "--run-depth-fix",
        action="store_true",
        help="Apply virtual depth alignment + planar projection between stage 9 and 10.",
    )
    p.add_argument("--ring-width", type=int, default=5)
    p.add_argument(
        "--max-classes",
        type=int,
        default=30,
        help="Max SAM classes after reduction (stage 2b).",
    )
    return p.parse_args()


def _build_train_manifest(image_dir: str, manifest_path: str) -> int:
    views = []
    for i, name in enumerate(sorted(os.listdir(image_dir))):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        views.append(
            {
                "view_id": f"train_{i:04d}",
                "image_path": os.path.join(image_dir, name),
            }
        )
    with open(manifest_path, "w") as f:
        json.dump({"views": views}, f, indent=2)
    return len(views)


def _sam3_npy_to_raw_sam_png(mask_npy_dir: str, raw_sam_dir: str) -> None:
    """Convert SAM3 npy outputs into raw_sam PNGs named by original image stems.

    run_sam3_multiview.py writes index-based files (0000.npy, ...). We must map
    them back to image-name stems using sam3_results.json; otherwise
    mask_associate.py won't find masks for camera image names.
    """
    os.makedirs(raw_sam_dir, exist_ok=True)
    # Always refresh to avoid stale files from prior runs (different view count / stems).
    for name in os.listdir(raw_sam_dir):
        if name.lower().endswith(".png"):
            os.remove(os.path.join(raw_sam_dir, name))

    results_json = os.path.join(mask_npy_dir, "sam3_results.json")
    if not os.path.isfile(results_json):
        raise SystemExit(f"Missing SAM3 results mapping file: {results_json}")
    with open(results_json, "r") as f:
        payload = json.load(f)

    wrote = 0
    for row in payload.get("results", []):
        image_path = row.get("image_path", "")
        mask_path = row.get("mask_path", "")
        if not image_path or not mask_path or not os.path.isfile(mask_path):
            continue
        stem = os.path.splitext(os.path.basename(image_path))[0]
        arr = np.load(mask_path)
        # Single-instance ID mask compatible with mask_associate.py:
        # 0 background, 1 target object.
        arr = (arr > 0).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(raw_sam_dir, stem + ".png"))
        wrote += 1
    print(f"[INFO] wrote {wrote} SAM3 masks into {raw_sam_dir}", flush=True)


def _assert_scene_ok(root: str, scene: str, context: str) -> None:
    """Fail if pipeline_summary marks any stage as non-ok for this scene."""
    summary = os.path.join(root, "output", "inpaint360", "pipeline_summary.json")
    if not os.path.isfile(summary):
        raise SystemExit(f"[{context}] missing pipeline summary: {summary}")
    with open(summary, "r") as f:
        data = json.load(f)
    st = data.get(scene) or {}
    stages = st.get("stages", {})
    bad = [k for k, v in stages.items() if v != "ok"]
    if bad:
        raise SystemExit(
            f"[{context}] scene {scene} has failed stages in pipeline_summary: {bad}"
        )


def main() -> int:
    args = parse_args()
    root = _root()
    inpaint_root = os.path.join(root, "external", "Inpaint360GS")
    scene = args.scene
    env = os.environ.copy()
    # Ensure Inpaint360GS local imports work for all subprocess stages.
    gs = os.path.join(inpaint_root, "gaussian_splatting")
    env["PYTHONPATH"] = os.pathsep.join([inpaint_root, gs, env.get("PYTHONPATH", "")]).strip(os.pathsep)

    # 0) Generate target masks with SAM3 wrapper, then convert to raw_sam PNGs.
    prompt = args.prompt.strip() or scene
    image_dir = os.path.join(root, "data", "inpaint360", scene, f"images_{args.resolution}")
    manifest = os.path.join(root, "output", "inpaint360", scene, "sam3_train_manifest.json")
    npy_dir = os.path.join(root, "output", "inpaint360", scene, "sam3_train_masks_npy")
    raw_sam_dir = os.path.join(root, "data", "inpaint360", scene, "raw_sam")
    n_views = _build_train_manifest(image_dir, manifest)
    print(f"[INFO] SAM3 manifest built with {n_views} views, prompt={prompt!r}", flush=True)

    _run(
        [
            sys.executable,
            os.path.join(root, "tools", "run_sam3_multiview.py"),
            "--sam3-root",
            os.path.join(root, args.sam3_root),
            "--checkpoint-path",
            os.path.join(root, args.sam3_checkpoint),
            "--manifest",
            manifest,
            "--prompt",
            prompt,
            "--output-dir",
            npy_dir,
            "--backend",
            "transformers",
            "--device",
            env.get("SAM3_DEVICE", "cuda"),
        ],
        cwd=root,
    )
    _sam3_npy_to_raw_sam_png(npy_dir, raw_sam_dir)

    # 1) Reduce segments (stage 2b equivalent).
    _run(
        [
            sys.executable,
            os.path.join(root, "tools", "reduce_sam_segments.py"),
            "--mask-dir",
            raw_sam_dir,
            "--max-classes",
            str(args.max_classes),
        ],
        cwd=root,
    )

    # 2) Run stages 3..9 using existing orchestrator.
    #    start-stage=3 skips stage 1 + stage 2a, keeping current 3DGS training.
    phase_a = [
        sys.executable,
        os.path.join(root, "tools", "run_inpaint360gs_full.py"),
        "--scenes",
        scene,
        "--resolution",
        str(args.resolution),
        "--start-stage",
        "3",
        "--stop-after-stage",
        "9",
        "--finetune-iterations",
        str(args.finetune_iters),
        "--data-root",
        os.path.join(root, "data", "inpaint360"),
        "--output-root",
        os.path.join(root, "output", "inpaint360"),
    ]
    if args.skip_fid_eval:
        phase_a.append("--skip-fid-eval")
    print(f"[INFO] PYTHONPATH={env['PYTHONPATH']}", flush=True)
    code = subprocess.run(phase_a, cwd=root, env=env).returncode
    if code != 0:
        raise SystemExit(code)
    _assert_scene_ok(root, scene, "phase_a")

    # 3) Optional depth fix bridge before fusion.
    if args.run_depth_fix:
        base = os.path.join(
            root,
            "output",
            "inpaint360",
            scene,
            "virtual",
            "ours_object_removal",
            "iteration_2000",
        )
        completed = os.path.join(base, "depth_completed")
        hole = os.path.join(base, "depth")
        mask = os.path.join(root, "data", "inpaint360", scene, "inpaint_2d_unseen_mask_virtual")

        _run(
            [
                sys.executable,
                os.path.join(root, "tools", "inpaint360_align_completed_depth.py"),
                "--completed-dir",
                completed,
                "--hole-dir",
                hole,
                "--mask-dir",
                mask,
                "--backup",
                "--ring-width",
                str(args.ring_width),
            ],
            cwd=root,
        )
        _run(
            [
                sys.executable,
                os.path.join(root, "tools", "inpaint360_project_completed_to_hole_plane.py"),
                "--completed-dir",
                completed,
                "--hole-dir",
                hole,
                "--mask-dir",
                mask,
                "--backup",
                "--ring-width",
                str(args.ring_width),
            ],
            cwd=root,
        )

    # 4) Resume stages 10..11.
    env["INPAINT360_CHECKPOINT_VIDEO_ITERS"] = args.checkpoint_video_iters
    phase_b = [
        sys.executable,
        os.path.join(root, "tools", "run_inpaint360gs_full.py"),
        "--scenes",
        scene,
        "--resolution",
        str(args.resolution),
        "--start-stage",
        "10",
        "--finetune-iterations",
        str(args.finetune_iters),
        "--data-root",
        os.path.join(root, "data", "inpaint360"),
        "--output-root",
        os.path.join(root, "output", "inpaint360"),
    ]
    if args.skip_fid_eval:
        phase_b.append("--skip-fid-eval")
    print(f"\n[INFO] INPAINT360_CHECKPOINT_VIDEO_ITERS={args.checkpoint_video_iters!r}", flush=True)
    code = subprocess.run(phase_b, cwd=root, env=env).returncode
    if code != 0:
        raise SystemExit(code)
    _assert_scene_ok(root, scene, "phase_b")
    print("\n[DONE] SAM3-driven pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
