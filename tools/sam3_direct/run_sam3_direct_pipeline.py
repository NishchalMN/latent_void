#!/usr/bin/env python3
"""Direct SAM3-target pipeline (separate from legacy raw_sam flow).

This path bypasses legacy stages 2..6 (segment-all, association, distillation,
target-id selection) and uses a known text prompt target directly.

High-level:
1) SAM3 prompt masks on training images (for pruning stats + traceability)
2) Projection-based Gaussian pruning (separate pruned model folder)
3) Render virtual views from:
   - original model -> virtual/ours_2000
   - pruned model   -> virtual/ours_object_removal/iteration_2000
4) SAM3 prompt masks on virtual/ours_2000/renders -> inpaint_2d_unseen_mask_virtual
5) Existing LaMa prep/run/postprocess
6) Optional depth fixes
7) Existing stage10/11 (fusion + inpaint + eval)

Pruning uses multi-view voting on SAM masks (--min-mask-views / --min-mask-fraction).
Small objects (e.g. cones) often need stricter votes plus optional --prune-mask-erode-pixels
when SAM boundaries jitter between cameras.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _run(cmd: list[str], cwd: str, env: dict | None = None) -> None:
    short = " ".join(cmd[:8]) + (" ..." if len(cmd) > 8 else "")
    print(f"\n[RUN] {short}", flush=True)
    code = subprocess.run(cmd, cwd=cwd, env=env).returncode
    if code != 0:
        raise SystemExit(code)


def _build_manifest_from_renders(renders_dir: str, manifest_path: str) -> None:
    views = []
    for idx, name in enumerate(sorted(os.listdir(renders_dir))):
        if not name.lower().endswith(".png"):
            continue
        views.append(
            {
                "view_id": f"virtual_{idx:04d}",
                "image_path": os.path.join(renders_dir, name),
            }
        )
    payload = {"views": views}
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)


def _erode_u8_mask(arr_u8: np.ndarray, erode_pixels: int) -> np.ndarray:
    if erode_pixels <= 0:
        return arr_u8
    try:
        import cv2
    except ImportError:
        return arr_u8
    k = 2 * int(erode_pixels) + 1
    kernel = np.ones((k, k), dtype=np.uint8)
    return cv2.erode(arr_u8, kernel)


def _npy_masks_to_png_255(mask_npy_dir: str, out_png_dir: str, erode_pixels: int = 0) -> None:
    """Write LaMa/fusion masks using **image stems** from sam3_results.json.

    Virtual renders use five-digit names (00000.png); SAM3 stores 0000.npy etc.
    Index-only filenames would break downstream paths.
    Optional erosion shrinks the inpainting hole slightly (fewer halos / less mismatch vs pruned GS).
    """
    os.makedirs(out_png_dir, exist_ok=True)
    results_json = os.path.join(mask_npy_dir, "sam3_results.json")
    if os.path.isfile(results_json):
        with open(results_json, "r") as handle:
            payload = json.load(handle)
        for row in payload.get("results", []):
            image_path = row.get("image_path", "")
            mask_path = row.get("mask_path", "")
            if not image_path or not mask_path or not os.path.isfile(mask_path):
                continue
            stem = os.path.splitext(os.path.basename(image_path))[0]
            arr = np.load(mask_path)
            arr = (arr > 0).astype(np.uint8) * 255
            arr = _erode_u8_mask(arr, erode_pixels)
            Image.fromarray(arr).save(os.path.join(out_png_dir, stem + ".png"))
        return
    for name in sorted(os.listdir(mask_npy_dir)):
        if not name.endswith(".npy"):
            continue
        stem = os.path.splitext(name)[0]
        arr = np.load(os.path.join(mask_npy_dir, name))
        arr = (arr > 0).astype(np.uint8) * 255
        arr = _erode_u8_mask(arr, erode_pixels)
        Image.fromarray(arr).save(os.path.join(out_png_dir, stem + ".png"))


def _install_paths(inpaint_root: str) -> None:
    for path in (inpaint_root, os.path.join(inpaint_root, "gaussian_splatting")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _render_virtual_views(
    inpaint_root: str,
    source_path: str,
    model_path: str,
    out_subdir: str,
    circle_radius: float,
    n_frames: int = 30,
) -> None:
    """Render virtual views (RGB+depth) without classifier dependency."""
    _install_paths(inpaint_root)
    import torch
    import cv2
    from scene import Scene
    from gaussian_renderer import GaussianModel, render
    from utils.pose_utils import generate_ellipse_path
    from utils.graphics_utils import getWorld2View2
    from utils.point_utils import get_intrinsics

    class _Dataset:
        pass

    dataset = _Dataset()
    dataset.sh_degree = 3
    dataset.source_path = source_path
    dataset.model_path = model_path
    dataset.images = "images"
    dataset.resolution = 2
    dataset.white_background = False
    dataset.train_test_exp = False
    dataset.data_device = "cuda"
    dataset.eval = True
    dataset.init_mode = "sparse"
    dataset.train_distill = False
    dataset.vanilla_3dgs_path = ""
    dataset.object_path = "associated_sam"
    dataset.n_views = 100
    dataset.random_init = False
    dataset.train_split = False

    class _Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        views = scene.getTrainCameras()
        base = os.path.join(model_path, "virtual", out_subdir)
        render_dir = os.path.join(base, "renders")
        depth_dir = os.path.join(base, "depth")
        os.makedirs(render_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        poses = generate_ellipse_path(
            views,
            n_frames=n_frames,
            is_circle=True,
            circle_radius=float(circle_radius),
        )
        view0 = views[0]
        for idx, pose in enumerate(poses):
            v = views[0]
            v.world_view_transform = torch.tensor(
                getWorld2View2(pose[:3, :3].T, pose[:3, 3], view0.trans, view0.scale)
            ).transpose(0, 1).cuda()
            v.full_proj_transform = (
                v.world_view_transform.unsqueeze(0).bmm(view0.projection_matrix.unsqueeze(0))
            ).squeeze(0)
            v.camera_center = v.world_view_transform.inverse()[3, :3]
            v.image_name = f"{idx:05d}"

            out = render(v, gaussians, _Pipe(), bg)
            rgb = (
                out["render"]
                .detach()
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .mul(255.0)
                .byte()
                .cpu()
                .numpy()
            )
            depth = out["depth_3dgs"].squeeze(0).detach().cpu().numpy().astype(np.float32)
            Image.fromarray(rgb).save(os.path.join(render_dir, f"{idx:05d}.png"))
            np.save(os.path.join(depth_dir, f"{idx:05d}.npy"), depth)
            # write a depth preview png for debugging parity
            dmin, dmax = float(depth.min()), float(depth.max())
            if dmax > dmin:
                dimg = ((depth - dmin) / (dmax - dmin) * 255.0).astype(np.uint8)
            else:
                dimg = np.zeros_like(depth, dtype=np.uint8)
            dimg = cv2.applyColorMap(dimg, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(depth_dir, f"{idx:05d}.png"), dimg)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct SAM3 target pipeline (separate files/workflow)")
    p.add_argument("--scene", required=True)
    p.add_argument("--prompt", required=True, help='SAM3 prompt, e.g. "car"')
    p.add_argument("--sam3-root", default="external/sam3")
    p.add_argument("--sam3-checkpoint", default="checkpoints/sam3")
    p.add_argument("--resolution", type=int, default=2)
    p.add_argument("--max-views", type=int, default=16)
    p.add_argument("--finetune-iters", type=int, default=12000)
    p.add_argument("--checkpoint-video-iters", default="5000 8000")
    p.add_argument("--run-depth-fix", action="store_true")
    p.add_argument("--ring-width", type=int, default=5)
    p.add_argument("--skip-fid-eval", action="store_true", default=True)
    p.add_argument("--min-mask-views", type=int, default=2)
    p.add_argument("--min-mask-fraction", type=float, default=0.25)
    p.add_argument(
        "--prune-mask-erode-pixels",
        type=int,
        default=0,
        help="Erode each train SAM mask before 3D prune (tighter removal; less boundary junk).",
    )
    p.add_argument(
        "--virtual-mask-erode-pixels",
        type=int,
        default=0,
        help="Erode virtual SAM masks when writing inpaint_2d_unseen_mask_virtual PNGs.",
    )
    p.add_argument(
        "--sam3-score-threshold",
        type=float,
        default=0.0,
        help="Passed to run_sam3_multiview (raise to drop low-confidence segments).",
    )
    p.add_argument(
        "--sam3-mask-threshold",
        type=float,
        default=0.5,
        help="Passed to run_sam3_multiview instance mask binarization.",
    )
    p.add_argument("--circle-radius", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = _root()
    inpaint_root = os.path.join(root, "external", "Inpaint360GS")
    scene = args.scene
    source = os.path.join(root, "data", "inpaint360", scene)
    scene_out = os.path.join(root, "output", "inpaint360", scene)
    base_model = os.path.join(scene_out, "3dgs_output")
    pruned_model = os.path.join(root, "output", "inpaint360", f"{scene}_sam3_direct_model", "3dgs_output")
    logs = os.path.join(scene_out, "logs")
    os.makedirs(logs, exist_ok=True)

    # 1) SAM3 masks on training images (traceability + prune input)
    train_manifest = os.path.join(scene_out, "sam3_direct_train_manifest.json")
    views = []
    image_dir = os.path.join(source, f"images_{args.resolution}")
    for i, name in enumerate(sorted(os.listdir(image_dir))):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        views.append({"view_id": f"train_{i:04d}", "image_path": os.path.join(image_dir, name)})
    with open(train_manifest, "w") as f:
        json.dump({"views": views}, f, indent=2)

    train_mask_dir = os.path.join(scene_out, "sam3_direct_train_masks_npy")
    _run(
        [
            sys.executable,
            os.path.join(root, "tools", "run_sam3_multiview.py"),
            "--sam3-root",
            os.path.join(root, args.sam3_root),
            "--checkpoint-path",
            os.path.join(root, args.sam3_checkpoint),
            "--manifest",
            train_manifest,
            "--prompt",
            args.prompt,
            "--output-dir",
            train_mask_dir,
            "--backend",
            "transformers",
            "--device",
            "cuda",
            "--score-threshold",
            str(args.sam3_score_threshold),
            "--mask-threshold",
            str(args.sam3_mask_threshold),
        ],
        cwd=root,
    )

    # 2) Projection prune into a separate model folder
    _run(
        [
            sys.executable,
            os.path.join(root, "tools", "prune_3dgs_with_inpaint360gs_masks.py"),
            "--inpaint360gs-root",
            inpaint_root,
            "--source-path",
            source,
            "--input-model-path",
            base_model,
            "--output-model-path",
            pruned_model,
            "--sam3-results-json",
            os.path.join(train_mask_dir, "sam3_results.json"),
            "--iteration",
            "30000",
            "--max-views",
            str(args.max_views),
            "--min-mask-views",
            str(args.min_mask_views),
            "--min-mask-fraction",
            str(args.min_mask_fraction),
            "--mask-erode-pixels",
            str(args.prune_mask_erode_pixels),
        ],
        cwd=root,
    )

    # 3) Render virtual views from original and pruned model into existing contract dirs
    _render_virtual_views(
        inpaint_root=inpaint_root,
        source_path=source,
        model_path=base_model,
        out_subdir="ours_2000",
        circle_radius=args.circle_radius,
        n_frames=30,
    )
    # Renders land under model_path/virtual/...; SAM3 + LaMa expect scene_out/virtual/...
    src_ours = os.path.join(base_model, "virtual", "ours_2000")
    dst_ours = os.path.join(scene_out, "virtual", "ours_2000")
    if os.path.isdir(dst_ours):
        shutil.rmtree(dst_ours)
    shutil.copytree(src_ours, dst_ours)

    _render_virtual_views(
        inpaint_root=inpaint_root,
        source_path=source,
        model_path=pruned_model,
        out_subdir="ours_object_removal/iteration_2000",
        circle_radius=args.circle_radius,
        n_frames=30,
    )
    # copy pruned virtual removal outputs into canonical scene output path
    src_removed = os.path.join(pruned_model, "virtual", "ours_object_removal", "iteration_2000")
    dst_removed = os.path.join(scene_out, "virtual", "ours_object_removal", "iteration_2000")
    if os.path.isdir(dst_removed):
        shutil.rmtree(dst_removed)
    shutil.copytree(src_removed, dst_removed)

    # 4) SAM3 masks on virtual ORIGINAL renders => inpaint_2d_unseen_mask_virtual
    virtual_manifest = os.path.join(scene_out, "sam3_direct_virtual_manifest.json")
    _build_manifest_from_renders(os.path.join(scene_out, "virtual", "ours_2000", "renders"), virtual_manifest)
    virtual_mask_npy = os.path.join(scene_out, "sam3_direct_virtual_masks_npy")
    _run(
        [
            sys.executable,
            os.path.join(root, "tools", "run_sam3_multiview.py"),
            "--sam3-root",
            os.path.join(root, args.sam3_root),
            "--checkpoint-path",
            os.path.join(root, args.sam3_checkpoint),
            "--manifest",
            virtual_manifest,
            "--prompt",
            args.prompt,
            "--output-dir",
            virtual_mask_npy,
            "--backend",
            "transformers",
            "--device",
            "cuda",
            "--score-threshold",
            str(args.sam3_score_threshold),
            "--mask-threshold",
            str(args.sam3_mask_threshold),
        ],
        cwd=root,
    )
    _npy_masks_to_png_255(
        virtual_mask_npy,
        os.path.join(source, "inpaint_2d_unseen_mask_virtual"),
        erode_pixels=args.virtual_mask_erode_pixels,
    )

    # 5) Existing LaMa prep/run/postprocess
    _run(
        [
            sys.executable,
            os.path.join(inpaint_root, "tools", "prepare_lama_data.py"),
            "-s",
            source,
            "-m",
            scene_out,
            "-r",
            str(args.resolution),
            "--inpaint2lama",
        ],
        cwd=inpaint_root,
    )
    _run(
        [
            sys.executable,
            os.path.join(inpaint_root, "LaMa", "bin", "predict_color.py"),
            "--data_name",
            f"360_{scene}_virtual",
        ],
        cwd=os.path.join(inpaint_root, "LaMa"),
    )
    _run(
        [
            sys.executable,
            os.path.join(inpaint_root, "LaMa", "bin", "predict_depth.py"),
            "--data_name",
            f"360_{scene}_virtual",
        ],
        cwd=os.path.join(inpaint_root, "LaMa"),
    )
    _run(
        [
            sys.executable,
            os.path.join(inpaint_root, "tools", "prepare_lama_data.py"),
            "-s",
            source,
            "-m",
            scene_out,
            "-r",
            str(args.resolution),
        ],
        cwd=inpaint_root,
    )

    # 6) Optional depth fix
    if args.run_depth_fix:
        base = os.path.join(scene_out, "virtual", "ours_object_removal", "iteration_2000")
        _run(
            [
                sys.executable,
                os.path.join(root, "tools", "inpaint360_align_completed_depth.py"),
                "--completed-dir",
                os.path.join(base, "depth_completed"),
                "--hole-dir",
                os.path.join(base, "depth"),
                "--mask-dir",
                os.path.join(source, "inpaint_2d_unseen_mask_virtual"),
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
                os.path.join(base, "depth_completed"),
                "--hole-dir",
                os.path.join(base, "depth"),
                "--mask-dir",
                os.path.join(source, "inpaint_2d_unseen_mask_virtual"),
                "--backup",
                "--ring-width",
                str(args.ring_width),
            ],
            cwd=root,
        )

    # 7) Existing stage10/11
    env = os.environ.copy()
    env["INPAINT360_CHECKPOINT_VIDEO_ITERS"] = args.checkpoint_video_iters
    cmd = [
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
        cmd.append("--skip-fid-eval")
    _run(cmd, cwd=root, env=env)

    print("\n[DONE] SAM3 direct-target pipeline completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
