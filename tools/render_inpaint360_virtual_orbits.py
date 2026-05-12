#!/usr/bin/env python3
"""Render virtual orbit RGB+depth for SAM3-direct / Inpaint360GS LaMa prep.

`_render_virtual_views` writes under ``<model_path>/virtual/<subdir>/``. Downstream
steps (SAM3 manifest, some docs) expect **canonical** paths under
``output/inpaint360/<scene>/virtual/``, so this tool copies there after each
render.

Typical use after manual SAM3 prune::

    python tools/render_inpaint360_virtual_orbits.py --scene car \\
        --base-model output/inpaint360/car/3dgs_output \\
        --pruned-model output/inpaint360/car_sam3_direct_model/3dgs_output

See project_memory/INPAINT360GS_SAM3_DIRECT_GUIDE.md for context.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_render_fn():
    path = os.path.join(_repo_root(), "tools", "sam3_direct", "run_sam3_direct_pipeline.py")
    spec = importlib.util.spec_from_file_location("sam3_direct_rv", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod._render_virtual_views


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", required=True, help="Scene name (e.g. car)")
    p.add_argument(
        "--inpaint360gs-root",
        default="external/Inpaint360GS",
        help="Inpaint360GS checkout root",
    )
    p.add_argument(
        "--source-path",
        default="",
        help="COLMAP scene root (default: data/inpaint360/<scene>)",
    )
    p.add_argument("--base-model", required=True, help="Full 3DGS model dir (…/3dgs_output)")
    p.add_argument("--pruned-model", required=True, help="Pruned 3DGS model dir (…/3dgs_output)")
    p.add_argument("--circle-radius", type=float, default=1.0)
    p.add_argument("--n-frames", type=int, default=30)
    p.add_argument(
        "--skip-pruned",
        action="store_true",
        help="Only render/copy original orbit (ours_2000)",
    )
    p.add_argument(
        "--skip-original",
        action="store_true",
        help="Only render/copy pruned removal orbit",
    )
    return p.parse_args()


def _copytree(src: str, dst: str) -> None:
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[OK] {src} -> {dst}", flush=True)


def main() -> int:
    args = parse_args()
    root = _repo_root()
    inpaint_root = os.path.join(root, args.inpaint360gs_root)
    source = args.source_path or os.path.join(root, "data", "inpaint360", args.scene)
    scene_out = os.path.join(root, "output", "inpaint360", args.scene)
    base = os.path.abspath(os.path.join(root, args.base_model))
    pruned = os.path.abspath(os.path.join(root, args.pruned_model))

    render_v = _load_render_fn()

    if not args.skip_original:
        print("[1/2] Virtual orbit: full model -> virtual/ours_2000", flush=True)
        render_v(
            inpaint_root=inpaint_root,
            source_path=source,
            model_path=base,
            out_subdir="ours_2000",
            circle_radius=args.circle_radius,
            n_frames=args.n_frames,
        )
        _copytree(
            os.path.join(base, "virtual", "ours_2000"),
            os.path.join(scene_out, "virtual", "ours_2000"),
        )

    if not args.skip_pruned:
        print("[2/2] Virtual orbit: pruned model -> virtual/ours_object_removal/iteration_2000", flush=True)
        render_v(
            inpaint_root=inpaint_root,
            source_path=source,
            model_path=pruned,
            out_subdir="ours_object_removal/iteration_2000",
            circle_radius=args.circle_radius,
            n_frames=args.n_frames,
        )
        _copytree(
            os.path.join(pruned, "virtual", "ours_object_removal", "iteration_2000"),
            os.path.join(scene_out, "virtual", "ours_object_removal", "iteration_2000"),
        )

    print("\nDone. Inspect:", flush=True)
    print(" ", os.path.join(scene_out, "virtual", "ours_2000", "renders"), flush=True)
    print(" ", os.path.join(scene_out, "virtual", "ours_object_removal", "iteration_2000", "renders"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
