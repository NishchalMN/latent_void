"""Run the full Inpaint360GS pipeline for one or more scenes.

Usage (on GPU node with venv activated):
    cd /scratch/zt1/project/msml612pcs3/user/gnanesh/latent_void
    export PYTHONPATH=external/Inpaint360GS:external/Inpaint360GS/gaussian_splatting:$PYTHONPATH
    python tools/run_inpaint360gs_full.py --scenes car bag --resolution 2

    # One scene, three 3D inpaint budgets (after LaMa / fusion, runs edit_object_inpaint per value):
    python tools/run_inpaint360gs_full.py --scenes car --finetune-iterations 5000 12000 20000

Stages:
  1. Train vanilla 3DGS (30k iterations)
  2. HQ-SAM 2D segmentation
  3. 3D mask association
  4. Label numbering
  5. Semantic distillation
  6. Auto-identify target object from unseen_mask overlap
  7. Object removal (semantic + convex hull)
  8. Virtual camera trajectory
  9. LaMa 2D inpainting (color + depth)
  10. Point cloud fusion + 3DGS inpainting
  11. Evaluation (PSNR/SSIM/LPIPS/FID)
"""

import argparse
import json
import os
import subprocess
import sys
import time


def get_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_inpaint_root():
    return os.path.join(get_root(), "external", "Inpaint360GS")


def run_cmd(cmd, cwd=None, log_path=None, env=None):
    """Run a command with live terminal output."""
    short = ' '.join(cmd[:6]) + ('...' if len(cmd) > 6 else '')
    print(f"  CMD: {short}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=env)
    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (code {proc.returncode})"
    print(f"  {status} ({elapsed:.0f}s)", flush=True)
    return proc.returncode


def stage_train_3dgs(scene_data, gs_output, resolution, inpaint_root, log_dir):
    """Stage 1: Train vanilla 3DGS."""
    ply = os.path.join(gs_output, "point_cloud", "iteration_30000", "point_cloud.ply")
    if os.path.exists(ply):
        print(f"  [skip] 3DGS already trained: {ply}")
        return True

    train_py = os.path.join(inpaint_root, "gaussian_splatting", "train.py")
    return run_cmd(
        [sys.executable, train_py,
         "-s", scene_data,
         "-m", gs_output,
         "--init_mode", "sparse",
         "--eval",
         "--resolution", str(resolution)],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "train_3dgs.log"),
    ) == 0


def stage_sam_segmentation(scene_name, data_root, resolution, inpaint_root, log_dir):
    """Stage 2: SAM ViT-H automatic instance segmentation."""
    env = os.environ.copy()
    env["PYTHONPATH"] = inpaint_root + ":" + os.path.join(inpaint_root, "seg") + ":" + env.get("PYTHONPATH", "")
    abs_data_root = os.path.abspath(data_root)
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "seg", "raw_mask_sam.py"),
         "--dataset_path", abs_data_root + "/",
         "--scene_name", scene_name,
         "--image_folder", f"images_{resolution}",
         "--method", "sam"],
        cwd=inpaint_root,
    ) == 0


def stage_reduce_segments(scene_data, max_classes=30):
    """Stage 2b: Reduce SAM mask classes to a manageable number."""
    mask_dir = os.path.join(scene_data, "raw_sam")
    if not os.path.isdir(mask_dir):
        print(f"  WARNING: {mask_dir} not found, skipping reduction")
        return True
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return run_cmd(
        [sys.executable, os.path.join(root, "tools", "reduce_sam_segments.py"),
         "--mask-dir", mask_dir,
         "--max-classes", str(max_classes)],
    ) == 0


def stage_mask_associate(scene_data, gs_output, resolution, inpaint_root, log_dir):
    """Stage 3: 3D mask association."""
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "seg", "mask_associate.py"),
         "--source_path", scene_data,
         "--model_path", gs_output,
         "--resolution", str(resolution),
         "--mask_generator", "sam",
         "--eval"],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "mask_associate.log"),
    ) == 0


def stage_add_labels(scene_data, resolution, inpaint_root, log_dir):
    """Stage 4: Add label numbers."""
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "tools", "add_label_num_hqsam.py"),
         "--source_path", scene_data,
         "--resolution", str(resolution),
         "--mask_generator", "sam"],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "add_labels.log"),
    ) == 0


def stage_distillation(scene_data, scene_output, gs_output, resolution, inpaint_root, log_dir):
    """Stage 5: Semantic distillation."""
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "seg", "distillation.py"),
         "--source_path", scene_data,
         "--model_path", scene_output,
         "--vanilla_3dgs_path", gs_output,
         "--resolution", str(resolution),
         "--object_path", "associated_sam",
         "--eval"],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "distillation.log"),
    ) == 0


def find_target_object_id(scene_data, scene_output, resolution, inpaint_root):
    """Identify the target object ID from unseen_mask overlap with label maps.

    Two strategies:
    1. Match unseen_mask test views to label maps (strip test_ prefix, try
       matching image numbers that exist in the label dir).
    2. If no test views match, find the label ID that occupies the image center
       across training views (the target object in 360 scenes is typically
       centered).
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        print("  WARNING: numpy/PIL not available, defaulting to target_id=1")
        return [1], []

    label_dir = os.path.join(scene_data, f"images_{resolution}_num")
    if not os.path.isdir(label_dir):
        print(f"  WARNING: No label dir {label_dir}, defaulting to target_id=1")
        return [1], []

    overlap_counts = {}

    unseen_dir = os.path.join(scene_data, "unseen_mask")
    if os.path.isdir(unseen_dir):
        mask_files = sorted([f for f in os.listdir(unseen_dir) if f.endswith(".png")])
        label_files = {os.path.splitext(f)[0].lower(): f for f in os.listdir(label_dir)}

        for mf in mask_files[:12]:
            mask = np.array(Image.open(os.path.join(unseen_dir, mf)).convert("L")) > 127
            stem = os.path.splitext(mf)[0]
            # Try: exact, strip test_, strip test_ with different case
            candidates = [stem, stem.replace("test_", ""), stem.replace("test_", "").upper()]
            label_path = None
            for c in candidates:
                for ext in (".png", ".jpg", ".JPG", ".jpeg"):
                    p = os.path.join(label_dir, c + ext)
                    if os.path.exists(p):
                        label_path = p
                        break
                if label_path:
                    break
                if c.lower() in label_files:
                    label_path = os.path.join(label_dir, label_files[c.lower()])
                    break

            if label_path is None:
                continue

            labels = np.array(Image.open(label_path))
            if labels.ndim == 3:
                labels = labels[:, :, 0]
            lh, lw = labels.shape[:2]
            mh, mw = mask.shape[:2]
            if (lh, lw) != (mh, mw):
                mask = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(
                    (lw, lh), Image.NEAREST)) > 127
            masked_labels = labels[mask]
            for lid in np.unique(masked_labels):
                if lid == 0:
                    continue
                overlap_counts[int(lid)] = overlap_counts.get(int(lid), 0) + int((masked_labels == lid).sum())

    if overlap_counts:
        sorted_ids = sorted(overlap_counts.items(), key=lambda x: -x[1])
        print(f"  Object ID overlaps (unseen_mask): {sorted_ids[:5]}")
        target_id = [sorted_ids[0][0]]
        surrounding = [sid for sid, _ in sorted_ids[1:3] if _ > sorted_ids[0][1] * 0.1]
        return target_id, surrounding

    # Fallback: use associated_sam masks (which match the 3D classifier IDs)
    assoc_dir = os.path.join(scene_data, "associated_sam")
    if not os.path.isdir(assoc_dir):
        assoc_dir = os.path.join(scene_data, "associated_hqsam")
    use_dir = assoc_dir if os.path.isdir(assoc_dir) else label_dir
    print(f"  No unseen_mask overlap; using center-region heuristic on {os.path.basename(use_dir)}")

    center_counts = {}
    label_files_list = sorted(f for f in os.listdir(use_dir) if f.endswith((".png", ".jpg", ".JPG")))[:20]
    for lf in label_files_list:
        labels = np.array(Image.open(os.path.join(use_dir, lf)))
        if labels.ndim == 3:
            labels = labels[:, :, 0]
        h, w = labels.shape
        cy, cx = h // 2, w // 2
        margin_h, margin_w = h // 6, w // 6
        center_crop = labels[cy - margin_h:cy + margin_h, cx - margin_w:cx + margin_w]
        for lid in np.unique(center_crop):
            if lid == 0:
                continue
            center_counts[int(lid)] = center_counts.get(int(lid), 0) + int((center_crop == lid).sum())

    if center_counts:
        sorted_ids = sorted(center_counts.items(), key=lambda x: -x[1])
        print(f"  Center-region label counts: {sorted_ids[:5]}")
        target_id = [sorted_ids[0][0]]
        surrounding = [sid for sid, _ in sorted_ids[1:3] if _ > sorted_ids[0][1] * 0.1]
        return target_id, surrounding

    print("  WARNING: Could not identify target, defaulting to target_id=1")
    return [1], []


def stage_init_configs(scene_name, target_id, surrounding_ids, scene_data, inpaint_root):
    """Stage 6: Initialize removal/inpaint configs."""
    target_str = ",".join(str(i) for i in target_id)
    surr_str = ",".join(str(i) for i in surrounding_ids) if surrounding_ids else "None"
    rc = run_cmd(
        [sys.executable, os.path.join(inpaint_root, "tools", "init_configs.py"),
         "--dataset_name", "inpaint360",
         "--scene", scene_name,
         "--target_id", target_str,
         "--target_surronding_id", surr_str],
        cwd=inpaint_root,
    )
    if rc != 0:
        return False

    # With many SAM classes, the default 0.7 threshold is too high; lower it
    scene_json = os.path.join(scene_data, "associated_sam", "scene.json")
    if not os.path.exists(scene_json):
        scene_json = os.path.join(scene_data, "associated_hqsam", "scene.json")
    num_classes = 256
    if os.path.exists(scene_json):
        import json as _json
        with open(scene_json) as f:
            num_classes = _json.load(f).get("num_classes", 256)
    if num_classes > 50:
        thresh = round(max(0.01, 5.0 / num_classes), 3)
        print(f"  Adjusting removal_thresh to {thresh} (num_classes={num_classes})", flush=True)
        import json as _json
        for cfg_type in ("object_removal", "object_inpaint"):
            cfg_path = os.path.join(inpaint_root, "config", cfg_type, "inpaint360", f"{scene_name}.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = _json.load(f)
                cfg["removal_thresh"] = thresh
                with open(cfg_path, "w") as f:
                    _json.dump(cfg, f, indent=4)
    return True


def stage_object_removal(scene_data, scene_output, scene_name, inpaint_root, log_dir):
    """Stage 7: Object removal."""
    config_file = os.path.join(inpaint_root, "config", "object_removal", "inpaint360", f"{scene_name}.json")
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "edit_object_removal.py"),
         "--source_path", scene_data,
         "-m", scene_output,
         "--config_file", config_file,
         "--render_video",
         "--skip_train", "--skip_test"],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "object_removal.log"),
    ) == 0


def stage_virtual_pose(scene_data, scene_output, scene_name, inpaint_root, log_dir):
    """Stage 8: Virtual camera trajectory generation."""
    config_file = os.path.join(inpaint_root, "config", "object_removal", "inpaint360", f"{scene_name}.json")
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "tools", "virtual_pose.py"),
         "-s", scene_data,
         "-m", scene_output,
         "--config_file", config_file],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "virtual_pose.log"),
    ) == 0


def stage_generate_virtual_masks(scene_data, scene_output, target_id):
    """Generate binary masks for virtual views from objects_pred maps."""
    from PIL import Image, ImageFilter
    objects_pred_dir = os.path.join(scene_output, "virtual", "ours_2000", "objects_pred")
    mask_dir = os.path.join(scene_data, "inpaint_2d_unseen_mask_virtual")
    os.makedirs(mask_dir, exist_ok=True)

    if not os.path.isdir(objects_pred_dir):
        print(f"  ERROR: objects_pred dir not found: {objects_pred_dir}", flush=True)
        return False

    target_ids = target_id if isinstance(target_id, list) else [target_id]
    count = 0
    for fname in sorted(os.listdir(objects_pred_dir)):
        if not fname.endswith(".png"):
            continue
        img = Image.open(os.path.join(objects_pred_dir, fname)).convert("L")
        data = bytearray(img.tobytes())
        mask_bytes = bytearray(len(data))
        for tid in target_ids:
            for i, val in enumerate(data):
                if val == tid:
                    mask_bytes[i] = 255
        mask = Image.frombytes("L", img.size, bytes(mask_bytes))
        mask = mask.filter(ImageFilter.MaxFilter(size=21))
        mask.save(os.path.join(mask_dir, fname))
        count += 1

    print(f"  Generated {count} virtual masks in {mask_dir}", flush=True)
    return count > 0


def stage_prepare_lama(scene_data, scene_output, resolution, inpaint_root, log_dir):
    """Stage 9a: Prepare data for LaMa (skip SAT, use pre-generated masks)."""
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "tools", "prepare_lama_data.py"),
         "-s", scene_data,
         "-m", scene_output,
         "-r", str(resolution),
         "--inpaint2lama"],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "prepare_lama.log"),
    ) == 0


def stage_lama_inpaint(scene_name, inpaint_root, log_dir):
    """Stage 9b: Run LaMa color and depth inpainting."""
    lama_dir = os.path.join(inpaint_root, "LaMa")
    env = os.environ.copy()
    env["TORCH_HOME"] = lama_dir
    env["PYTHONPATH"] = lama_dir

    rc1 = run_cmd(
        [sys.executable, os.path.join(lama_dir, "bin", "predict_color.py"),
         "--data_name", f"360_{scene_name}_virtual"],
        cwd=lama_dir,
        log_path=os.path.join(log_dir, "lama_color.log"),
        env=env,
    )

    rc2 = run_cmd(
        [sys.executable, os.path.join(lama_dir, "bin", "predict_depth.py"),
         "--data_name", f"360_{scene_name}_virtual"],
        cwd=lama_dir,
        log_path=os.path.join(log_dir, "lama_depth.log"),
        env=env,
    )

    return rc1 == 0 and rc2 == 0


def stage_postprocess_lama(scene_data, scene_output, resolution, inpaint_root, log_dir):
    """Stage 9c: Post-process LaMa output."""
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "tools", "prepare_lama_data.py"),
         "-s", scene_data,
         "-m", scene_output,
         "-r", str(resolution)],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "postprocess_lama.log"),
    ) == 0


def stage_ply_fusion(scene_data, scene_output, scene_name, inpaint_root, log_dir):
    """Stage 10a: Colorful point cloud fusion."""
    config_file = os.path.join(inpaint_root, "config", "object_removal", "inpaint360", f"{scene_name}.json")
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "edit_object_removal_plyfusion.py"),
         "-s", scene_data,
         "-m", scene_output,
         "--config_file", config_file],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "ply_fusion.log"),
    ) == 0


def stage_gs_inpaint(scene_data, scene_output, scene_name, resolution, inpaint_root, log_dir):
    """Stage 10b: 3DGS inpainting optimization."""
    config_file = os.path.join(inpaint_root, "config", "object_inpaint", "inpaint360", f"{scene_name}.json")
    return run_cmd(
        [sys.executable, os.path.join(inpaint_root, "edit_object_inpaint.py"),
         "-s", scene_data,
         "-m", scene_output,
         "--config_file", config_file,
         "--resolution", str(resolution),
         "--render_video"],
        cwd=inpaint_root,
        log_path=os.path.join(log_dir, "gs_inpaint.log"),
    ) == 0


def patch_object_inpaint_finetune_iteration(inpaint_root, scene_name, finetune_iteration):
    """Set finetune_iteration in object_inpaint JSON (stage 10b reads this)."""
    cfg_path = os.path.join(
        inpaint_root, "config", "object_inpaint", "inpaint360", f"{scene_name}.json")
    if not os.path.isfile(cfg_path):
        print(f"  ERROR: missing object_inpaint config: {cfg_path}", flush=True)
        return False
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["finetune_iteration"] = int(finetune_iteration)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=4)
    print(f"  Patched {cfg_path} -> finetune_iteration={finetune_iteration}", flush=True)
    return True


def stage_evaluate(scene_output, inpaint_root, log_dir, skip_fid=False):
    """Stage 11: Evaluation (bundled script, repo-root cwd for data/inpaint360 paths)."""
    root = get_root()
    script = os.path.join(root, "tools", "inpaint360gs_metrics_fid_masked.py")
    env = os.environ.copy()
    gs = os.path.join(inpaint_root, "gaussian_splatting")
    _pp = [inpaint_root, gs]
    if env.get("PYTHONPATH"):
        _pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(_pp)
    cmd = [sys.executable, script, "-m", os.path.abspath(scene_output)]
    if skip_fid:
        cmd.append("--skip-fid")
    return run_cmd(
        cmd,
        cwd=root,
        env=env,
        log_path=os.path.join(log_dir, "evaluation.log"),
    ) == 0


def process_scene(scene_name, data_root, output_root, resolution, inpaint_root,
                   skip_seg=False, skip_sam=False, start_stage=1, skip_fid_eval=False,
                   finetune_iterations=None):
    """Run the full pipeline for a single scene.

    If finetune_iterations is a non-empty list, stage 10b runs once per value
    (same LaMa/fusion preamble), writing e.g. .../iteration_5000, iteration_12000.
    Evaluation then picks up every inpaint/*/iteration_* folder.
    """
    scene_data = os.path.join(data_root, scene_name)
    scene_output = os.path.join(output_root, scene_name)
    gs_output = os.path.join(scene_output, "3dgs_output")
    log_dir = os.path.join(scene_output, "logs")
    os.makedirs(log_dir, exist_ok=True)

    status = {"scene": scene_name, "stages": {}}
    t0 = time.time()

    run_seg_stages = not skip_seg and not skip_sam and start_stage <= 2

    if run_seg_stages:
        print(f"\n--- Stage 1/11: Train 3DGS ---", flush=True)
        ok = stage_train_3dgs(scene_data, gs_output, resolution, inpaint_root, log_dir)
        status["stages"]["train_3dgs"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: 3DGS training failed for {scene_name}", flush=True)
            return status

        print(f"\n--- Stage 2a/11: SAM segmentation ---", flush=True)
        ok = stage_sam_segmentation(scene_name, data_root, resolution, inpaint_root, log_dir)
        status["stages"]["sam_seg"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: SAM segmentation failed for {scene_name}", flush=True)
            return status

        print(f"\n--- Stage 2b/11: Reduce SAM segments ---", flush=True)
        ok = stage_reduce_segments(scene_data, max_classes=30)
        status["stages"]["reduce_seg"] = "ok" if ok else "fail"

    if not skip_seg and start_stage <= 5:
        print(f"\n--- Stage 3/11: 3D mask association ---", flush=True)
        ok = stage_mask_associate(scene_data, gs_output, resolution, inpaint_root, log_dir)
        status["stages"]["mask_associate"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: Mask association failed for {scene_name}", flush=True)
            return status

        print(f"\n--- Stage 4/11: Label numbering ---", flush=True)
        ok = stage_add_labels(scene_data, resolution, inpaint_root, log_dir)
        status["stages"]["add_labels"] = "ok" if ok else "fail"

        print(f"\n--- Stage 5/11: Semantic distillation ---", flush=True)
        ok = stage_distillation(scene_data, scene_output, gs_output, resolution, inpaint_root, log_dir)
        status["stages"]["distillation"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: Distillation failed for {scene_name}", flush=True)
            return status

    if start_stage <= 6:
        print(f"\n--- Stage 6/11: Auto-identify target object ---", flush=True)
        target_id, surrounding_ids = find_target_object_id(
            scene_data, scene_output, resolution, inpaint_root)
        status["target_id"] = target_id
        status["surrounding_ids"] = surrounding_ids
        print(f"  Target: {target_id}, Surrounding: {surrounding_ids}")

        ok = stage_init_configs(scene_name, target_id, surrounding_ids, scene_data, inpaint_root)
        status["stages"]["init_configs"] = "ok" if ok else "fail"
    else:
        config_file = os.path.join(inpaint_root, "config", "object_removal", "inpaint360", f"{scene_name}.json")
        if os.path.exists(config_file):
            import json as _json
            with open(config_file) as f:
                cfg = _json.load(f)
            target_id = cfg.get("select_obj_id", [182])
            surrounding_ids = cfg.get("target_surronding_id", [])
            print(f"  Loaded target from config: {target_id}, Surrounding: {surrounding_ids}", flush=True)
        else:
            target_id = [182]
            surrounding_ids = []

    if start_stage <= 7:
        print(f"\n--- Stage 7/11: Object removal ---")
        ok = stage_object_removal(scene_data, scene_output, scene_name, inpaint_root, log_dir)
        status["stages"]["object_removal"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: Object removal failed for {scene_name}", flush=True)
            status["elapsed_seconds"] = round(time.time() - t0, 1)
            return status

    if start_stage <= 8:
        print(f"\n--- Stage 8/11: Virtual camera trajectory ---")
        ok = stage_virtual_pose(scene_data, scene_output, scene_name, inpaint_root, log_dir)
        status["stages"]["virtual_pose"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: Virtual pose failed for {scene_name}", flush=True)
            status["elapsed_seconds"] = round(time.time() - t0, 1)
            return status

    if start_stage <= 9:
        print(f"\n--- Stage 8b/11: Generate virtual view masks ---")
        ok = stage_generate_virtual_masks(scene_data, scene_output, target_id)
        status["stages"]["virtual_masks"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: Virtual mask generation failed for {scene_name}", flush=True)
            status["elapsed_seconds"] = round(time.time() - t0, 1)
            return status

        print(f"\n--- Stage 9/11: LaMa inpainting ---")
        ok = stage_prepare_lama(scene_data, scene_output, resolution, inpaint_root, log_dir)
        status["stages"]["prepare_lama"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: LaMa data prep failed for {scene_name}", flush=True)
            status["elapsed_seconds"] = round(time.time() - t0, 1)
            return status

        ok = stage_lama_inpaint(scene_name, inpaint_root, log_dir)
        status["stages"]["lama_inpaint"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: LaMa inpaint failed for {scene_name}", flush=True)
            status["elapsed_seconds"] = round(time.time() - t0, 1)
            return status

        ok = stage_postprocess_lama(scene_data, scene_output, resolution, inpaint_root, log_dir)
        status["stages"]["postprocess_lama"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: LaMa postprocess failed for {scene_name}", flush=True)
            status["elapsed_seconds"] = round(time.time() - t0, 1)
            return status

    if start_stage <= 10:
        print(f"\n--- Stage 10/11: 3D inpainting ---")
        ok = stage_ply_fusion(scene_data, scene_output, scene_name, inpaint_root, log_dir)
        status["stages"]["ply_fusion"] = "ok" if ok else "fail"
        if not ok:
            print(f"  FATAL: PLY fusion failed for {scene_name}", flush=True)
            status["elapsed_seconds"] = round(time.time() - t0, 1)
            return status

        if finetune_iterations:
            for it in finetune_iterations:
                print(f"\n  --- 3D inpaint (finetune_iteration={it}) ---", flush=True)
                if not patch_object_inpaint_finetune_iteration(inpaint_root, scene_name, it):
                    status["stages"][f"gs_inpaint_{it}"] = "fail"
                    ok = False
                    break
                ok = stage_gs_inpaint(
                    scene_data, scene_output, scene_name, resolution, inpaint_root, log_dir)
                status["stages"][f"gs_inpaint_{it}"] = "ok" if ok else "fail"
                if not ok:
                    print(f"  FATAL: 3D inpaint failed at finetune_iteration={it}", flush=True)
                    break
        else:
            ok = stage_gs_inpaint(
                scene_data, scene_output, scene_name, resolution, inpaint_root, log_dir)
            status["stages"]["gs_inpaint"] = "ok" if ok else "fail"

    if start_stage <= 11:
        print(f"\n--- Stage 11/11: Evaluation ---")
        ok = stage_evaluate(scene_output, inpaint_root, log_dir, skip_fid=skip_fid_eval)
        status["stages"]["evaluation"] = "ok" if ok else "fail"

    status["elapsed_seconds"] = round(time.time() - t0, 1)
    status_path = os.path.join(scene_output, "pipeline_status.json")
    with open(status_path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"\n  Pipeline status written to {status_path}")

    return status


def main():
    parser = argparse.ArgumentParser(description="Run full Inpaint360GS pipeline")
    parser.add_argument("--scenes", nargs="+", default=["car"],
                        help="Scene names to process")
    parser.add_argument("--resolution", type=int, default=2,
                        help="Image resolution factor (1, 2, 4, 8)")
    parser.add_argument("--skip-seg", action="store_true",
                        help="Skip segmentation stages (1-5), assume already done")
    parser.add_argument("--skip-sam", action="store_true",
                        help="Skip SAM + 3DGS training but re-run mask association and distillation")
    parser.add_argument("--start-stage", type=int, default=1,
                        help="Start from this stage (1-11). 9=LaMa onward, 10=PLY+GS inpaint, 11=eval only")
    parser.add_argument("--skip-fid-eval", action="store_true",
                        help="Skip FID in metrics (offline GPU nodes); PSNR/SSIM/LPIPS still run")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--finetune-iterations",
        type=int,
        nargs="+",
        default=None,
        help="Optional: run stage 10b once per value (e.g. 5000 12000 20000). "
             "Patches object_inpaint/<scene>.json finetune_iteration before each run.",
    )
    args = parser.parse_args()

    root = os.path.abspath(get_root())
    inpaint_root = os.path.abspath(get_inpaint_root())
    data_root = os.path.abspath(args.data_root or os.path.join(root, "data", "inpaint360"))
    output_root = os.path.abspath(args.output_root or os.path.join(root, "output", "inpaint360"))

    sys.path.insert(0, inpaint_root)
    sys.path.insert(0, os.path.join(inpaint_root, "gaussian_splatting"))

    all_status = {}
    for scene in args.scenes:
        print(f"\n{'='*60}")
        print(f"  Processing scene: {scene}")
        print(f"{'='*60}")
        status = process_scene(
            scene, data_root, output_root, args.resolution, inpaint_root,
            skip_seg=args.skip_seg, skip_sam=args.skip_sam,
            start_stage=args.start_stage,
            skip_fid_eval=args.skip_fid_eval,
            finetune_iterations=args.finetune_iterations,
        )
        all_status[scene] = status

    summary_path = os.path.join(output_root, "pipeline_summary.json")
    os.makedirs(output_root, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_status, f, indent=2)
    print(f"\n{'='*60}")
    print(f"  Pipeline summary: {summary_path}")
    for scene, st in all_status.items():
        fails = [k for k, v in st.get("stages", {}).items() if v != "ok"]
        if fails:
            print(f"  {scene}: FAILURES in {fails}")
        else:
            print(f"  {scene}: ALL STAGES OK ({st.get('elapsed_seconds', '?')}s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
