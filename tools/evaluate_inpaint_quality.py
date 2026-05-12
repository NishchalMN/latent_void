"""Evaluate inpainting quality: PSNR, SSIM, LPIPS against ground-truth object-free views.

Usage:
    python tools/evaluate_inpaint_quality.py \
        --scenes car bag cube garden_toys truck fruits \
        --output-root output/inpaint360 \
        --data-root data/inpaint360
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def ssim_channel(c1, c2, k1=0.01, k2=0.03, L=255):
    c1 = c1.astype(np.float64)
    c2 = c2.astype(np.float64)
    mu1, mu2 = c1.mean(), c2.mean()
    s1, s2 = c1.var(), c2.var()
    s12 = ((c1 - mu1) * (c2 - mu2)).mean()
    C1, C2 = (k1 * L) ** 2, (k2 * L) ** 2
    num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
    return num / den


def ssim(img1, img2):
    vals = []
    for c in range(min(img1.shape[2], img2.shape[2])):
        vals.append(ssim_channel(img1[:, :, c], img2[:, :, c]))
    return float(np.mean(vals))


def compute_lpips(img1_path, img2_path, lpips_fn):
    """Compute LPIPS using torchvision if available."""
    if lpips_fn is None:
        return None
    import torch
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize(0.5, 0.5)])
    i1 = transform(Image.open(img1_path).convert("RGB")).unsqueeze(0)
    i2 = transform(Image.open(img2_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        return float(lpips_fn(i1, i2).item())


def find_rendered_views(scene_output):
    """Find rendered inpainted views in the Inpaint360GS output."""
    candidates = [
        os.path.join(scene_output, "renders"),
        os.path.join(scene_output, "train", "ours_30000", "renders"),
        os.path.join(scene_output, "test", "ours_30000", "renders"),
    ]
    for d in candidates:
        if os.path.isdir(d) and os.listdir(d):
            return d
    return None


def find_gt_views(scene_data):
    """Find ground-truth object-free views from the dataset."""
    gt_candidates = []
    for img_dir in ["images", "images_2", "images_4", "images_8"]:
        d = os.path.join(scene_data, img_dir)
        if os.path.isdir(d):
            test_files = sorted([f for f in os.listdir(d) if f.startswith("test_")])
            if test_files:
                return d, test_files
    return None, []


def evaluate_scene(scene_name, scene_data, scene_output, lpips_fn=None):
    """Evaluate a single scene."""
    render_dir = find_rendered_views(scene_output)
    gt_dir, gt_files = find_gt_views(scene_data)

    if render_dir is None:
        print(f"  No rendered views found in {scene_output}")
        return None
    if gt_dir is None:
        print(f"  No GT views found in {scene_data}")
        return None

    rendered_files = sorted(os.listdir(render_dir))
    metrics_per_view = []

    print(f"  Renders: {render_dir} ({len(rendered_files)} files)")
    print(f"  GT: {gt_dir} ({len(gt_files)} test files)")

    for gt_file in gt_files:
        gt_path = os.path.join(gt_dir, gt_file)
        gt_img = np.array(Image.open(gt_path).convert("RGB"))

        stem = os.path.splitext(gt_file)[0]
        render_path = None
        for rf in rendered_files:
            if stem in rf or os.path.splitext(rf)[0] == stem:
                render_path = os.path.join(render_dir, rf)
                break

        if render_path is None:
            continue

        render_img = np.array(Image.open(render_path).convert("RGB"))

        if gt_img.shape != render_img.shape:
            render_img = np.array(
                Image.fromarray(render_img).resize(
                    (gt_img.shape[1], gt_img.shape[0]), Image.LANCZOS))

        p = psnr(gt_img, render_img)
        s = ssim(gt_img, render_img)
        lp = compute_lpips(render_path, gt_path, lpips_fn) if lpips_fn else None

        view_metrics = {"view": gt_file, "psnr": round(p, 4), "ssim": round(s, 6)}
        if lp is not None:
            view_metrics["lpips"] = round(lp, 6)
        metrics_per_view.append(view_metrics)

    if not metrics_per_view:
        print(f"  No matching view pairs found")
        return None

    avg = {
        "num_views": len(metrics_per_view),
        "mean_psnr": round(np.mean([m["psnr"] for m in metrics_per_view]), 4),
        "mean_ssim": round(np.mean([m["ssim"] for m in metrics_per_view]), 6),
    }
    if metrics_per_view[0].get("lpips") is not None:
        avg["mean_lpips"] = round(np.mean([m["lpips"] for m in metrics_per_view]), 6)

    return {"scene": scene_name, "per_view": metrics_per_view, "average": avg}


def generate_comparison_sheet(scene_name, scene_data, scene_output, out_path):
    """Generate a visual comparison sheet: source | inpainted | target."""
    gt_dir, gt_files = find_gt_views(scene_data)
    render_dir = find_rendered_views(scene_output)

    if not gt_dir or not render_dir or not gt_files:
        return False

    from PIL import Image as PILImage
    gt_files = gt_files[:8]
    rows = []

    for gt_file in gt_files:
        gt_path = os.path.join(gt_dir, gt_file)
        gt_img = PILImage.open(gt_path).convert("RGB")

        stem = os.path.splitext(gt_file)[0]
        render_path = None
        for rf in sorted(os.listdir(render_dir)):
            if stem in rf:
                render_path = os.path.join(render_dir, rf)
                break
        if render_path is None:
            continue

        render_img = PILImage.open(render_path).convert("RGB").resize(gt_img.size, PILImage.LANCZOS)

        w, h = gt_img.size
        row = PILImage.new("RGB", (w * 2, h))
        row.paste(render_img, (0, 0))
        row.paste(gt_img, (w, 0))
        rows.append(row)

    if not rows:
        return False

    total_h = sum(r.height for r in rows)
    sheet = PILImage.new("RGB", (rows[0].width, total_h))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sheet.save(out_path)
    print(f"  Comparison sheet: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", nargs="+", default=["car"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--eval-output", default=None)
    parser.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_root = args.data_root or os.path.join(root, "data", "inpaint360")
    output_root = args.output_root or os.path.join(root, "output", "inpaint360")
    eval_output = args.eval_output or os.path.join(root, "runs", "evaluation")
    os.makedirs(eval_output, exist_ok=True)

    lpips_fn = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net="alex")
            lpips_fn.eval()
            print("LPIPS loaded (AlexNet)")
        except Exception as e:
            print(f"LPIPS not available: {e}")

    all_results = {}
    for scene in args.scenes:
        print(f"\n--- Evaluating {scene} ---")
        scene_data = os.path.join(data_root, scene)
        scene_output = os.path.join(output_root, scene)

        result = evaluate_scene(scene, scene_data, scene_output, lpips_fn)
        if result:
            all_results[scene] = result
            print(f"  PSNR: {result['average']['mean_psnr']:.2f}")
            print(f"  SSIM: {result['average']['mean_ssim']:.4f}")
            if "mean_lpips" in result["average"]:
                print(f"  LPIPS: {result['average']['mean_lpips']:.4f}")

        sheet_path = os.path.join(eval_output, f"{scene}_comparison.png")
        generate_comparison_sheet(scene, scene_data, scene_output, sheet_path)

    summary_path = os.path.join(eval_output, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nEvaluation summary: {summary_path}")

    if all_results:
        print("\n=== Summary Table ===")
        print(f"{'Scene':<15} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Views':>6}")
        print("-" * 50)
        for scene, r in all_results.items():
            avg = r["average"]
            lp = f"{avg.get('mean_lpips', 0):.4f}" if "mean_lpips" in avg else "N/A"
            print(f"{scene:<15} {avg['mean_psnr']:>8.2f} {avg['mean_ssim']:>8.4f} {lp:>8} {avg['num_views']:>6}")


if __name__ == "__main__":
    main()
