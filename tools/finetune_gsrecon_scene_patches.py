#!/usr/bin/env python3
"""Fine-tune DiffSplat GSRecon on latent_void scene-local patch manifests."""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.diffsplat_compat import patch_gaussian_rasterizer_compat, patch_optional_imports, patch_transformers_compat
from latent_void.io import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diffsplat-root", default="external/DiffSplat")
    parser.add_argument("--weights", default="checkpoints/diffsplat/gsrecon_gobj265k_cnp_even4")
    parser.add_argument("--init-model-state", default="")
    parser.add_argument("--opt-type", default="gsvae_sdxl_fp16")
    parser.add_argument("--ckpt-iter", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--trainable", choices=["heads", "heads_and_embed", "last_blocks", "all"], default="heads")
    parser.add_argument("--train-last-blocks", type=int, default=2)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--alpha-weight", type=float, default=0.2)
    parser.add_argument("--coord-weight", type=float, default=0.05)
    parser.add_argument("--normal-weight", type=float, default=0.02)
    parser.add_argument("--l1-weight", type=float, default=0.0)
    parser.add_argument("--ssim-weight", type=float, default=0.0)
    parser.add_argument("--alpha-bce-weight", type=float, default=0.0)
    parser.add_argument("--alpha-dice-weight", type=float, default=0.0)
    parser.add_argument("--depth-weight", type=float, default=0.0)
    parser.add_argument("--source-opacity-weight", type=float, default=0.0)
    parser.add_argument("--source-rgb-weight", type=float, default=0.0)
    parser.add_argument("--foreground-weight", type=float, default=1.0)
    parser.add_argument("--background-weight", type=float, default=1.0)
    parser.add_argument("--alpha-foreground-weight", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--fixed-eval-interval", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--sample-id-contains", default="")
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--random-view-subset", action="store_true")
    parser.add_argument("--fixed-eval-samples", type=int, default=3)
    parser.add_argument("--skip-transformers-patch", action="store_true")
    return parser.parse_args()


def _checkpoint_dir(path):
    nested = os.path.join(path, "checkpoints")
    return nested if os.path.isdir(nested) else path


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _load_chw(path):
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("expected CHW/HW array at %s, got %s" % (path, array.shape))
    return array


def _select_views(views, max_views, rng=None):
    if len(views) <= max_views:
        return list(views)
    if rng is None:
        return list(views[:max_views])
    indices = sorted(rng.choice(len(views), size=max_views, replace=False).tolist())
    return [views[idx] for idx in indices]


def _load_sample(sample, torch, device, max_views, rng=None):
    manifest = _load_json(sample["patch_manifest"])
    views = _select_views(manifest.get("views", []), max_views, rng)
    if len(views) < 4:
        raise RuntimeError("patch manifest has fewer than 4 views: %s" % sample["patch_manifest"])
    images, masks, normals, coords, depths, c2ws, intrinsics = [], [], [], [], [], [], []
    for view in views:
        image = _load_chw(view["rgb_npy"])[:3]
        mask = _load_chw(view.get("mask_npy"))[:1]
        normal = _load_chw(view["normal_npy"])[:3]
        coord = _load_chw(view["coord_npy"])[:3]
        depth = _load_chw(view["depth_npy"])[:1]
        if mask.max() > 1.0:
            mask = mask / 255.0
        images.append(image)
        masks.append(mask)
        normals.append(normal)
        coords.append(coord)
        depths.append(depth)
        c2ws.append(np.asarray(view["camera"]["c2w"], dtype=np.float32))
        intrinsics.append(np.asarray(view["scaled_intrinsics"]["fxfycxcy_normalized"], dtype=np.float32))
    batch = {
        "image": torch.from_numpy(np.stack(images, axis=0)[None]).to(device),
        "mask": torch.from_numpy(np.stack(masks, axis=0)[None]).to(device),
        "normal": torch.from_numpy(np.stack(normals, axis=0)[None]).to(device),
        "coord": torch.from_numpy(np.stack(coords, axis=0)[None]).to(device),
        "depth": torch.from_numpy(np.stack(depths, axis=0)[None]).to(device),
        "C2W": torch.from_numpy(np.stack(c2ws, axis=0)[None]).to(device),
        "fxfycxcy": torch.from_numpy(np.stack(intrinsics, axis=0)[None]).to(device),
    }
    return batch


def _set_trainable(model, mode):
    for parameter in model.parameters():
        parameter.requires_grad_(mode == "all")
    if mode == "all":
        return
    names = ["out_depth", "out_rgb", "out_scale", "out_rotation", "out_opacity", "ln_out"]
    if mode in ("heads_and_embed", "last_blocks"):
        names.append("x_embedder")
    for module_name, module in model.named_modules():
        if any(module_name == name or module_name.startswith(name + ".") for name in names):
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(True)
    if mode == "last_blocks":
        blocks = getattr(getattr(model, "transformer", None), "blocks", [])
        count = min(len(blocks), int(getattr(model, "_latent_void_train_last_blocks", 2)))
        for block in list(blocks)[-count:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)


def _weighted_mse(pred, target, weight):
    return ((pred - target) ** 2 * weight).sum() / weight.sum().clamp_min(1e-6)


def _weighted_l1(pred, target, weight):
    return ((pred - target).abs() * weight).sum() / weight.sum().clamp_min(1e-6)


def _ssim_loss(pred, target, torch):
    pred_flat = pred.flatten(0, 1)
    target_flat = target.flatten(0, 1)
    mu_x = torch.nn.functional.avg_pool2d(pred_flat, 11, stride=1, padding=5)
    mu_y = torch.nn.functional.avg_pool2d(target_flat, 11, stride=1, padding=5)
    sigma_x = torch.nn.functional.avg_pool2d(pred_flat * pred_flat, 11, stride=1, padding=5) - mu_x * mu_x
    sigma_y = torch.nn.functional.avg_pool2d(target_flat * target_flat, 11, stride=1, padding=5) - mu_y * mu_y
    sigma_xy = torch.nn.functional.avg_pool2d(pred_flat * target_flat, 11, stride=1, padding=5) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return ((1.0 - ssim.clamp(-1.0, 1.0)) * 0.5).mean()


def _dice_loss(pred, target):
    pred = pred.clamp(1e-4, 1.0 - 1e-4)
    numerator = 2.0 * (pred * target).sum() + 1e-6
    denominator = pred.sum() + target.sum() + 1e-6
    return 1.0 - numerator / denominator


def _normalized_depth(depth, mask):
    valid = mask > 0.05
    if not bool(valid.any()):
        return depth
    masked = depth[valid]
    lo = masked.quantile(0.05)
    hi = masked.quantile(0.95)
    return ((depth - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _forward_loss(model, batch, torch, args, dtype):
    images = batch["image"].to(dtype)
    masks = batch["mask"].to(dtype)
    normals = batch["normal"].to(dtype)
    coords = batch["coord"].to(dtype)
    depths = batch["depth"].to(dtype)
    c2w = batch["C2W"].to(dtype)
    intr = batch["fxfycxcy"].to(dtype)
    num_input = model.opt.num_input_views
    input_images = torch.cat([images[:, :num_input], normals[:, :num_input], coords[:, :num_input]], dim=2)
    model_outputs = model.forward_gaussians(input_images, c2w[:, :num_input], intr[:, :num_input])
    render_outputs = model.gs_renderer.render(model_outputs, c2w[:, :num_input], intr[:, :num_input], c2w, intr)
    render_images = render_outputs["image"].to(dtype)
    render_masks = render_outputs["alpha"].to(dtype)
    render_coords = render_outputs.get("coord")
    render_normals = render_outputs.get("normal")
    render_depths = render_outputs.get("raw_depth")
    image_mse = torch.nn.functional.mse_loss(render_images, images)
    alpha_mse = torch.nn.functional.mse_loss(render_masks, masks)
    image_weight = args.background_weight + args.foreground_weight * masks
    alpha_weight = 1.0 + args.alpha_foreground_weight * masks
    foreground_image_mse = _weighted_mse(render_images, images, image_weight)
    foreground_alpha_mse = _weighted_mse(render_masks, masks, alpha_weight)
    loss = args.image_weight * foreground_image_mse + args.alpha_weight * foreground_alpha_mse
    if args.l1_weight > 0:
        foreground_l1 = _weighted_l1(render_images, images, image_weight)
        loss = loss + args.l1_weight * foreground_l1
    else:
        foreground_l1 = None
    if args.ssim_weight > 0:
        ssim = _ssim_loss(render_images, images, torch)
        loss = loss + args.ssim_weight * ssim
    else:
        ssim = None
    if args.alpha_bce_weight > 0:
        alpha_bce = torch.nn.functional.binary_cross_entropy(render_masks.clamp(1e-4, 1.0 - 1e-4), masks, weight=alpha_weight)
        loss = loss + args.alpha_bce_weight * alpha_bce
    else:
        alpha_bce = None
    if args.alpha_dice_weight > 0:
        alpha_dice = _dice_loss(render_masks, masks)
        loss = loss + args.alpha_dice_weight * alpha_dice
    else:
        alpha_dice = None
    if args.source_opacity_weight > 0 and "opacity" in model_outputs:
        source_opacity = (model_outputs["opacity"].to(dtype) * 0.5 + 0.5).clamp(1e-4, 1.0 - 1e-4)
        source_masks = masks[:, :source_opacity.shape[1]]
        source_opacity_bce = torch.nn.functional.binary_cross_entropy(source_opacity, source_masks)
        source_opacity_dice = _dice_loss(source_opacity, source_masks)
        source_opacity_loss = source_opacity_bce + source_opacity_dice
        loss = loss + args.source_opacity_weight * source_opacity_loss
    else:
        source_opacity_bce = None
        source_opacity_dice = None
    if args.source_rgb_weight > 0 and "rgb" in model_outputs:
        source_rgb = model_outputs["rgb"].to(dtype) * 0.5 + 0.5
        source_images = images[:, :source_rgb.shape[1]]
        source_masks = masks[:, :source_rgb.shape[1]]
        source_rgb_l1 = _weighted_l1(source_rgb, source_images, source_masks + 0.01)
        loss = loss + args.source_rgb_weight * source_rgb_l1
    else:
        source_rgb_l1 = None
    metrics = {
        "loss": loss,
        "image_mse": image_mse.detach(),
        "alpha_mse": alpha_mse.detach(),
        "foreground_image_mse": foreground_image_mse.detach(),
        "foreground_alpha_mse": foreground_alpha_mse.detach(),
        "render_alpha_mean": render_masks.detach().mean(),
        "target_alpha_mean": masks.detach().mean(),
    }
    if foreground_l1 is not None:
        metrics["foreground_l1"] = foreground_l1.detach()
    if ssim is not None:
        metrics["dssim"] = ssim.detach()
    if alpha_bce is not None:
        metrics["alpha_bce"] = alpha_bce.detach()
    if alpha_dice is not None:
        metrics["alpha_dice"] = alpha_dice.detach()
    if source_opacity_bce is not None:
        metrics["source_opacity_bce"] = source_opacity_bce.detach()
        metrics["source_opacity_dice"] = source_opacity_dice.detach()
    if source_rgb_l1 is not None:
        metrics["source_rgb_l1"] = source_rgb_l1.detach()
    if args.depth_weight > 0 and render_depths is not None:
        render_depth_norm = _normalized_depth(render_depths.to(dtype), masks)
        target_depth_norm = _normalized_depth(depths, masks)
        depth_l1 = _weighted_l1(render_depth_norm, target_depth_norm, masks + 1e-3)
        loss = loss + args.depth_weight * depth_l1
        metrics["depth_l1"] = depth_l1.detach()
    if args.coord_weight > 0 and render_coords is not None:
        coord_mse = torch.nn.functional.mse_loss(render_coords.to(dtype), coords)
        loss = loss + args.coord_weight * coord_mse
        metrics["coord_mse"] = coord_mse.detach()
    if args.normal_weight > 0 and render_normals is not None:
        normal_cos = torch.nn.functional.cosine_similarity(render_normals.to(dtype), normals, dim=2).mean()
        normal_loss = 1.0 - normal_cos
        loss = loss + args.normal_weight * normal_loss
        metrics["normal_cosim"] = normal_cos.detach()
    metrics["loss"] = loss
    metrics["render_images"] = render_images.detach()
    metrics["target_images"] = images.detach()
    metrics["render_alpha"] = render_masks.detach()
    metrics["target_alpha"] = masks.detach()
    return metrics


def _to_rgb_image(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:
        array = array[0]
    array = np.clip(array[:3], 0.0, 1.0)
    return (np.transpose(array, (1, 2, 0)) * 255.0).astype(np.uint8)


def _label(image, label):
    image = Image.fromarray(image).convert("RGB")
    pad = 24
    canvas = Image.new("RGB", (image.width, image.height + pad), "white")
    canvas.paste(image, (0, pad))
    ImageDraw.Draw(canvas).text((4, 4), label, fill=(0, 0, 0))
    return canvas


def _write_eval_sheet(path, metrics, title=""):
    renders = metrics["render_images"].detach().float().cpu().numpy()[0]
    targets = metrics["target_images"].detach().float().cpu().numpy()[0]
    cells = []
    for idx in range(min(renders.shape[0], 4)):
        pred = _to_rgb_image(renders[idx])
        target = _to_rgb_image(targets[idx])
        diff = np.abs(pred.astype(np.float32) - target.astype(np.float32)) / 255.0
        diff = (np.clip(diff / max(float(diff.max()), 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)
        cells.extend([
            _label(pred, "%srender %d" % (title, idx)),
            _label(target, "target %d" % idx),
            _label(diff, "abs diff %d" % idx),
        ])
    if not cells:
        return
    width = max(cell.width for cell in cells)
    height = max(cell.height for cell in cells)
    cols = 3
    rows = (len(cells) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), "white")
    for idx, cell in enumerate(cells):
        sheet.paste(cell, ((idx % cols) * width, (idx // cols) * height))
    sheet.save(path)


def _evaluate_fixed(model, samples, torch, device, args, dtype, eval_dir, label, max_samples):
    model.eval()
    records = []
    with torch.no_grad():
        for idx, sample in enumerate(samples[:max_samples]):
            batch = _load_sample(sample, torch, device, int(args.max_views), None)
            metrics = _forward_loss(model, batch, torch, args, dtype)
            record = {
                "sample_id": sample.get("sample_id", str(idx)),
                "loss": float(metrics["loss"].detach().cpu()),
                "image_mse": float(metrics["image_mse"].detach().cpu()),
                "alpha_mse": float(metrics["alpha_mse"].detach().cpu()),
                "foreground_image_mse": float(metrics["foreground_image_mse"].detach().cpu()),
                "foreground_alpha_mse": float(metrics["foreground_alpha_mse"].detach().cpu()),
                "render_alpha_mean": float(metrics["render_alpha_mean"].detach().cpu()),
                "target_alpha_mean": float(metrics["target_alpha_mean"].detach().cpu()),
            }
            for key in ["foreground_l1", "dssim", "alpha_bce", "alpha_dice", "depth_l1"]:
                if key in metrics:
                    record[key] = float(metrics[key].detach().cpu())
            records.append(record)
            _write_eval_sheet(
                os.path.join(eval_dir, "%s_sample_%02d_%s.png" % (label, idx, record["sample_id"])),
                metrics,
                title=label + " ",
            )
            if idx == 0:
                _write_eval_sheet(os.path.join(eval_dir, "%s_sheet.png" % label), metrics, title=label + " ")
    model.train()
    if not records:
        return {"num_samples": 0}
    return {
        "num_samples": len(records),
        "mean_loss": float(np.mean([record["loss"] for record in records])),
        "mean_image_mse": float(np.mean([record["image_mse"] for record in records])),
        "mean_alpha_mse": float(np.mean([record["alpha_mse"] for record in records])),
        "mean_foreground_image_mse": float(np.mean([record["foreground_image_mse"] for record in records])),
        "mean_foreground_alpha_mse": float(np.mean([record["foreground_alpha_mse"] for record in records])),
        "mean_render_alpha": float(np.mean([record["render_alpha_mean"] for record in records])),
        "mean_target_alpha": float(np.mean([record["target_alpha_mean"] for record in records])),
        "records": records,
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    sys.path.insert(0, args.diffsplat_root)
    print(json.dumps({"event": "startup", "output_dir": args.output_dir, "diffsplat_root": args.diffsplat_root}), flush=True)
    if not args.skip_transformers_patch:
        print(json.dumps({"event": "patch_transformers_start"}), flush=True)
        patch_transformers_compat()
        print(json.dumps({"event": "patch_transformers_done"}), flush=True)
    else:
        print(json.dumps({"event": "patch_transformers_skipped"}), flush=True)
    print(json.dumps({"event": "patch_optional_start"}), flush=True)
    patch_optional_imports()
    print(json.dumps({"event": "patch_optional_done"}), flush=True)
    print(json.dumps({"event": "patch_rasterizer_start"}), flush=True)
    patch_gaussian_rasterizer_compat()
    print(json.dumps({"event": "patch_rasterizer_done"}), flush=True)
    print(json.dumps({"event": "compat_patched"}), flush=True)
    import torch
    from src.models import GSRecon
    from src.options import opt_dict
    from src.utils import util

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(json.dumps({"event": "imports_ready", "device": str(device), "cuda_available": bool(torch.cuda.is_available())}), flush=True)
    opt = opt_dict[args.opt_type]
    opt.chunk_size = 1
    opt.render_type = "default"
    opt.lpips_weight = 0.0
    opt.coord_weight = 0.0
    opt.normal_weight = 0.0
    print(json.dumps({"event": "construct_model", "opt_type": args.opt_type}), flush=True)
    model = GSRecon(opt).to(device)
    print(json.dumps({"event": "load_checkpoint", "weights": args.weights, "ckpt_iter": int(args.ckpt_iter)}), flush=True)
    model = util.load_ckpt(_checkpoint_dir(args.weights), args.ckpt_iter, None, model)
    if args.init_model_state:
        print(json.dumps({"event": "load_init_model_state", "path": args.init_model_state}), flush=True)
        state = torch.load(args.init_model_state, map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state), strict=True)
    model.to(device)
    model._latent_void_train_last_blocks = int(args.train_last_blocks)
    _set_trainable(model, args.trainable)
    num_trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(json.dumps({"event": "model_ready", "trainable": args.trainable, "num_trainable": num_trainable}), flush=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(args.lr))

    dataset = _load_json(args.patch_dataset)
    samples = [sample for sample in dataset.get("samples", []) if os.path.exists(sample.get("patch_manifest", ""))]
    if args.sample_id_contains:
        samples = [sample for sample in samples if args.sample_id_contains in str(sample.get("sample_id", ""))]
    if args.max_samples > 0:
        samples = samples[:int(args.max_samples)]
    if not samples:
        raise RuntimeError("no usable samples in patch dataset")
    print(json.dumps({"event": "data_ready", "num_samples": len(samples), "steps": int(args.steps)}), flush=True)
    rng = np.random.default_rng(args.seed)
    dtype = torch.float32
    losses = []
    eval_dir = ensure_dir(os.path.join(args.output_dir, "eval"))
    fixed_eval_count = min(int(args.fixed_eval_samples), len(samples))
    base_eval = _evaluate_fixed(model, samples, torch, device, args, dtype, eval_dir, "base", fixed_eval_count)
    print(json.dumps({"event": "base_eval", **{k: v for k, v in base_eval.items() if k != "records"}}), flush=True)
    best_eval_loss = float(base_eval.get("mean_loss", float("inf")))
    stale_eval_count = 0
    for step in range(int(args.steps)):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        step_metrics = None
        ids = rng.integers(0, len(samples), size=int(args.batch_size))
        for sample_idx in ids:
            view_rng = rng if args.random_view_subset else None
            batch = _load_sample(samples[int(sample_idx)], torch, device, int(args.max_views), view_rng)
            metrics = _forward_loss(model, batch, torch, args, dtype)
            loss = metrics["loss"] / float(args.batch_size)
            loss.backward()
            step_losses.append(float(metrics["loss"].detach().cpu()))
            step_metrics = metrics
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        record = {"step": step + 1, "loss": float(np.mean(step_losses))}
        if step_metrics is not None:
            for key in [
                "image_mse",
                "alpha_mse",
                "foreground_image_mse",
                "foreground_alpha_mse",
                "foreground_l1",
                "dssim",
                "alpha_bce",
                "alpha_dice",
                "source_opacity_bce",
                "source_opacity_dice",
                "source_rgb_l1",
                "depth_l1",
                "render_alpha_mean",
                "target_alpha_mean",
                "coord_mse",
                "normal_cosim",
            ]:
                if key in step_metrics:
                    record[key] = float(step_metrics[key].detach().cpu())
        losses.append(record)
        if args.log_interval > 0 and ((step + 1) % int(args.log_interval) == 0 or step == 0):
            print(json.dumps(record), flush=True)
        if step_metrics is not None and args.eval_interval > 0 and ((step + 1) % int(args.eval_interval) == 0 or step == 0):
            _write_eval_sheet(os.path.join(eval_dir, "step_%06d_sheet.png" % (step + 1)), step_metrics, title="step %d " % (step + 1))
        if args.fixed_eval_interval > 0 and ((step + 1) % int(args.fixed_eval_interval) == 0):
            fixed_eval = _evaluate_fixed(
                model,
                samples,
                torch,
                device,
                args,
                dtype,
                eval_dir,
                "fixed_step_%06d" % (step + 1),
                fixed_eval_count,
            )
            record["fixed_eval_loss"] = float(fixed_eval.get("mean_loss", float("inf")))
            print(json.dumps({"event": "fixed_eval", "step": step + 1, **{k: v for k, v in fixed_eval.items() if k != "records"}}), flush=True)
            if record["fixed_eval_loss"] + float(args.early_stop_min_delta) < best_eval_loss:
                best_eval_loss = record["fixed_eval_loss"]
                stale_eval_count = 0
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model_state_dict.pt"))
            else:
                stale_eval_count += 1
                if args.early_stop_patience > 0 and stale_eval_count >= int(args.early_stop_patience):
                    print(json.dumps({
                        "event": "early_stop",
                        "step": step + 1,
                        "best_eval_loss": best_eval_loss,
                        "stale_eval_count": stale_eval_count,
                    }), flush=True)
                    break

    model_path = os.path.join(args.output_dir, "gsrecon_scene_patch_finetuned.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "opt_type": args.opt_type,
        "trainable": args.trainable,
        "num_trainable": num_trainable,
        "weights": args.weights,
        "init_model_state": args.init_model_state,
        "ckpt_iter": int(args.ckpt_iter),
    }, model_path)
    final_eval = _evaluate_fixed(model, samples, torch, device, args, dtype, eval_dir, "finetuned", fixed_eval_count)
    print(json.dumps({"event": "final_eval", **{k: v for k, v in final_eval.items() if k != "records"}}), flush=True)
    if losses and step_metrics is not None:
        _write_eval_sheet(os.path.join(eval_dir, "last_training_sample_sheet.png"), step_metrics, title="last train ")
    status = {
        "ok": True,
        "patch_dataset": args.patch_dataset,
        "output_dir": args.output_dir,
        "device": str(device),
        "num_samples": len(samples),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "trainable": args.trainable,
        "train_last_blocks": int(args.train_last_blocks),
        "num_trainable": num_trainable,
        "initial_loss": losses[0]["loss"],
        "final_loss": losses[-1]["loss"],
        "model_path": model_path,
        "base_sheet": os.path.join(eval_dir, "base_sheet.png"),
        "final_sheet": os.path.join(eval_dir, "finetuned_sheet.png"),
        "last_training_sample_sheet": os.path.join(eval_dir, "last_training_sample_sheet.png"),
        "base_eval": base_eval,
        "final_eval": final_eval,
        "losses": losses,
    }
    write_json(os.path.join(args.output_dir, "finetune_status.json"), status)
    print(json.dumps({key: status[key] for key in ["ok", "device", "num_samples", "steps", "initial_loss", "final_loss", "model_path", "final_sheet"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
