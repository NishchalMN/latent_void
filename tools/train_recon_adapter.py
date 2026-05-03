#!/usr/bin/env python3
"""Train a lightweight reconstruction-adapter smoke model on patch targets."""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.io import ensure_dir, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-targets", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rgb-weight", type=float, default=1.0)
    parser.add_argument("--alpha-weight", type=float, default=0.2)
    parser.add_argument("--depth-weight", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _load_chw(path, channels=None):
    if not path or not os.path.exists(path):
        if channels is None:
            raise RuntimeError("missing array path")
        return np.zeros((channels, 1, 1), dtype=np.float32)
    array = np.load(path).astype(np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError("expected CHW/HW array at %s, got %s" % (path, array.shape))
    return array


def _sample_pairs(manifest):
    pairs = []
    for sample in manifest.get("samples", []):
        inputs = sample.get("input_views", [])
        targets = sample.get("targets", [])
        if not inputs or not targets:
            continue
        source = inputs[0]
        for target in targets:
            pairs.append({"sample_id": sample.get("sample_id"), "source": source, "target": target})
    return pairs


def _load_pair(pair, torch, device):
    source = pair["source"]
    target = pair["target"]
    source_rgb = _load_chw(source.get("rgb_npy"), channels=3)
    source_mask = _load_chw(source.get("mask_npy"), channels=1)
    if source_mask.max() > 1.0:
        source_mask = source_mask / 255.0
    source_depth = _load_chw(source.get("depth_npy", ""), channels=1)
    source_normal = _load_chw(source.get("normal_npy", ""), channels=3)
    source_coord = _load_chw(source.get("coord_npy", ""), channels=3)
    x = np.concatenate([source_rgb, source_mask[:1], source_depth[:1], source_normal[:3], source_coord[:3]], axis=0)

    target_rgb = _load_chw(target.get("rgb_npy"), channels=3)
    target_alpha = _load_chw(target.get("alpha_npy"), channels=1)
    if target_alpha.max() > 1.0:
        target_alpha = target_alpha / 255.0
    target_depth = _load_chw(target.get("depth_npy"), channels=1)
    y = np.concatenate([target_rgb, target_alpha[:1], target_depth[:1]], axis=0)
    return torch.from_numpy(x[None]).to(device), torch.from_numpy(y[None]).to(device)


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    manifest = _load_json(args.teacher_targets)
    pairs = _sample_pairs(manifest)
    if not pairs:
        raise RuntimeError("teacher target manifest has no trainable pairs")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    first_x, first_y = _load_pair(pairs[0], torch, device)
    in_channels = int(first_x.shape[1])
    out_channels = int(first_y.shape[1])
    model = nn.Sequential(
        nn.Conv2d(in_channels, args.hidden_channels, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(args.hidden_channels, args.hidden_channels, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(args.hidden_channels, args.hidden_channels, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(args.hidden_channels, out_channels, 3, padding=1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    rng = np.random.default_rng(args.seed)
    losses = []
    for step in range(int(args.steps)):
        batch_ids = rng.integers(0, len(pairs), size=int(args.batch_size))
        xs, ys = [], []
        for pair_idx in batch_ids:
            x, y = _load_pair(pairs[int(pair_idx)], torch, device)
            xs.append(x)
            ys.append(y)
        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        pred = model(x)
        if pred.shape[-2:] != y.shape[-2:]:
            pred = F.interpolate(pred, size=y.shape[-2:], mode="bilinear", align_corners=False)
        rgb_loss = F.mse_loss(torch.sigmoid(pred[:, :3]), y[:, :3])
        alpha_loss = F.mse_loss(torch.sigmoid(pred[:, 3:4]), y[:, 3:4])
        depth_loss = F.l1_loss(pred[:, 4:5], y[:, 4:5])
        loss = args.rgb_weight * rgb_loss + args.alpha_weight * alpha_loss + args.depth_weight * depth_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append({
            "step": step,
            "loss": float(loss.detach().cpu()),
            "rgb_loss": float(rgb_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "depth_loss": float(depth_loss.detach().cpu()),
        })
        if args.log_interval > 0 and ((step + 1) % int(args.log_interval) == 0 or step == 0):
            latest = dict(losses[-1])
            latest["step"] = step + 1
            latest["steps"] = int(args.steps)
            print(json.dumps(latest), flush=True)
    ensure_dir(args.output_dir)
    model_path = os.path.join(args.output_dir, "recon_adapter_smoke.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "hidden_channels": int(args.hidden_channels),
        "note": "Smoke reconstruction adapter. Use this gate before wiring full GSRecon fine-tuning.",
    }, model_path)
    status = {
        "ok": True,
        "teacher_targets": args.teacher_targets,
        "output_dir": args.output_dir,
        "device": str(device),
        "num_pairs": len(pairs),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "initial_loss": losses[0]["loss"],
        "final_loss": losses[-1]["loss"],
        "model_path": model_path,
        "losses": losses,
        "contract": [
            "GSVAE remains frozen for this stage.",
            "Loss terms mirror the intended GSRecon adaptation gates: RGB render, alpha, and depth consistency.",
            "This smoke adapter validates data plumbing before heavy DiffSplat module fine-tuning.",
        ],
    }
    write_json(os.path.join(args.output_dir, "train_recon_adapter_status.json"), status)
    print(json.dumps({key: status[key] for key in ["ok", "device", "num_pairs", "initial_loss", "final_loss", "model_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
