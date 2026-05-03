#!/usr/bin/env python3
"""Train a masked latent denoiser with hard unmasked-cell clamping."""

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
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--noise-min", type=float, default=0.02)
    parser.add_argument("--noise-max", type=float, default=0.5)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _to_tensor(array, torch, device):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError("expected latent-like CHW/BCHW array, got %s" % (array.shape,))
    return torch.from_numpy(array).to(device)


def _load_sample(sample, torch, device):
    source = _to_tensor(np.load(sample["source_latent"]), torch, device)
    masked = _to_tensor(np.load(sample["masked_latent"]), torch, device)
    mask = np.load(sample["latent_mask"]).astype(np.float32)
    if mask.ndim == 2:
        mask = mask[None, None, ...]
    elif mask.ndim == 3:
        mask = mask[:, None, ...]
    else:
        raise ValueError("expected mask with 2 or 3 dims, got %s" % (mask.shape,))
    return source, masked, torch.from_numpy(mask).to(device)


def _load_compatible_weights(torch, model, path):
    if not path:
        return {"loaded": False, "reason": "no init checkpoint provided"}
    if not os.path.exists(path):
        return {"loaded": False, "reason": "checkpoint does not exist"}
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    current = model.state_dict()
    compatible = {key: value for key, value in state.items() if key in current and tuple(current[key].shape) == tuple(value.shape)}
    current.update(compatible)
    model.load_state_dict(current)
    return {"loaded": bool(compatible), "path": path, "num_tensors": len(compatible)}


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )

        def forward(self, x):
            return F.silu(x + self.net(x))

    class MaskedLatentDenoiser(nn.Module):
        def __init__(self, latent_channels, base_channels):
            super().__init__()
            self.in_proj = nn.Conv2d(latent_channels + 2, base_channels, 3, padding=1)
            self.down = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
            self.mid = nn.Sequential(ResidualBlock(base_channels), ResidualBlock(base_channels))
            self.up = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)
            self.out = nn.Sequential(ResidualBlock(base_channels), nn.Conv2d(base_channels, latent_channels, 3, padding=1))

        def forward(self, latent, mask, sigma):
            sigma_map = sigma[:, None, None, None].expand(mask.shape[0], 1, mask.shape[-2], mask.shape[-1])
            h = self.in_proj(torch.cat([latent, mask, sigma_map], dim=1))
            skip = h
            h = self.down(h)
            h = self.mid(h)
            h = self.up(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.out(h + skip)

    torch.manual_seed(args.seed)
    manifest = _load_json(args.dataset_manifest)
    samples = manifest.get("samples", [])
    if not samples:
        raise RuntimeError("dataset manifest has no samples")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    first_source, _, _ = _load_sample(samples[0], torch, device)
    latent_channels = int(first_source.shape[1])
    model = MaskedLatentDenoiser(latent_channels, int(args.base_channels)).to(device)
    init_status = _load_compatible_weights(torch, model, args.init_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    rng = np.random.default_rng(args.seed)
    losses = []
    for step in range(int(args.steps)):
        batch_ids = rng.integers(0, len(samples), size=int(args.batch_size))
        sources, maskeds, masks = [], [], []
        for sample_idx in batch_ids:
            source, masked, mask = _load_sample(samples[int(sample_idx)], torch, device)
            sources.append(source)
            maskeds.append(masked)
            masks.append(mask)
        source = torch.cat(sources, dim=0)
        masked = torch.cat(maskeds, dim=0)
        mask = torch.cat(masks, dim=0)
        sigma = torch.empty(source.shape[0], device=device).uniform_(float(args.noise_min), float(args.noise_max))
        noisy = masked + torch.randn_like(masked) * sigma[:, None, None, None] * mask
        denoised_delta = model(noisy, mask, sigma)
        pred = denoised_delta * mask + masked * (1.0 - mask)
        pred = pred * mask + source * (1.0 - mask)
        denom = mask.sum() * source.shape[1] + 1e-6
        loss = (((pred - source) ** 2) * mask).sum() / denom
        context_error = (((pred - source) ** 2) * (1.0 - mask)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append({"step": step, "loss": float(loss.detach().cpu()), "context_error": float(context_error.detach().cpu())})
        if args.log_interval > 0 and ((step + 1) % int(args.log_interval) == 0 or step == 0):
            print(
                json.dumps({
                    "step": step + 1,
                    "steps": int(args.steps),
                    "loss": losses[-1]["loss"],
                    "context_error": losses[-1]["context_error"],
                }),
                flush=True,
            )
    ensure_dir(args.output_dir)
    model_path = os.path.join(args.output_dir, "masked_latent_denoiser.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "latent_channels": latent_channels,
        "base_channels": int(args.base_channels),
        "init_status": init_status,
    }, model_path)
    status = {
        "ok": True,
        "dataset_manifest": args.dataset_manifest,
        "output_dir": args.output_dir,
        "device": str(device),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "initial_loss": losses[0]["loss"],
        "final_loss": losses[-1]["loss"],
        "final_context_error": losses[-1]["context_error"],
        "model_path": model_path,
        "init_status": init_status,
        "losses": losses,
        "contract": "Unmasked cells are clamped from the source latent during training; only masked cells drive the objective.",
    }
    write_json(os.path.join(args.output_dir, "train_masked_latent_denoiser_status.json"), status)
    print(json.dumps({key: status[key] for key in ["ok", "device", "initial_loss", "final_loss", "final_context_error", "model_path"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
