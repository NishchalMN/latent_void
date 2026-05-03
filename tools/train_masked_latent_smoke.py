#!/usr/bin/env python3
"""Tiny masked latent reconstruction smoke trainer.

This is not the final DiffSplat denoiser. It verifies that generated masked
latent samples can drive a trainable H100 adaptation loop with the invariant
that only masked cells contribute to the reconstruction objective.
"""

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
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _to_tensor(array, torch, device):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim == 4:
        return torch.from_numpy(array).to(device)
    raise ValueError("expected latent-like array with 3 or 4 dims, got %s" % (array.shape,))


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


def main():
    args = parse_args()
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(args.seed)
    manifest = _load_json(args.dataset_manifest)
    samples = manifest.get("samples", [])
    if not samples:
        raise RuntimeError("dataset manifest has no samples")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    first_source, first_masked, first_mask = _load_sample(samples[0], torch, device)
    channels = int(first_source.shape[1])
    model = nn.Sequential(
        nn.Conv2d(channels + 1, args.hidden_channels, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(args.hidden_channels, args.hidden_channels, 3, padding=1),
        nn.SiLU(),
        nn.Conv2d(args.hidden_channels, channels, 3, padding=1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    losses = []
    rng = np.random.default_rng(args.seed)
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
        pred_delta = model(torch.cat([masked, mask], dim=1))
        pred = pred_delta * mask + masked * (1.0 - mask)
        denom = mask.sum() * source.shape[1] + 1e-6
        loss = (((pred - source) ** 2) * mask).sum() / denom
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    ensure_dir(args.output_dir)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "masked_latent_smoke_model.pt"))
    status = {
        "ok": True,
        "dataset_manifest": args.dataset_manifest,
        "output_dir": args.output_dir,
        "device": str(device),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "losses": losses,
        "note": "Smoke trainer only; final model should be a DiffSplat/PixArt initialized masked latent denoiser.",
    }
    write_json(os.path.join(args.output_dir, "train_status.json"), status)
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
