#!/usr/bin/env python3
"""Apply a trained MaskedLatentDenoiser checkpoint to GSVAE latent + void mask."""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from latent_void.io import load_array, save_array
from latent_void.latent import expand_mask_to_latent

from masked_latent_denoiser_nn import MaskedLatentDenoiser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--weights", required=True, help="masked_latent_denoiser.pt from train_masked_latent_denoiser")
    parser.add_argument(
        "--refine-steps",
        type=int,
        default=8,
        help="Masked refinement passes; maps to pipeline {latent_inpaint_iterations}",
    )
    parser.add_argument("--noise-min", type=float, default=0.02)
    parser.add_argument("--noise-max", type=float, default=0.5)
    parser.add_argument("--context-fill", choices=["mean", "zero"], default="mean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--status-path", default="")
    return parser.parse_args()


def _to_bchw(array):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError("expected latent CHW or BCHW, got shape %s" % (array.shape,))
    return array


def _mask_to_torch(mask_np, latent_bchw, torch):
    if mask_np.ndim == 2:
        mask_tf = torch.from_numpy(mask_np.astype(np.float32))[None, None, ...]
    elif mask_np.ndim == 3:
        mask_tf = torch.from_numpy(mask_np.astype(np.float32))[:, None, ...]
    else:
        raise ValueError("expected mask shape [H,W] or [B,H,W], got %s" % (mask_np.shape,))
    batch = latent_bchw.shape[0]
    if mask_tf.shape[0] == 1 and batch > 1:
        mask_tf = mask_tf.expand(batch, -1, -1, -1)
    if mask_tf.shape[0] != batch:
        raise ValueError("mask batch %d does not match latent batch %d" % (mask_tf.shape[0], batch))
    if mask_tf.shape[-2:] != latent_bchw.shape[-2:]:
        raise ValueError("mask spatial %s mismatches latent %s" % (mask_tf.shape[-2:], latent_bchw.shape[-2:]))
    return mask_tf


def _masked_latent(latent_np, expanded_mask_bool, context_fill):
    latent_np = np.asarray(latent_np, dtype=np.float32)
    expanded = expanded_mask_bool
    output = latent_np.copy()
    channels_first = np.moveaxis(output, -3, 0)
    source_channels = np.moveaxis(latent_np, -3, 0)
    for channel_idx in range(channels_first.shape[0]):
        channel = channels_first[channel_idx]
        source = source_channels[channel_idx]
        if context_fill == "mean" and (~expanded).any():
            fill = float(source[~expanded].mean())
        else:
            fill = 0.0
        channel[expanded] = fill
    return output


def _training_style_step(model, latent_bchw, mask_b11hw, masked_bchw, source_bchw, sigma_vec, torch):
    noisy = latent_bchw + torch.randn_like(latent_bchw) * sigma_vec[:, None, None, None] * mask_b11hw
    denoised = model(noisy, mask_b11hw, sigma_vec)
    pred = denoised * mask_b11hw + masked_bchw * (1.0 - mask_b11hw)
    pred = pred * mask_b11hw + source_bchw * (1.0 - mask_b11hw)
    return pred


def main():
    args = parse_args()
    import torch

    if int(args.seed) >= 0:
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
    weights = args.weights
    if not os.path.isfile(weights):
        raise SystemExit("weights not found: %s" % weights)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    latent = load_array(args.latent_path).astype(np.float32)
    mask_raw = np.load(args.mask_path)
    latent_bnp = _to_bchw(latent)
    expanded_bool = expand_mask_to_latent(mask_raw, latent_bnp)

    ckpt = torch.load(weights, map_location="cpu")
    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise SystemExit("checkpoint missing model_state_dict: %s" % weights)
    latent_channels = int(ckpt.get("latent_channels") or latent_bnp.shape[1])
    base_channels = int(ckpt.get("base_channels") or 96)
    if latent_channels != latent_bnp.shape[1]:
        raise SystemExit("checkpoint latent_channels=%d does not match file C=%d" % (latent_channels, latent_bnp.shape[1]))

    model = MaskedLatentDenoiser(latent_channels, base_channels).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    source_tf = torch.from_numpy(latent_bnp).to(device)
    masked_np = _masked_latent(latent_bnp, expanded_bool, args.context_fill)
    masked_tf = torch.from_numpy(masked_np.astype(np.float32)).to(device)
    mask_tf = _mask_to_torch(expanded_bool.astype(np.float32), latent_bnp, torch).to(device)

    refine_steps = max(1, int(args.refine_steps))
    sigma_min = float(args.noise_min)
    sigma_max = float(args.noise_max)

    batch = latent_bnp.shape[0]
    current = masked_tf.detach().clone()

    with torch.inference_mode():
        for idx in range(refine_steps):
            if refine_steps <= 1:
                t = 0.0
            else:
                t = idx / float(refine_steps - 1)
            sigma_val = sigma_max + (sigma_min - sigma_max) * t
            sigma_vec = torch.full((batch,), sigma_val, device=device, dtype=current.dtype)
            current = _training_style_step(model, current, mask_tf, masked_tf, source_tf, sigma_vec, torch)

    output_np = current.detach().cpu().numpy()
    if latent.ndim == 3:
        output_np = output_np[0]

    save_array(args.output_path, output_np.astype(np.float32))

    expanded_mask_float = expanded_bool.astype(np.float32)
    status = {
        "ok": True,
        "latent_path": args.latent_path,
        "mask_path": args.mask_path,
        "weights": weights,
        "output_path": args.output_path,
        "latent_channels": latent_channels,
        "base_channels": base_channels,
        "refine_steps": refine_steps,
        "noise_min": sigma_min,
        "noise_max": sigma_max,
        "context_fill": args.context_fill,
        "device": str(device),
        "latent_shape": list(latent_bnp.shape),
        "masked_void_cells": float(expanded_mask_float.sum()),
    }
    status_path = args.status_path or os.path.join(os.path.dirname(args.output_path), "masked_denoiser_inpaint_status.json")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w") as fp:
        json.dump(status, fp, indent=2)
        fp.write("\n")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
