#!/usr/bin/env python3
"""Context-only native latent inpainting baseline."""

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from latent_void.io import load_array, save_array
from latent_void.latent import context_inpaint_latent, expand_mask_to_latent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-path", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--status-path", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    latent = load_array(args.latent_path)
    mask = load_array(args.mask_path).astype(bool)
    expanded_mask = expand_mask_to_latent(mask, latent)
    output = context_inpaint_latent(latent, mask, iterations=args.iterations)
    original_channels = np.moveaxis(latent, -3, 0)
    output_channels = np.moveaxis(output, -3, 0)
    for channel_idx in range(output_channels.shape[0]):
        if not np.allclose(output_channels[channel_idx][~expanded_mask], original_channels[channel_idx][~expanded_mask]):
            raise RuntimeError("unmasked latent cells changed in channel %d" % channel_idx)
    save_array(args.output_path, output)
    status = {
        "ok": True,
        "latent_path": args.latent_path,
        "mask_path": args.mask_path,
        "output_path": args.output_path,
        "iterations": int(args.iterations),
        "latent_shape": list(latent.shape),
        "mask_shape": list(mask.shape),
        "expanded_mask_shape": list(expanded_mask.shape),
        "masked_cells": int(expanded_mask.sum()),
    }
    status_path = args.status_path or os.path.join(os.path.dirname(args.output_path), "context_inpaint_status.json")
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    with open(status_path, "w") as handle:
        json.dump(status, handle, indent=2)
        handle.write("\n")
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
