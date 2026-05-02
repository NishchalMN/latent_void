#!/usr/bin/env python3
"""Placeholder/export contract for DiffSplat GSRecon scene encoding.

DiffSplat exposes `GSRecon` in training scripts, but the upstream repo does not
currently ship a simple "multi-view scene -> gaussians.npz + latent.npy" CLI.
This script documents and checks the contract expected by `latent_void`.

The next implementation step is to instantiate DiffSplat's `GSRecon` and
`GSAutoencoderKL` here once compatible checkpoints and a real model environment
are present. Until then, this script fails loudly instead of producing fake
model outputs.
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffsplat-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    status = {
        "ok": False,
        "reason": "DiffSplat GSRecon exporter is not implemented yet.",
        "expected_outputs": [
            "gaussians.npz with uvs [V,N,2] and visibility [V,N]",
            "latent.npy with GSVAE splat latents",
        ],
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "gsrecon_export_status.json"), "w") as handle:
        json.dump(status, handle, indent=2)
    print(json.dumps(status, indent=2), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
