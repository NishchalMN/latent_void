#!/usr/bin/env python3
"""Download DiffSplat auxiliary Diffusers VAE snapshots for offline H100 jobs."""

import argparse
import os

from huggingface_hub import snapshot_download


MODELS = [
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl-vae-fp16-fix"),
    ("madebyollin/taesdxl", "taesdxl"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/diffsplat_aux")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for repo_id, name in MODELS:
        local_dir = os.path.abspath(os.path.join(args.output_dir, name))
        print("[diffsplat-aux] downloading %s -> %s" % (repo_id, local_dir), flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        config_path = os.path.join(local_dir, "config.json")
        if not os.path.exists(config_path):
            raise RuntimeError("downloaded snapshot is missing config.json: %s" % local_dir)
    print("[diffsplat-aux] ready under %s" % os.path.abspath(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
