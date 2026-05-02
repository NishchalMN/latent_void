#!/usr/bin/env python3
"""Download Marigold model snapshots for offline Slurm jobs."""

import argparse
import os

from huggingface_hub import snapshot_download


MODELS = [
    ("prs-eth/marigold-depth-v1-1", "depth-v1-1"),
    ("prs-eth/marigold-normals-v1-1", "normals-v1-1"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/marigold")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for repo_id, name in MODELS:
        local_dir = os.path.abspath(os.path.join(args.output_dir, name))
        print("[marigold] downloading %s -> %s" % (repo_id, local_dir), flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
    print("[marigold] ready under %s" % os.path.abspath(args.output_dir), flush=True)


if __name__ == "__main__":
    main()
