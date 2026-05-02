#!/usr/bin/env python3
"""Check whether the current Hugging Face auth can access SAM 3 weights."""

import argparse

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="facebook/sam3")
    parser.add_argument("--local-dir", default="checkpoints/sam3")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        result = snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            allow_patterns=[
                "*.json",
                "*.yaml",
                "*.txt",
                "*.safetensors",
                "*.pt",
                "*.bin",
            ],
            dry_run=not args.download,
        )
    except Exception as exc:
        print("SAM 3 checkpoint access failed.")
        print(type(exc).__name__ + ": " + str(exc))
        print()
        print("Needed from user:")
        print("1. Request/accept access to https://huggingface.co/%s" % args.repo_id)
        print("2. On Zaratan, run: hf auth login")
        print("3. Re-run: python scripts/check_sam3_access.py")
        return 1

    print("SAM 3 checkpoint access looks OK.")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
