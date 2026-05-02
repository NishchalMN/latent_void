#!/usr/bin/env python3
"""Run SAM 3 text-prompt segmentation for a multi-view manifest.

This is intentionally a thin wrapper around the official SAM 3 package. It
writes one binary `.npy` mask per input view plus a manifest with scores/boxes.
Checkpoint downloading is handled by SAM 3/Hugging Face, so users must run
`hf auth login` and have access to the Meta SAM 3 checkpoints before this stage.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _load_sam3(sam3_root, device, confidence_threshold, checkpoint_path=""):
    sys.path.insert(0, sam3_root)
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(
        device=device,
        checkpoint_path=checkpoint_path or None,
        load_from_HF=not bool(checkpoint_path),
    )
    model.eval()
    return torch, Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)


def _best_mask(output, score_threshold):
    masks = output.get("masks")
    scores = output.get("scores")
    boxes = output.get("boxes")
    if masks is None or len(masks) == 0:
        return None, None, None
    if hasattr(scores, "detach"):
        score_values = scores.detach().float().cpu().numpy()
    else:
        score_values = np.asarray(scores, dtype=np.float32)
    best = int(score_values.argmax()) if score_values.size else 0
    if score_values.size and float(score_values[best]) < score_threshold:
        return None, float(score_values[best]), None
    mask = masks[best]
    if hasattr(mask, "detach"):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask)
    while mask.ndim > 2:
        mask = mask[0]
    box = None
    if boxes is not None and len(boxes) > best:
        box_value = boxes[best]
        if hasattr(box_value, "detach"):
            box_value = box_value.detach().cpu().numpy()
        box = np.asarray(box_value).tolist()
    score = float(score_values[best]) if score_values.size else None
    return mask > 0.5, score, box


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.manifest, "r") as handle:
        manifest = json.load(handle)

    torch, processor = _load_sam3(args.sam3_root, args.device, args.score_threshold, args.checkpoint_path)
    results = []
    with torch.inference_mode():
        for idx, view in enumerate(manifest.get("views", [])):
            image_path = view["image_path"]
            image = Image.open(image_path).convert("RGB")
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=args.prompt)
            mask, score, box = _best_mask(output, args.score_threshold)
            mask_path = os.path.join(args.output_dir, "%04d.npy" % idx)
            if mask is None:
                mask = np.zeros((image.height, image.width), dtype=np.uint8)
            np.save(mask_path, mask.astype(np.uint8))
            results.append({
                "view_id": view.get("view_id"),
                "image_path": image_path,
                "mask_path": mask_path,
                "score": score,
                "box": box,
            })

    with open(os.path.join(args.output_dir, "sam3_results.json"), "w") as handle:
        json.dump({"prompt": args.prompt, "results": results}, handle, indent=2)


if __name__ == "__main__":
    main()
