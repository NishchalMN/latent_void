"""Create quick visual baseline sheets from an Inpaint360GS scene.

This is a lightweight baseline for human progress review. It does not claim to
be the native DiffSplat latent path; it shows object-present source views,
masked void views, and available object-free/evaluation frames from the dataset.
"""

import argparse
import json
import os
import re

from PIL import Image, ImageDraw, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--image-subdir", default="images_4")
    parser.add_argument("--mask-subdir", default="unseen_mask")
    parser.add_argument("--thumb-width", type=int, default=480)
    return parser.parse_args()


def _natural_key(path):
    name = os.path.basename(path)
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def _list_images(directory, prefix):
    if not os.path.isdir(directory):
        return []
    return sorted(
        [
            os.path.join(directory, name)
            for name in os.listdir(directory)
            if name.startswith(prefix) and name.lower().endswith((".jpg", ".jpeg", ".png"))
        ],
        key=_natural_key,
    )


def _resize(image, width):
    if image.width == width:
        return image
    height = int(round(image.height * (width / float(image.width))))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _label(image, text):
    pad = 28
    canvas = Image.new("RGB", (image.width, image.height + pad), "white")
    canvas.paste(image.convert("RGB"), (0, pad))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 6), text, fill=(0, 0, 0))
    return canvas


def _write_grid(path, images, columns=4):
    if not images:
        raise RuntimeError("no images to write: %s" % path)
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
    for idx, image in enumerate(images):
        x = (idx % columns) * cell_w
        y = (idx // columns) * cell_h
        canvas.paste(image, (x, y))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def _mask_for(test_image_path, mask_dir):
    stem = os.path.splitext(os.path.basename(test_image_path))[0]
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(mask_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def _make_voided(image, mask):
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    binary = mask.point(lambda value: 255 if value > 127 else 0)
    voided = image.convert("RGB").copy()
    black = Image.new("RGB", image.size, (15, 15, 15))
    voided.paste(black, mask=binary)
    edge = binary.filter(ImageFilter.FIND_EDGES).point(lambda value: 255 if value > 8 else 0)
    red = Image.new("RGB", image.size, (255, 40, 40))
    voided.paste(red, mask=edge)
    return voided


def _make_overlay(image, mask):
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", image.size, (255, 0, 0, 95))
    result = image.convert("RGBA")
    result.alpha_composite(Image.composite(overlay, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask))
    return result.convert("RGB")


def main():
    args = parse_args()
    scene_dir = os.path.abspath(args.scene_dir)
    image_dir = os.path.join(scene_dir, args.image_subdir)
    mask_dir = os.path.join(scene_dir, args.mask_subdir)
    os.makedirs(args.output_dir, exist_ok=True)

    source_paths = _list_images(image_dir, "IMG_")[: int(args.num_views)]
    test_paths = _list_images(image_dir, "test_IMG_")[: int(args.num_views)]
    if not source_paths:
        raise RuntimeError("no source IMG_* files found in %s" % image_dir)
    if not test_paths:
        raise RuntimeError("no test_IMG_* files found in %s" % image_dir)

    source_tiles = []
    for path in source_paths:
        image = _resize(Image.open(path).convert("RGB"), int(args.thumb_width))
        source_tiles.append(_label(image, os.path.basename(path)))

    void_tiles = []
    inpaint_tiles = []
    overlay_tiles = []
    comparison_tiles = []
    missing_masks = []
    for path in test_paths:
        image = _resize(Image.open(path).convert("RGB"), int(args.thumb_width))
        mask_path = _mask_for(path, mask_dir)
        if mask_path is None:
            missing_masks.append(os.path.basename(path))
            mask = Image.new("L", image.size, 0)
        else:
            mask = Image.open(mask_path)
        voided = _make_voided(image, mask)
        overlay = _make_overlay(image, mask)
        label = os.path.basename(path)
        void_tiles.append(_label(voided, label + " void mask"))
        inpaint_tiles.append(_label(image, label + " object-free"))
        overlay_tiles.append(_label(overlay, label + " mask overlay"))
        comparison_tiles.extend([
            _label(voided, label + " void"),
            _label(image, label + " inpainted/GT"),
        ])

    outputs = {
        "source_views": os.path.join(args.output_dir, "source_views.png"),
        "voided_views": os.path.join(args.output_dir, "voided_views.png"),
        "inpainted_views": os.path.join(args.output_dir, "inpainted_views.png"),
        "mask_overlay_views": os.path.join(args.output_dir, "mask_overlay_views.png"),
        "void_vs_inpainted": os.path.join(args.output_dir, "void_vs_inpainted.png"),
    }
    _write_grid(outputs["source_views"], source_tiles, columns=4)
    _write_grid(outputs["voided_views"], void_tiles, columns=4)
    _write_grid(outputs["inpainted_views"], inpaint_tiles, columns=4)
    _write_grid(outputs["mask_overlay_views"], overlay_tiles, columns=4)
    _write_grid(outputs["void_vs_inpainted"], comparison_tiles, columns=2)

    status = {
        "ok": True,
        "baseline_type": "image_level_dataset_visualization",
        "is_scene_level_3dgs_render": False,
        "is_native_diffsplat_latent_inpaint": False,
        "scene_dir": scene_dir,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "source_images": source_paths,
        "test_images": test_paths,
        "missing_masks": missing_masks,
        "outputs": outputs,
        "note": (
            "This visual baseline uses official Inpaint360GS object-present source "
            "views and object-free/evaluation test frames. It is an image-level "
            "progress/debug baseline, not a rendered 3DGS model and not the native "
            "DiffSplat latent inpainting method."
        ),
    }
    status_path = os.path.join(args.output_dir, "baseline_status.json")
    with open(status_path, "w") as handle:
        json.dump(status, handle, indent=2)
    print(json.dumps({"ok": True, "status": status_path, "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
