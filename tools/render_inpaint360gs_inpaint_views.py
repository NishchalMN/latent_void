"""Render an Inpaint360GS 3DGS model on object-free evaluation camera poses."""

import argparse
import json
import os
import sys
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inpaint360gs-root", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=8)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=480)
    return parser.parse_args()


def _install_paths(root):
    root = os.path.abspath(root)
    gs_root = os.path.join(root, "gaussian_splatting")
    for path in (gs_root, root):
        if path not in sys.path:
            sys.path.insert(0, path)


def _tensor_to_image(tensor):
    array = (
        tensor.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .byte()
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def _resize(image, width):
    if image.width == width:
        return image
    height = int(round(image.height * (width / float(image.width))))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _label(image, text):
    pad = 28
    canvas = Image.new("RGB", (image.width, image.height + pad), "white")
    canvas.paste(image.convert("RGB"), (0, pad))
    ImageDraw.Draw(canvas).text((5, 6), text, fill=(0, 0, 0))
    return canvas


def _write_grid(path, images, columns=4):
    if not images:
        raise RuntimeError("no images to write: %s" % path)
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (cell_w * columns, cell_h * rows), "white")
    for idx, image in enumerate(images):
        canvas.paste(image, ((idx % columns) * cell_w, (idx // columns) * cell_h))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def _mask_for(source_path, image_name):
    mask_dir = os.path.join(source_path, "unseen_mask")
    stem = os.path.splitext(os.path.basename(image_name))[0]
    for ext in (".png", ".jpg", ".jpeg"):
        path = os.path.join(mask_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def _voided(image, mask):
    mask = mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
    binary = mask.point(lambda value: 255 if value > 127 else 0)
    result = image.convert("RGB").copy()
    result.paste(Image.new("RGB", image.size, (15, 15, 15)), mask=binary)
    edge = binary.filter(ImageFilter.FIND_EDGES).point(lambda value: 255 if value > 8 else 0)
    result.paste(Image.new("RGB", image.size, (255, 40, 40)), mask=edge)
    return result


def main():
    args = parse_args()
    _install_paths(args.inpaint360gs_root)

    import torch
    from scene import Scene, GaussianModel
    from gaussian_splatting.gaussian_renderer import render

    source_path = os.path.abspath(args.source_path)
    model_path = os.path.abspath(args.model_path)
    output_dir = os.path.abspath(args.output_dir)
    render_dir = os.path.join(output_dir, "renders")
    gt_dir = os.path.join(output_dir, "gt")
    void_dir = os.path.join(output_dir, "voided")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(void_dir, exist_ok=True)

    dataset = SimpleNamespace(
        sh_degree=3,
        source_path=source_path,
        model_path=model_path,
        images="images",
        resolution=args.resolution,
        white_background=False,
        train_test_exp=False,
        data_device="cuda",
        eval=True,
        init_mode="sparse",
        train_distill=False,
        vanilla_3dgs_path="",
        object_path="object_mask",
        n_views=100,
        random_init=False,
        train_split=False,
    )
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        views = scene.getInpaintCameras()[: args.max_views]
        rows = []
        source_tiles = []
        void_tiles = []
        target_tiles = []
        comparison_tiles = []
        missing_masks = []

        for idx, view in enumerate(views):
            result = render(view, gaussians, pipeline, background, separate_sh=False)["render"]
            rendered = _tensor_to_image(result)
            target = _tensor_to_image(view.original_image[0:3, :, :])
            image_name = view.image_name
            render_path = os.path.join(render_dir, "%05d_%s.png" % (idx, image_name))
            target_path = os.path.join(gt_dir, "%05d_%s.png" % (idx, image_name))
            rendered.save(render_path)
            target.save(target_path)

            mask_path = _mask_for(source_path, image_name)
            if mask_path is None:
                mask = Image.new("L", rendered.size, 0)
                missing_masks.append(image_name)
            else:
                mask = Image.open(mask_path)
            void = _voided(rendered, mask)
            void_path = os.path.join(void_dir, "%05d_%s.png" % (idx, image_name))
            void.save(void_path)

            rows.append({
                "image_name": image_name,
                "render": render_path,
                "target": target_path,
                "voided": void_path,
                "mask": mask_path,
            })
            source_tiles.append(_label(_resize(rendered, args.thumb_width), image_name + " 3DGS before"))
            void_tiles.append(_label(_resize(void, args.thumb_width), image_name + " 3DGS void mask"))
            target_tiles.append(_label(_resize(target, args.thumb_width), image_name + " object-free target"))
            comparison_tiles.extend([
                _label(_resize(rendered, args.thumb_width), image_name + " before render"),
                _label(_resize(void, args.thumb_width), image_name + " voided render"),
                _label(_resize(target, args.thumb_width), image_name + " target"),
            ])

    outputs = {
        "source_render_views": os.path.join(output_dir, "source_render_views.png"),
        "voided_render_views": os.path.join(output_dir, "voided_render_views.png"),
        "inpaint_target_views": os.path.join(output_dir, "inpaint_target_views.png"),
        "render_void_target_comparison": os.path.join(output_dir, "render_void_target_comparison.png"),
    }
    _write_grid(outputs["source_render_views"], source_tiles, columns=4)
    _write_grid(outputs["voided_render_views"], void_tiles, columns=4)
    _write_grid(outputs["inpaint_target_views"], target_tiles, columns=4)
    _write_grid(outputs["render_void_target_comparison"], comparison_tiles, columns=3)

    status = {
        "ok": True,
        "baseline_type": "scene_level_3dgs_render_on_inpaint360gs_inpaint_cameras",
        "is_true_3d_removal": False,
        "is_true_3d_inpaint": False,
        "source_path": source_path,
        "model_path": model_path,
        "iteration": scene.loaded_iter,
        "num_views": len(rows),
        "missing_masks": missing_masks,
        "views": rows,
        "outputs": outputs,
        "note": (
            "Before images are true 3DGS renders. Voided images apply the official "
            "2D unseen masks to those renders. Targets are official object-free "
            "dataset frames; they are not generated by this 3DGS model."
        ),
    }
    status_path = os.path.join(output_dir, "render_inpaint_views_status.json")
    with open(status_path, "w") as handle:
        json.dump(status, handle, indent=2)
    print(json.dumps({"ok": True, "status": status_path, "outputs": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
