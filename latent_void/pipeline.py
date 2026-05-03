import os

from latent_void.config import get_nested, validate_config
from latent_void.datasets import Inpaint360GSDataset
from latent_void.external import run_command
from latent_void.io import ensure_dir, write_json


def run_dirs(config):
    output_dir = get_nested(config, "project.output_dir")
    return {
        "root": ensure_dir(output_dir),
        "manifests": ensure_dir(os.path.join(output_dir, "manifests")),
        "geometry": ensure_dir(os.path.join(output_dir, "geometry")),
        "gsrecon": ensure_dir(os.path.join(output_dir, "gsrecon")),
        "masks": ensure_dir(os.path.join(output_dir, "masks")),
        "void": ensure_dir(os.path.join(output_dir, "void")),
        "inpaint": ensure_dir(os.path.join(output_dir, "inpaint")),
        "renders": ensure_dir(os.path.join(output_dir, "renders")),
    }


def dataset_from_config(config):
    return Inpaint360GSDataset(config)


def validate(config, strict_paths=False):
    validate_config(config, strict_paths=strict_paths)
    dirs = run_dirs(config)
    write_json(os.path.join(dirs["root"], "resolved_config.json"), config)
    return {"ok": True, "output_dir": dirs["root"]}


def discover_dataset(config):
    dataset = dataset_from_config(config)
    max_views = get_nested(config, "pipeline.max_views")
    summary = dataset.summary(max_views=max_views)
    dirs = run_dirs(config)
    write_json(os.path.join(dirs["manifests"], "dataset_summary.json"), summary)
    return summary


def prepare_geometry(config, dry_run=False):
    dirs = run_dirs(config)
    command = get_nested(config, "external.geometry_command", "")
    if not command:
        return {"skipped": True, "reason": "external.geometry_command is empty"}
    values = {
        "config_path": config.get("_config_path", ""),
        "dataset_root": get_nested(config, "dataset.root"),
        "scene_id": get_nested(config, "dataset.scene"),
        "max_views": get_nested(config, "pipeline.max_views"),
        "geometry_dir": dirs["geometry"],
        "geometry_manifest": os.path.join(dirs["geometry"], "geometry_manifest.json"),
    }
    result = run_command(command, values, dry_run=dry_run)
    write_json(os.path.join(dirs["geometry"], "geometry_command.json"), result)
    return result


def run_gsrecon(config, dry_run=False):
    dirs = run_dirs(config)
    values = {
        "config_path": config.get("_config_path", ""),
        "diffsplat_root": get_nested(config, "checkpoints.diffsplat_root"),
        "gsrecon_weights": get_nested(config, "checkpoints.gsrecon_weights"),
        "gsvae_weights": get_nested(config, "checkpoints.gsvae_weights"),
        "dataset_root": get_nested(config, "dataset.root"),
        "scene_id": get_nested(config, "dataset.scene"),
        "geometry_dir": dirs["geometry"],
        "geometry_manifest": os.path.join(dirs["geometry"], "geometry_manifest.json"),
        "gsrecon_dir": dirs["gsrecon"],
    }
    result = run_command(get_nested(config, "external.gsrecon_command", ""), values, dry_run=dry_run)
    write_json(os.path.join(dirs["gsrecon"], "gsrecon_command.json"), result)
    return result


def run_segmentation(config, dry_run=False):
    from latent_void.masks import Sam3CommandAdapter

    dirs = run_dirs(config)
    dataset = dataset_from_config(config)
    views = dataset.views(max_views=get_nested(config, "pipeline.max_views"))
    prompt = get_nested(config, "prompts.object", "")
    adapter = Sam3CommandAdapter(config)
    return adapter.run(views, prompt=prompt, output_dir=dirs["masks"], dry_run=dry_run)


def _resolve_gaussian_npz(config):
    configured = get_nested(config, "pipeline.gaussian_npz", "")
    if configured:
        return configured
    return os.path.join(run_dirs(config)["gsrecon"], "gaussians.npz")


def _resolve_latent_npy(config):
    configured = get_nested(config, "pipeline.latent_npy", "")
    if configured:
        return configured
    return os.path.join(run_dirs(config)["gsrecon"], "latent.npy")


def fuse_void(config):
    import numpy as np

    from latent_void.gaussians import delete_gaussians, load_gaussian_npz, require_projection_arrays, save_gaussian_npz
    from latent_void.io import save_array
    from latent_void.latent import latent_mask_from_gaussian_mask
    from latent_void.masks import fuse_gaussian_masks, load_masks_from_dir

    dirs = run_dirs(config)
    gaussian_npz = _resolve_gaussian_npz(config)
    arrays = load_gaussian_npz(gaussian_npz)
    uvs, visibility = require_projection_arrays(arrays)
    mask_paths, masks = load_masks_from_dir(dirs["masks"])
    if not masks:
        raise RuntimeError("no SAM/provided masks found in %s" % dirs["masks"])
    if len(masks) > uvs.shape[0]:
        masks = masks[:uvs.shape[0]]
        mask_paths = mask_paths[:uvs.shape[0]]
    deletion, scores, visible_votes = fuse_gaussian_masks(
        uvs[:len(masks)],
        visibility[:len(masks)],
        masks,
        threshold=float(get_nested(config, "pipeline.mask_threshold")),
    )
    latent_shape = None
    latent_path = _resolve_latent_npy(config)
    if os.path.exists(latent_path):
        latent_shape = np.load(latent_path, mmap_mode="r").shape
    latent_mask = latent_mask_from_gaussian_mask(
        deletion,
        latent_shape=latent_shape,
        downsample=int(get_nested(config, "pipeline.latent_downsample")),
        gaussian_grid_shape=arrays.get("gaussian_grid_shape"),
    )
    save_array(os.path.join(dirs["void"], "gaussian_deletion_mask.npy"), deletion.astype(np.uint8))
    save_array(os.path.join(dirs["void"], "gaussian_mask_scores.npy"), scores)
    save_array(os.path.join(dirs["void"], "gaussian_visible_votes.npy"), visible_votes)
    save_array(os.path.join(dirs["void"], "latent_void_mask.npy"), latent_mask.astype(np.uint8))
    deleted_arrays = delete_gaussians(arrays, deletion)
    save_gaussian_npz(os.path.join(dirs["void"], "gaussians_deleted.npz"), deleted_arrays)
    manifest = {
        "gaussian_npz": gaussian_npz,
        "mask_paths": mask_paths,
        "num_deleted_gaussians": int(deletion.sum()),
        "num_gaussians": int(deletion.shape[0]),
        "latent_mask_shape": list(latent_mask.shape),
    }
    write_json(os.path.join(dirs["void"], "void_manifest.json"), manifest)
    return manifest


def run_latent_inpaint(config, dry_run=False):
    from latent_void.io import load_array, save_array
    from latent_void.latent import fallback_inpaint_latent

    dirs = run_dirs(config)
    command = get_nested(config, "external.latent_inpaint_command", "")
    latent_path = _resolve_latent_npy(config)
    mask_path = os.path.join(dirs["void"], "latent_void_mask.npy")
    output_path = os.path.join(dirs["inpaint"], "latent_inpainted.npy")
    if command:
        values = {
            "config_path": config.get("_config_path", ""),
            "latent_path": latent_path,
            "mask_path": mask_path,
            "output_path": output_path,
            "inpaint_dir": dirs["inpaint"],
            "latent_inpaint_weights": get_nested(config, "checkpoints.latent_inpaint_weights", ""),
        }
        result = run_command(command, values, dry_run=dry_run)
        write_json(os.path.join(dirs["inpaint"], "latent_inpaint_command.json"), result)
        return result

    allow_fallback = bool(get_nested(config, "pipeline.allow_fallback_inpaint", False))
    if not allow_fallback:
        raise RuntimeError("external.latent_inpaint_command is empty and fallback is disabled")
    if dry_run:
        result = {"dry_run": True, "fallback": True, "output_path": output_path}
        write_json(os.path.join(dirs["inpaint"], "latent_inpaint_command.json"), result)
        return result
    latent = load_array(latent_path)
    mask = load_array(mask_path).astype(bool)
    inpainted = fallback_inpaint_latent(latent, mask)
    save_array(output_path, inpainted)
    result = {"dry_run": False, "fallback": True, "output_path": output_path}
    write_json(os.path.join(dirs["inpaint"], "latent_inpaint_command.json"), result)
    return result


def run_render(config, dry_run=False):
    dirs = run_dirs(config)
    command = get_nested(config, "external.render_command", "")
    if not command:
        return {"skipped": True, "reason": "external.render_command is empty"}
    values = {
        "config_path": config.get("_config_path", ""),
        "diffsplat_root": get_nested(config, "checkpoints.diffsplat_root"),
        "gsvae_weights": get_nested(config, "checkpoints.gsvae_weights"),
        "gaussian_npz": _resolve_gaussian_npz(config),
        "latent_path": _resolve_latent_npy(config),
        "inpainted_latent_path": os.path.join(dirs["inpaint"], "latent_inpainted.npy"),
        "render_dir": dirs["renders"],
    }
    result = run_command(command, values, dry_run=dry_run)
    write_json(os.path.join(dirs["renders"], "render_command.json"), result)
    return result


def run_pipeline(config, dry_run=False, skip_geometry=False, skip_reconstruct=False, skip_segment=False):
    validate(config)
    results = {"dataset": discover_dataset(config)}
    if not skip_geometry:
        geometry_result = prepare_geometry(config, dry_run=dry_run)
        if not geometry_result.get("skipped"):
            results["geometry"] = geometry_result
    if not skip_reconstruct:
        results["reconstruct"] = run_gsrecon(config, dry_run=dry_run)
    if not skip_segment:
        results["segment"] = run_segmentation(config, dry_run=dry_run)
    if dry_run:
        results["inpaint"] = run_latent_inpaint(config, dry_run=True)
        render_result = run_render(config, dry_run=True)
        if not render_result.get("skipped"):
            results["render"] = render_result
    else:
        results["fuse"] = fuse_void(config)
        results["inpaint"] = run_latent_inpaint(config, dry_run=False)
        render_result = run_render(config, dry_run=False)
        if not render_result.get("skipped"):
            results["render"] = render_result
    write_json(os.path.join(run_dirs(config)["root"], "run_manifest.json"), results)
    return results
