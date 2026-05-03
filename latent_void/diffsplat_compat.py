"""Runtime compatibility helpers for the external DiffSplat checkout."""

import importlib.machinery
import os
import sys
import types


SDXL_VAE_REPO_ID = "madebyollin/sdxl-vae-fp16-fix"
TAESDXL_REPO_ID = "madebyollin/taesdxl"


def resolve_aux_model_paths(sdxl_vae_path="", tiny_vae_path=""):
    return (
        sdxl_vae_path or os.environ.get("LATENT_VOID_SDXL_VAE_PATH", ""),
        tiny_vae_path or os.environ.get("LATENT_VOID_TINY_VAE_PATH", ""),
    )


def validate_aux_model_paths(sdxl_vae_path="", tiny_vae_path=""):
    missing = []
    for label, path in [
        ("sdxl_vae_path", sdxl_vae_path),
        ("tiny_vae_path", tiny_vae_path),
    ]:
        if not path:
            missing.append("%s is empty" % label)
        elif not os.path.isdir(path):
            missing.append("%s does not exist: %s" % (label, path))
        elif not os.path.exists(os.path.join(path, "config.json")):
            missing.append("%s missing config.json: %s" % (label, path))
    if missing:
        raise RuntimeError("missing DiffSplat auxiliary VAE snapshots: " + "; ".join(missing))


def patch_transformers_compat():
    try:
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except Exception:
        return False
    patched = False
    for name in ("apply_chunking_to_forward", "prune_linear_layer", "Conv1D"):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))
            patched = True
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        import torch

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            heads = set(heads) - set(already_pruned_heads)
            mask = torch.ones(n_heads, head_size)
            for head in heads:
                head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        modeling_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
        patched = True
    return patched


def patch_optional_imports():
    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")
        wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)

        def noop(*args, **kwargs):
            return None

        wandb.init = noop
        wandb.log = noop
        wandb.finish = noop
        wandb.define_metric = noop
        wandb.Image = lambda *args, **kwargs: args[0] if args else None
        wandb.run = None
        sys.modules["wandb"] = wandb


def patch_diffusers_model_paths(sdxl_vae_path="", tiny_vae_path=""):
    sdxl_vae_path, tiny_vae_path = resolve_aux_model_paths(sdxl_vae_path, tiny_vae_path)
    mapping = {
        SDXL_VAE_REPO_ID: sdxl_vae_path,
        TAESDXL_REPO_ID: tiny_vae_path,
    }
    mapping = {repo_id: path for repo_id, path in mapping.items() if path}
    if not mapping:
        return False

    from diffusers import AutoencoderKL, AutoencoderTiny

    original_kl = AutoencoderKL.from_pretrained
    original_tiny = AutoencoderTiny.from_pretrained

    def mapped_path(model_id):
        return mapping.get(str(model_id), model_id)

    def kl_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return original_kl(mapped_path(pretrained_model_name_or_path), *args, **kwargs)

    def tiny_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return original_tiny(mapped_path(pretrained_model_name_or_path), *args, **kwargs)

    AutoencoderKL.from_pretrained = classmethod(kl_from_pretrained)
    AutoencoderTiny.from_pretrained = classmethod(tiny_from_pretrained)
    return True


def patch_gaussian_rasterizer_compat():
    """Adapt older diff-gaussian-rasterization builds to DiffSplat's render API."""
    try:
        import torch
        import diff_gaussian_rasterization as dgr
    except Exception:
        return False

    patched = False
    settings = getattr(dgr, "GaussianRasterizationSettings", None)
    fields = getattr(settings, "_fields", ()) if settings is not None else ()
    if settings is not None and "require_coord" not in fields:
        original_settings = settings

        def compat_settings(*args, **kwargs):
            kwargs.pop("require_coord", None)
            return original_settings(*args, **kwargs)

        compat_settings._latent_void_original = original_settings
        compat_settings._fields = fields
        dgr.GaussianRasterizationSettings = compat_settings
        patched = True

    rasterizer = getattr(dgr, "GaussianRasterizer", None)
    forward = getattr(rasterizer, "forward", None) if rasterizer is not None else None
    if forward is not None and not getattr(forward, "_latent_void_compat", False):

        def compat_forward(self, *args, **kwargs):
            outputs = forward(self, *args, **kwargs)
            if isinstance(outputs, tuple) and len(outputs) == 6:
                image, radii, depth, mdepth, alpha, normal = outputs
                empty = torch.empty(0, dtype=image.dtype, device=image.device)
                return image, radii, empty, empty, depth, mdepth, alpha, normal
            return outputs

        compat_forward._latent_void_compat = True
        rasterizer.forward = compat_forward
        patched = True

    return patched
