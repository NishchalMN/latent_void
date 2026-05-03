import os
import re
from copy import deepcopy

import yaml


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


class ConfigError(Exception):
    """Raised when a run config is missing required fields."""


def _expand_env_value(value):
    if isinstance(value, str):
        def repl(match):
            name = match.group(1)
            default = match.group(3)
            return os.environ.get(name, default if default is not None else "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_value(item) for key, item in value.items()}
    return value


def load_config(path):
    with open(path, "r") as handle:
        data = yaml.safe_load(handle) or {}
    data = _expand_env_value(data)
    data["_config_path"] = os.path.abspath(path)
    return data


def get_nested(config, dotted_key, default=None):
    current = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def set_nested(config, dotted_key, value):
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def validate_config(config, strict_paths=False):
    required = [
        "project.output_dir",
        "dataset.type",
        "dataset.root",
        "dataset.scene",
        "checkpoints.diffsplat_root",
        "checkpoints.gsvae_weights",
        "checkpoints.sam3_root",
        "pipeline.mask_threshold",
        "pipeline.latent_downsample",
        "hpc.account",
        "hpc.partition",
    ]
    missing = [key for key in required if get_nested(config, key) in (None, "")]
    if missing:
        raise ConfigError("missing required config keys: " + ", ".join(missing))

    if get_nested(config, "dataset.type") not in ("inpaint360gs", "dl3dv"):
        raise ConfigError("dataset.type must be inpaint360gs or dl3dv")

    threshold = float(get_nested(config, "pipeline.mask_threshold"))
    if threshold <= 0.0 or threshold > 1.0:
        raise ConfigError("pipeline.mask_threshold must be in (0, 1]")

    downsample = int(get_nested(config, "pipeline.latent_downsample"))
    if downsample < 1:
        raise ConfigError("pipeline.latent_downsample must be >= 1")

    if strict_paths:
        for dotted_key in [
            "dataset.root",
            "checkpoints.diffsplat_root",
            "checkpoints.gsrecon_weights",
            "checkpoints.gsvae_weights",
            "checkpoints.sam3_root",
            "checkpoints.sdxl_vae_path",
            "checkpoints.tiny_vae_path",
        ]:
            path = get_nested(config, dotted_key)
            if path and not os.path.exists(path):
                raise ConfigError("%s does not exist: %s" % (dotted_key, path))
        for dotted_key in ["checkpoints.sdxl_vae_path", "checkpoints.tiny_vae_path"]:
            path = get_nested(config, dotted_key)
            if path and os.path.isdir(path) and not os.path.exists(os.path.join(path, "config.json")):
                raise ConfigError("%s missing config.json: %s" % (dotted_key, path))

    return True


def with_overrides(config, overrides):
    merged = deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError("override must be KEY=VALUE: %s" % item)
        key, value = item.split("=", 1)
        set_nested(merged, key, value)
    return merged
