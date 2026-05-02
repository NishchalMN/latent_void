import numpy as np


GAUSSIAN_CHANNELS = [
    "rgb_r",
    "rgb_g",
    "rgb_b",
    "depth",
    "scale_x",
    "scale_y",
    "scale_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "opacity",
]


def validate_gaussian_grid(grid):
    array = np.asarray(grid)
    if array.ndim < 3:
        raise ValueError("gaussian grid must have at least [channels, height, width]")
    if array.shape[-3] != 12:
        raise ValueError("expected 12 Gaussian channels, got %s" % (array.shape[-3],))
    return array


def load_gaussian_npz(path):
    data = np.load(path)
    return {key: data[key] for key in data.files}


def save_gaussian_npz(path, arrays):
    np.savez_compressed(path, **arrays)


def require_projection_arrays(arrays):
    missing = [key for key in ["uvs", "visibility"] if key not in arrays]
    if missing:
        raise ValueError(
            "gaussian npz is missing %s. Export projected uvs/visibility from "
            "GSRecon/rendering or run projection upstream." % ", ".join(missing)
        )
    return arrays["uvs"], arrays["visibility"]


def delete_gaussians(arrays, deletion_mask):
    result = dict(arrays)
    mask = np.asarray(deletion_mask).astype(bool)
    if "opacity" in result and result["opacity"].shape[0] == mask.shape[0]:
        opacity = np.array(result["opacity"], copy=True)
        opacity[mask] = 0.0
        result["opacity"] = opacity
    if "features" in result and result["features"].shape[0] == mask.shape[0]:
        features = np.array(result["features"], copy=True)
        features[mask] = 0.0
        result["features"] = features
    result["deletion_mask"] = mask.astype(np.uint8)
    return result
