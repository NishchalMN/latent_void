import os

from latent_void.config import get_nested
from latent_void.external import run_command, write_views_manifest
from latent_void.io import ensure_dir, expand_brace_glob, load_mask, write_json


class ManualMaskProvider(object):
    def __init__(self, mask_paths):
        self.mask_paths = list(mask_paths)

    def load(self):
        return [load_mask(path) for path in self.mask_paths]


class Sam3CommandAdapter(object):
    def __init__(self, config):
        self.config = config

    def run(self, views, prompt, output_dir, dry_run=False):
        ensure_dir(output_dir)
        manifest_path = os.path.join(output_dir, "sam3_views_manifest.json")
        write_views_manifest(manifest_path, views, extra={"prompt": prompt})
        values = {
            "sam3_root": get_nested(self.config, "checkpoints.sam3_root"),
            "sam3_weights": get_nested(self.config, "checkpoints.sam3_weights"),
            "manifest_path": manifest_path,
            "prompt": prompt,
            "mask_dir": output_dir,
            "mask_resize": int(get_nested(self.config, "geometry.input_res", 0) or 0),
        }
        result = run_command(get_nested(self.config, "external.sam3_command", ""), values, dry_run=dry_run)
        result["manifest_path"] = manifest_path
        result["mask_dir"] = output_dir
        write_json(os.path.join(output_dir, "sam3_command.json"), result)
        return result


def mask_center(mask):
    import numpy as np

    coords = np.argwhere(mask.astype(bool))
    if coords.size == 0:
        return None
    yx = coords.mean(axis=0)
    return np.array([float(yx[1]), float(yx[0])], dtype=np.float32)


def shadow_offset(object_mask, shadow_mask):
    import numpy as np

    object_center = mask_center(object_mask)
    shadow_center = mask_center(shadow_mask)
    if object_center is None or shadow_center is None:
        return None
    return shadow_center - object_center


def load_masks_from_dir(mask_dir):
    paths = []
    for pattern in ["*.npy", "*.png", "*.jpg", "*.jpeg"]:
        paths.extend(expand_brace_glob(os.path.join(mask_dir, pattern)))
    paths = sorted(set(paths))
    return paths, [load_mask(path) for path in paths]


def _binary_morph(mask, radius, op):
    import numpy as np

    radius = int(radius)
    if radius <= 0:
        return np.asarray(mask).astype(bool)
    result = np.asarray(mask).astype(bool)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=(op == "erode"))
        neighborhoods = [
            padded[dy:dy + result.shape[0], dx:dx + result.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        stack = np.stack(neighborhoods, axis=0)
        result = stack.all(axis=0) if op == "erode" else stack.any(axis=0)
    return result


def _component_filter(mask, min_area=0, max_area_fraction=1.0):
    import numpy as np

    mask = np.asarray(mask).astype(bool)
    min_area = int(min_area or 0)
    max_area = int(float(max_area_fraction) * mask.size)
    if min_area <= 1 and max_area >= mask.size:
        return mask
    keep = np.zeros(mask.shape, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        component = []
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        area = len(component)
        if area >= min_area and area <= max_area:
            for y, x in component:
                keep[y, x] = True
    return keep


def clean_binary_mask(mask, min_area=0, max_area_fraction=1.0, erode_pixels=0, dilate_pixels=0):
    cleaned = _component_filter(mask, min_area=min_area, max_area_fraction=max_area_fraction)
    cleaned = _binary_morph(cleaned, erode_pixels, "erode")
    cleaned = _binary_morph(cleaned, dilate_pixels, "dilate")
    return cleaned


def sample_mask(mask, uv):
    h, w = mask.shape[:2]
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(mask[y, x])


def fuse_gaussian_masks(uvs, visibility, masks, threshold=0.55):
    import numpy as np

    """Fuse per-view 2D masks into a Gaussian deletion mask.

    Args:
        uvs: array [V, N, 2] with pixel coordinates for each view/Gaussian.
        visibility: array [V, N] bool visibility.
        masks: list of V binary masks.
        threshold: delete if masked visible vote ratio >= threshold.
    """
    uvs = np.asarray(uvs)
    visibility = np.asarray(visibility).astype(bool)
    if uvs.ndim != 3 or uvs.shape[2] != 2:
        raise ValueError("uvs must have shape [views, gaussians, 2]")
    if visibility.shape != uvs.shape[:2]:
        raise ValueError("visibility must have shape [views, gaussians]")
    if len(masks) != uvs.shape[0]:
        raise ValueError("number of masks must match uvs views")

    num_views, num_gaussians = visibility.shape
    masked_votes = np.zeros(num_gaussians, dtype=np.float32)
    visible_votes = np.zeros(num_gaussians, dtype=np.float32)
    for view_idx in range(num_views):
        mask = masks[view_idx]
        for gaussian_idx in np.where(visibility[view_idx])[0]:
            visible_votes[gaussian_idx] += 1.0
            if sample_mask(mask, uvs[view_idx, gaussian_idx]):
                masked_votes[gaussian_idx] += 1.0
    scores = np.zeros(num_gaussians, dtype=np.float32)
    valid = visible_votes > 0
    scores[valid] = masked_votes[valid] / visible_votes[valid]
    deletion = scores >= float(threshold)
    return deletion, scores, visible_votes
