import json
import os


def scaled_intrinsics(camera, width, height):
    intrinsics = camera.get("intrinsics", {})
    source_width = float(camera.get("width", width))
    source_height = float(camera.get("height", height))
    sx = float(width) / source_width if source_width else 1.0
    sy = float(height) / source_height if source_height else 1.0
    fx = float(intrinsics.get("fx", intrinsics.get("fxfycxcy", [1.0, 1.0, 0.5, 0.5])[0])) * sx
    fy = float(intrinsics.get("fy", intrinsics.get("fxfycxcy", [1.0, 1.0, 0.5, 0.5])[1])) * sy
    cx = float(intrinsics.get("cx", intrinsics.get("fxfycxcy", [1.0, 1.0, 0.5, 0.5])[2])) * sx
    cy = float(intrinsics.get("cy", intrinsics.get("fxfycxcy", [1.0, 1.0, 0.5, 0.5])[3])) * sy
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fxfycxcy": [fx, fy, cx, cy],
        "fxfycxcy_normalized": [fx / float(width), fy / float(height), cx / float(width), cy / float(height)],
    }


def unproject_depth_to_world(depth, camera, depth_scale=5.0):
    import numpy as np

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    height, width = depth.shape
    intrinsics = scaled_intrinsics(camera, width, height)
    fx = max(float(intrinsics["fx"]), 1e-6)
    fy = max(float(intrinsics["fy"]), 1e-6)
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    z = depth.astype(np.float32) * float(depth_scale)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points_camera = np.stack([x, y, z], axis=-1)
    c2w = np.asarray(camera.get("c2w"), dtype=np.float32)
    if c2w.shape != (4, 4):
        raise ValueError("camera must include c2w with shape [4, 4]")
    rotation = c2w[:3, :3]
    translation = c2w[:3, 3]
    points_world = points_camera.reshape(-1, 3) @ rotation.T + translation
    return points_world.reshape(height, width, 3).astype(np.float32), intrinsics


def normalize_coordinate_maps(coord_maps):
    import numpy as np

    if not coord_maps:
        return [], {"min": [], "max": []}
    stacked = np.stack([np.asarray(item, dtype=np.float32) for item in coord_maps], axis=0)
    valid = np.isfinite(stacked).all(axis=-1)
    if not valid.any():
        return [np.zeros_like(item, dtype=np.float32) for item in coord_maps], {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
    values = stacked[valid]
    min_value = values.min(axis=0)
    max_value = values.max(axis=0)
    denom = max_value - min_value
    denom[denom < 1e-6] = 1.0
    normalized = [(np.asarray(item, dtype=np.float32) - min_value) / denom for item in coord_maps]
    normalized = [np.clip(item, 0.0, 1.0).astype(np.float32) for item in normalized]
    return normalized, {"min": min_value.tolist(), "max": max_value.tolist()}


def load_geometry_manifest(path):
    with open(path, "r") as handle:
        return json.load(handle)


def geometry_manifest_path(geometry_dir):
    return os.path.join(geometry_dir, "geometry_manifest.json")
