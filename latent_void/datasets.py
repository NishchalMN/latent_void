import os

from latent_void.colmap import load_colmap_scene
from latent_void.config import get_nested
from latent_void.io import expand_brace_glob, read_json


class SceneView(object):
    def __init__(self, image_path, mask_path=None, camera=None, view_id=None):
        self.image_path = image_path
        self.mask_path = mask_path
        self.camera = camera or {}
        self.view_id = view_id

    def to_manifest(self):
        return {
            "view_id": self.view_id,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "camera": self.camera,
        }


class Inpaint360GSDataset(object):
    def __init__(self, config):
        self.config = config
        self.root = get_nested(config, "dataset.root")
        self.scene = get_nested(config, "dataset.scene")
        self.scene_dir = os.path.join(self.root, self.scene)
        self.images_glob = get_nested(config, "dataset.images_glob", "images/*.{png,jpg,jpeg}")
        self.masks_glob = get_nested(config, "dataset.masks_glob", "masks/*.{npy,png,jpg,jpeg}")
        self.cameras_path = get_nested(config, "dataset.cameras_path", "sparse/0")

    def _scene_glob(self, pattern):
        return os.path.join(self.scene_dir, pattern)

    def image_paths(self):
        return expand_brace_glob(self._scene_glob(self.images_glob))

    def mask_paths(self):
        return expand_brace_glob(self._scene_glob(self.masks_glob))

    def camera_map(self):
        path = os.path.join(self.scene_dir, self.cameras_path)
        if os.path.isdir(path):
            return load_colmap_scene(path)
        fallback_colmap = os.path.join(self.scene_dir, "sparse", "0")
        if not os.path.exists(path) and os.path.isdir(fallback_colmap):
            return load_colmap_scene(fallback_colmap)
        if not os.path.exists(path):
            return {}
        data = read_json(path)
        if isinstance(data, dict) and "views" in data:
            data = data["views"]
        if isinstance(data, list):
            result = {}
            for item in data:
                key = item.get("image") or item.get("image_path") or item.get("file") or item.get("view_id")
                if key is not None:
                    result[os.path.basename(str(key))] = item
            return result
        return data if isinstance(data, dict) else {}

    def views(self, max_views=None):
        images = self.image_paths()
        masks = self.mask_paths()
        cameras = self.camera_map()
        mask_by_stem = {}
        for path in masks:
            stem = os.path.splitext(os.path.basename(path))[0]
            mask_by_stem[stem] = path
        views = []
        for idx, image_path in enumerate(images):
            if max_views is not None and idx >= int(max_views):
                break
            basename = os.path.basename(image_path)
            stem = os.path.splitext(basename)[0]
            camera = cameras.get(basename) or cameras.get(stem) or {}
            views.append(SceneView(
                image_path=image_path,
                mask_path=mask_by_stem.get(stem),
                camera=camera,
                view_id=stem,
            ))
        return views

    def summary(self, max_views=None):
        views = self.views(max_views=max_views)
        return {
            "type": "inpaint360gs",
            "root": self.root,
            "scene": self.scene,
            "scene_dir": self.scene_dir,
            "num_images": len(self.image_paths()),
            "num_masks": len(self.mask_paths()),
            "num_views_selected": len(views),
            "has_cameras": bool(self.camera_map()),
            "views": [view.to_manifest() for view in views],
        }
