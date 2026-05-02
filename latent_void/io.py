import glob
import json
import os


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def write_json(path, data):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def expand_brace_glob(pattern):
    if "{" not in pattern or "}" not in pattern:
        return sorted(glob.glob(pattern))
    prefix = pattern[:pattern.index("{")]
    suffix = pattern[pattern.index("}") + 1:]
    inner = pattern[pattern.index("{") + 1:pattern.index("}")]
    paths = []
    for option in inner.split(","):
        paths.extend(glob.glob(prefix + option + suffix))
    return sorted(set(paths))


def load_mask(path):
    import numpy as np

    lower = path.lower()
    if lower.endswith(".npy"):
        mask = np.load(path)
    else:
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required to read image masks: %s" % path)
        mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype(np.float32) > 0.5


def save_mask(path, mask):
    import numpy as np

    ensure_dir(os.path.dirname(path))
    if path.lower().endswith(".npy"):
        np.save(path, mask.astype(np.uint8))
    else:
        try:
            from PIL import Image
        except ImportError:
            raise RuntimeError("Pillow is required to write image masks: %s" % path)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_array(path, array):
    import numpy as np

    ensure_dir(os.path.dirname(path))
    np.save(path, array)


def load_array(path):
    import numpy as np

    return np.load(path)
