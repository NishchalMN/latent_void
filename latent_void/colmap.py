import os
import struct


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

CAMERA_MODEL_IDS = {name: (model_id, count) for model_id, (name, count) in CAMERA_MODELS.items()}


def _qvec_to_rotmat(qvec):
    qw, qx, qy, qz = qvec
    return [
        [
            1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
            2.0 * qx * qy - 2.0 * qz * qw,
            2.0 * qx * qz + 2.0 * qy * qw,
        ],
        [
            2.0 * qx * qy + 2.0 * qz * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
            2.0 * qy * qz - 2.0 * qx * qw,
        ],
        [
            2.0 * qx * qz - 2.0 * qy * qw,
            2.0 * qy * qz + 2.0 * qx * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
        ],
    ]


def _transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def _matvec(matrix, vector):
    return [sum(row[i] * vector[i] for i in range(len(vector))) for row in matrix]


def _camera_to_world(qvec, tvec):
    world_to_camera = _qvec_to_rotmat(qvec)
    rotation = _transpose(world_to_camera)
    translation = [-value for value in _matvec(rotation, tvec)]
    return [
        rotation[0] + [translation[0]],
        rotation[1] + [translation[1]],
        rotation[2] + [translation[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _intrinsics(model, width, height, params):
    if model == "SIMPLE_PINHOLE":
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    elif model in ("SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx, fy, cx, cy = params[:4]
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "fxfycxcy": [fx, fy, cx, cy],
        "fxfycxcy_normalized": [fx / width, fy / height, cx / width, cy / height],
    }


def _camera_payload(camera):
    return {
        "camera_id": camera["camera_id"],
        "model": camera["model"],
        "width": camera["width"],
        "height": camera["height"],
        "params": camera["params"],
        "intrinsics": _intrinsics(camera["model"], camera["width"], camera["height"], camera["params"]),
    }


def read_colmap_cameras_text(path):
    cameras = {}
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": int(parts[2]),
                "height": int(parts[3]),
                "params": [float(value) for value in parts[4:]],
            }
    return cameras


def read_colmap_images_text(path, cameras):
    views = {}
    with open(path, "r") as handle:
        lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    idx = 0
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) < 10:
            idx += 1
            continue
        image_id = int(parts[0])
        qvec = [float(value) for value in parts[1:5]]
        tvec = [float(value) for value in parts[5:8]]
        camera_id = int(parts[8])
        image_name = " ".join(parts[9:])
        camera = cameras.get(camera_id, {})
        payload = {
            "source": "colmap_text",
            "image_id": image_id,
            "image_name": image_name,
            "qvec": qvec,
            "tvec": tvec,
            "c2w": _camera_to_world(qvec, tvec),
        }
        payload.update(_camera_payload(camera) if camera else {"camera_id": camera_id})
        views[os.path.basename(image_name)] = payload
        idx += 2
    return views


def _read_bytes(handle, num_bytes, fmt):
    data = handle.read(num_bytes)
    if len(data) != num_bytes:
        raise IOError("unexpected end of COLMAP binary file")
    return struct.unpack("<" + fmt, data)


def read_colmap_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as handle:
        (num_cameras,) = _read_bytes(handle, 8, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_bytes(handle, 24, "iiQQ")
            model, num_params = CAMERA_MODELS[model_id]
            params = list(_read_bytes(handle, 8 * num_params, "d" * num_params))
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model": model,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def _read_c_string(handle):
    chars = []
    while True:
        char = handle.read(1)
        if char == b"":
            raise IOError("unexpected end of COLMAP binary image name")
        if char == b"\x00":
            return b"".join(chars).decode("utf-8")
        chars.append(char)


def read_colmap_images_binary(path, cameras):
    views = {}
    with open(path, "rb") as handle:
        (num_images,) = _read_bytes(handle, 8, "Q")
        for _ in range(num_images):
            values = _read_bytes(handle, 64, "idddddddi")
            image_id = int(values[0])
            qvec = [float(value) for value in values[1:5]]
            tvec = [float(value) for value in values[5:8]]
            camera_id = int(values[8])
            image_name = _read_c_string(handle)
            (num_points2d,) = _read_bytes(handle, 8, "Q")
            handle.seek(num_points2d * 24, os.SEEK_CUR)
            camera = cameras.get(camera_id, {})
            payload = {
                "source": "colmap_binary",
                "image_id": image_id,
                "image_name": image_name,
                "qvec": qvec,
                "tvec": tvec,
                "c2w": _camera_to_world(qvec, tvec),
            }
            payload.update(_camera_payload(camera) if camera else {"camera_id": camera_id})
            views[os.path.basename(image_name)] = payload
    return views


def load_colmap_scene(sparse_dir):
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")
    images_txt = os.path.join(sparse_dir, "images.txt")
    cameras_bin = os.path.join(sparse_dir, "cameras.bin")
    images_bin = os.path.join(sparse_dir, "images.bin")

    if os.path.exists(cameras_txt) and os.path.exists(images_txt):
        cameras = read_colmap_cameras_text(cameras_txt)
        return read_colmap_images_text(images_txt, cameras)
    if os.path.exists(cameras_bin) and os.path.exists(images_bin):
        cameras = read_colmap_cameras_binary(cameras_bin)
        return read_colmap_images_binary(images_bin, cameras)
    return {}
