import unittest

import numpy as np

from latent_void.geometry import normalize_coordinate_maps, scaled_intrinsics, unproject_depth_to_world


class GeometryTests(unittest.TestCase):
    def test_unproject_depth_identity_camera(self):
        camera = {
            "width": 2,
            "height": 2,
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
            "c2w": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
        coords, intrinsics = unproject_depth_to_world(np.ones((2, 2), dtype=np.float32), camera, depth_scale=1.0)
        self.assertEqual(intrinsics["fxfycxcy_normalized"], [0.5, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(coords[1, 1], [1.0, 1.0, 1.0])

    def test_coordinate_normalization(self):
        first = np.array([[[0.0, 2.0, 4.0]]], dtype=np.float32)
        second = np.array([[[2.0, 4.0, 8.0]]], dtype=np.float32)
        normalized, stats = normalize_coordinate_maps([first, second])
        np.testing.assert_allclose(normalized[0][0, 0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(normalized[1][0, 0], [1.0, 1.0, 1.0])
        self.assertEqual(stats["min"], [0.0, 2.0, 4.0])

    def test_scaled_intrinsics(self):
        camera = {
            "width": 400,
            "height": 300,
            "intrinsics": {"fx": 200.0, "fy": 150.0, "cx": 100.0, "cy": 75.0},
        }
        self.assertEqual(scaled_intrinsics(camera, 200, 150)["fxfycxcy"], [100.0, 75.0, 50.0, 37.5])


if __name__ == "__main__":
    unittest.main()
