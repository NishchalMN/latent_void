import unittest

import numpy as np

from latent_void.gaussians import GAUSSIAN_CHANNELS
from latent_void.latent import context_inpaint_latent, expand_mask_to_latent, fallback_inpaint_latent, latent_mask_from_gaussian_mask


class LatentTests(unittest.TestCase):
    def test_latent_mask_downsamples_any_deleted_cell(self):
        mask = np.zeros((4, 4), dtype=bool)
        mask[1, 2] = True
        latent_mask = latent_mask_from_gaussian_mask(mask, downsample=2)
        self.assertEqual(latent_mask.shape, (2, 2))
        self.assertTrue(latent_mask[0, 1])

    def test_fallback_inpaint_preserves_unmasked_cells(self):
        latent = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        mask = np.zeros((4, 4), dtype=bool)
        mask[0, 0] = True
        output = fallback_inpaint_latent(latent, mask)
        self.assertEqual(output[0, 1, 1], latent[0, 1, 1])
        self.assertNotEqual(output[0, 0, 0], latent[0, 0, 0])

    def test_context_inpaint_preserves_unmasked_cells(self):
        latent = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        latent[0, 1:3, 1:3] = 100.0
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        output = context_inpaint_latent(latent, mask, iterations=4)
        self.assertTrue(np.allclose(output[:, ~mask], latent[:, ~mask]))
        self.assertFalse(np.allclose(output[:, mask], latent[:, mask]))

    def test_gaussian_grid_mask_maps_to_per_view_latent_shape(self):
        gaussian_mask = np.zeros(2 * 4 * 4, dtype=bool)
        gaussian_mask[0] = True
        gaussian_mask[16 + 15] = True
        latent_mask = latent_mask_from_gaussian_mask(
            gaussian_mask,
            latent_shape=(2, 4, 2, 2),
            gaussian_grid_shape=(1, 2, 4, 4),
        )
        self.assertEqual(latent_mask.shape, (2, 2, 2))
        self.assertTrue(latent_mask[0, 0, 0])
        self.assertTrue(latent_mask[1, 1, 1])

    def test_expand_2d_mask_broadcasts_over_latent_views(self):
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        latent = np.zeros((3, 4, 2, 2), dtype=np.float32)
        expanded = expand_mask_to_latent(mask, latent)
        self.assertEqual(expanded.shape, (3, 2, 2))
        self.assertTrue(expanded[:, 0, 0].all())

    def test_diffsplat_gaussian_channel_order_has_depth_last(self):
        self.assertEqual(GAUSSIAN_CHANNELS[:3], ["rgb_r", "rgb_g", "rgb_b"])
        self.assertEqual(GAUSSIAN_CHANNELS[3:6], ["scale_x", "scale_y", "scale_z"])
        self.assertEqual(GAUSSIAN_CHANNELS[-1], "depth")


if __name__ == "__main__":
    unittest.main()
