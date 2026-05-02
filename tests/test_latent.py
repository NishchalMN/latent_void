import unittest

import numpy as np

from latent_void.latent import fallback_inpaint_latent, latent_mask_from_gaussian_mask


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


if __name__ == "__main__":
    unittest.main()
