import unittest

import numpy as np

from latent_void.masks import fuse_gaussian_masks, shadow_offset


class MaskTests(unittest.TestCase):
    def test_fuse_gaussian_masks_by_visible_votes(self):
        masks = [
            np.array([[1, 0], [0, 0]], dtype=bool),
            np.array([[1, 0], [0, 0]], dtype=bool),
        ]
        uvs = np.array([
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1]],
        ], dtype=np.float32)
        visibility = np.array([
            [1, 1],
            [1, 1],
        ], dtype=bool)
        deletion, scores, votes = fuse_gaussian_masks(uvs, visibility, masks, threshold=0.75)
        self.assertTrue(deletion[0])
        self.assertFalse(deletion[1])
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(votes[0], 2.0)

    def test_shadow_offset(self):
        obj = np.zeros((4, 4), dtype=bool)
        shadow = np.zeros((4, 4), dtype=bool)
        obj[1, 1] = True
        shadow[2, 3] = True
        offset = shadow_offset(obj, shadow)
        self.assertTrue(np.allclose(offset, np.array([2.0, 1.0], dtype=np.float32)))

    def test_resize_mask_uses_nearest_binary_values(self):
        try:
            from tools.run_sam3_multiview import _resize_mask
        except ModuleNotFoundError as exc:
            if exc.name == "PIL":
                self.skipTest("Pillow is not installed in the local smoke environment")
            raise
        mask = np.zeros((4, 4), dtype=bool)
        mask[:2, :2] = True
        resized = _resize_mask(mask, 2)
        self.assertEqual(resized.shape, (2, 2))
        self.assertTrue(resized[0, 0])
        self.assertFalse(resized[1, 1])


if __name__ == "__main__":
    unittest.main()
