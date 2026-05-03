import os
import shutil
import tempfile
import unittest

import numpy as np

from latent_void.gaussians import load_gaussian_npz
from tools.merge_local_inpaint import main as merge_main


class MergeLocalInpaintTests(unittest.TestCase):
    def test_merge_removes_deleted_and_appends_visible_local_gaussians(self):
        tmp = tempfile.mkdtemp()
        old_argv = __import__("sys").argv
        try:
            full_path = os.path.join(tmp, "full.npz")
            local_path = os.path.join(tmp, "local.npz")
            out_path = os.path.join(tmp, "merged.npz")
            mask_path = os.path.join(tmp, "delete.npy")
            np.savez_compressed(
                full_path,
                opacity=np.array([[1.0], [0.8], [0.5]], dtype=np.float32),
                features=np.ones((3, 2), dtype=np.float32),
            )
            np.savez_compressed(
                local_path,
                opacity=np.array([[0.9], [0.0]], dtype=np.float32),
                features=np.ones((2, 2), dtype=np.float32) * 2.0,
                visibility=np.array([[1, 0], [1, 0]], dtype=np.uint8),
            )
            np.save(mask_path, np.array([False, True, False], dtype=np.uint8))
            __import__("sys").argv = [
                "merge_local_inpaint.py",
                "--full-gaussian-npz", full_path,
                "--local-gaussian-npz", local_path,
                "--output-npz", out_path,
                "--deletion-mask", mask_path,
                "--remove-deleted-full",
                "--min-local-opacity", "0.1",
                "--min-visible-views", "1",
            ]
            self.assertEqual(merge_main(), 0)
            merged = load_gaussian_npz(out_path)
            self.assertEqual(merged["opacity"].shape[0], 3)
            self.assertTrue(np.allclose(merged["features"][-1], np.array([2.0, 2.0], dtype=np.float32)))
        finally:
            __import__("sys").argv = old_argv
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
