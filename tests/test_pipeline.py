import os
import shutil
import tempfile
import unittest

import numpy as np

from latent_void.pipeline import fuse_void, run_latent_inpaint


class PipelineTests(unittest.TestCase):
    def test_fuse_and_fallback_inpaint(self):
        tmp = tempfile.mkdtemp()
        try:
            output_dir = os.path.join(tmp, "run")
            gs_dir = os.path.join(output_dir, "gsrecon")
            mask_dir = os.path.join(output_dir, "masks")
            os.makedirs(gs_dir)
            os.makedirs(mask_dir)

            uvs = np.array([
                [[0, 0], [1, 1], [0, 1]],
                [[0, 0], [1, 1], [0, 1]],
            ], dtype=np.float32)
            visibility = np.ones((2, 3), dtype=np.uint8)
            opacity = np.ones(3, dtype=np.float32)
            features = np.ones((3, 4), dtype=np.float32)
            np.savez_compressed(
                os.path.join(gs_dir, "gaussians.npz"),
                uvs=uvs,
                visibility=visibility,
                opacity=opacity,
                features=features,
            )
            np.save(os.path.join(gs_dir, "latent.npy"), np.arange(16, dtype=np.float32).reshape(1, 4, 4))
            np.save(os.path.join(mask_dir, "000.npy"), np.array([[1, 0], [0, 0]], dtype=np.uint8))
            np.save(os.path.join(mask_dir, "001.npy"), np.array([[1, 0], [0, 0]], dtype=np.uint8))

            config = {
                "project": {"output_dir": output_dir},
                "dataset": {"type": "inpaint360gs", "root": tmp, "scene": "scene"},
                "checkpoints": {},
                "pipeline": {
                    "mask_threshold": 0.75,
                    "latent_downsample": 1,
                    "allow_fallback_inpaint": True,
                },
                "external": {"latent_inpaint_command": ""},
            }

            manifest = fuse_void(config)
            self.assertEqual(manifest["num_deleted_gaussians"], 1)
            result = run_latent_inpaint(config)
            self.assertTrue(result["fallback"])
            self.assertTrue(os.path.exists(result["output_path"]))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
