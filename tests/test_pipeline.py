import os
import shutil
import tempfile
import unittest

import numpy as np

from latent_void.pipeline import fuse_void, prepare_geometry, run_latent_inpaint


class PipelineTests(unittest.TestCase):
    def test_prepare_geometry_formats_max_views_override(self):
        tmp = tempfile.mkdtemp()
        try:
            config = {
                "_config_path": "/tmp/config.yaml",
                "project": {"output_dir": tmp},
                "dataset": {"type": "inpaint360gs", "root": "/tmp/data", "scene": "scene"},
                "pipeline": {"max_views": 3},
                "external": {
                    "geometry_command": "python tools/preprocess_geometry.py --config {config_path} --output-dir {geometry_dir} --max-views {max_views}"
                },
            }
            result = prepare_geometry(config, dry_run=True)
            self.assertIn("--max-views 3", result["command"])
        finally:
            shutil.rmtree(tmp)

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

    def test_latent_inpaint_command_formats_iterations(self):
        tmp = tempfile.mkdtemp()
        try:
            config = {
                "project": {"output_dir": tmp},
                "dataset": {"type": "inpaint360gs", "root": tmp, "scene": "scene"},
                "checkpoints": {"latent_inpaint_weights": ""},
                "pipeline": {"latent_inpaint_iterations": 7},
                "external": {
                    "latent_inpaint_command": "python tools/inpaint_latent_context.py --latent-path {latent_path} --mask-path {mask_path} --output-path {output_path} --iterations {latent_inpaint_iterations}"
                },
            }
            result = run_latent_inpaint(config, dry_run=True)
            self.assertIn("--iterations 7", result["command"])
            self.assertFalse(result.get("fallback", False))
        finally:
            shutil.rmtree(tmp)

    def test_latent_inpaint_masked_denoiser_formats_weights_and_steps(self):
        tmp = tempfile.mkdtemp()
        try:
            w = "/path/to/masked_latent_denoiser.pt"
            config = {
                "project": {"output_dir": tmp},
                "dataset": {"type": "inpaint360gs", "root": tmp, "scene": "scene"},
                "checkpoints": {"latent_inpaint_weights": w},
                "pipeline": {"latent_inpaint_iterations": 12},
                "external": {
                    "latent_inpaint_command": (
                        "python tools/inpaint_latent_masked_denoiser.py "
                        "--weights {latent_inpaint_weights} "
                        "--latent-path {latent_path} --mask-path {mask_path} "
                        "--output-path {output_path} --refine-steps {latent_inpaint_iterations}"
                    )
                },
            }
            result = run_latent_inpaint(config, dry_run=True)
            self.assertIn("--weights %s" % w, result["command"])
            self.assertIn("--refine-steps 12", result["command"])
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
