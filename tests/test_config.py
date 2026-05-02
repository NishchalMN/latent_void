import os
import tempfile
import unittest

import yaml

from latent_void.config import load_config, validate_config


class ConfigTests(unittest.TestCase):
    def test_env_default_expansion_and_validation(self):
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        with open(path, "w") as handle:
            yaml.safe_dump({
                "project": {"output_dir": "${LV_TEST_OUT:-runs/test}", "device": "cpu"},
                "dataset": {"type": "inpaint360gs", "root": "/tmp/data", "scene": "scene"},
                "checkpoints": {
                    "diffsplat_root": "/tmp/diffsplat",
                    "gsvae_weights": "/tmp/gsvae.ckpt",
                    "sam3_root": "/tmp/sam3",
                },
                "pipeline": {"mask_threshold": 0.5, "latent_downsample": 8},
                "hpc": {"account": "msml612pcs3-class", "partition": "gpu-h100"},
            }, handle)
        try:
            config = load_config(path)
            self.assertEqual(config["project"]["output_dir"], "runs/test")
            self.assertTrue(validate_config(config))
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
