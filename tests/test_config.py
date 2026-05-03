import os
import tempfile
import unittest

import yaml

from latent_void.config import ConfigError, load_config, validate_config


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

    def test_strict_paths_require_aux_vae_configs(self):
        tmp = tempfile.mkdtemp()
        try:
            for name in ["data", "diffsplat", "gsrecon", "gsvae", "sam3", "sdxl", "tiny"]:
                os.makedirs(os.path.join(tmp, name))
            config = {
                "project": {"output_dir": os.path.join(tmp, "out"), "device": "cpu"},
                "dataset": {"type": "inpaint360gs", "root": os.path.join(tmp, "data"), "scene": "scene"},
                "checkpoints": {
                    "diffsplat_root": os.path.join(tmp, "diffsplat"),
                    "gsrecon_weights": os.path.join(tmp, "gsrecon"),
                    "gsvae_weights": os.path.join(tmp, "gsvae"),
                    "sam3_root": os.path.join(tmp, "sam3"),
                    "sdxl_vae_path": os.path.join(tmp, "sdxl"),
                    "tiny_vae_path": os.path.join(tmp, "tiny"),
                },
                "pipeline": {"mask_threshold": 0.5, "latent_downsample": 8},
                "hpc": {"account": "msml612pcs3-class", "partition": "gpu-h100"},
            }
            with self.assertRaises(ConfigError):
                validate_config(config, strict_paths=True)
            open(os.path.join(tmp, "sdxl", "config.json"), "w").close()
            open(os.path.join(tmp, "tiny", "config.json"), "w").close()
            self.assertTrue(validate_config(config, strict_paths=True))
        finally:
            import shutil
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
