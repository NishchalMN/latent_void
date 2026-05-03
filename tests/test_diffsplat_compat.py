import os
import shutil
import tempfile
import unittest

from latent_void.diffsplat_compat import validate_aux_model_paths


class DiffSplatCompatTests(unittest.TestCase):
    def test_validate_aux_model_paths_requires_configs(self):
        tmp = tempfile.mkdtemp()
        try:
            sdxl = os.path.join(tmp, "sdxl")
            tiny = os.path.join(tmp, "tiny")
            os.makedirs(sdxl)
            os.makedirs(tiny)
            with self.assertRaises(RuntimeError):
                validate_aux_model_paths(sdxl, tiny)
            open(os.path.join(sdxl, "config.json"), "w").close()
            open(os.path.join(tiny, "config.json"), "w").close()
            self.assertIsNone(validate_aux_model_paths(sdxl, tiny))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
