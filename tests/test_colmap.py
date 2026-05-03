import os
import shutil
import tempfile
import unittest

from latent_void.datasets import DL3DVDataset, Inpaint360GSDataset, dataset_from_config


class ColmapTests(unittest.TestCase):
    def test_dataset_reads_colmap_text_camera(self):
        tmp = tempfile.mkdtemp()
        try:
            scene_dir = os.path.join(tmp, "scene")
            image_dir = os.path.join(scene_dir, "images")
            sparse_dir = os.path.join(scene_dir, "sparse", "0")
            os.makedirs(image_dir)
            os.makedirs(sparse_dir)
            open(os.path.join(image_dir, "IMG_0001.JPG"), "w").close()
            with open(os.path.join(sparse_dir, "cameras.txt"), "w") as handle:
                handle.write("1 PINHOLE 400 300 200 210 190 140\n")
            with open(os.path.join(sparse_dir, "images.txt"), "w") as handle:
                handle.write("1 1 0 0 0 1 2 3 1 IMG_0001.JPG\n")
                handle.write("\n")

            dataset = Inpaint360GSDataset({
                "dataset": {
                    "root": tmp,
                    "scene": "scene",
                    "images_glob": "images/*.{JPG,jpg}",
                    "cameras_path": "sparse/0",
                }
            })
            views = dataset.views()
            self.assertEqual(len(views), 1)
            self.assertEqual(views[0].camera["model"], "PINHOLE")
            self.assertEqual(views[0].camera["intrinsics"]["fxfycxcy"], [200.0, 210.0, 190.0, 140.0])
            self.assertEqual(views[0].camera["c2w"][0][3], -1.0)
        finally:
            shutil.rmtree(tmp)

    def test_dl3dv_loader_reads_json_frames(self):
        tmp = tempfile.mkdtemp()
        try:
            image_dir = os.path.join(tmp, "dl3dv_scene", "images")
            os.makedirs(image_dir)
            open(os.path.join(image_dir, "0001.png"), "w").close()
            with open(os.path.join(tmp, "dl3dv_scene", "transforms.json"), "w") as handle:
                handle.write('{"frames": [{"file_path": "images/0001.png", "camera": {"c2w": [[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,1]]}}]}')

            dataset = DL3DVDataset({
                "dataset": {
                    "root": tmp,
                    "scene": "dl3dv_scene",
                    "type": "dl3dv",
                    "images_glob": "images/*.png",
                }
            })
            views = dataset.views()
            self.assertEqual(len(views), 1)
            self.assertEqual(views[0].view_id, "0001")
            self.assertEqual(views[0].camera["c2w"][2][3], 1)
            self.assertTrue(dataset.summary()["has_cameras"])
            self.assertIsInstance(dataset_from_config({"dataset": {"type": "dl3dv", "root": tmp, "scene": "dl3dv_scene"}}), DL3DVDataset)
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
