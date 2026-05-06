# This file is part of inpaint360gs: Inpaint360GS: Efficient Object-Aware 3D Inpainting via Gaussian Splatting for 360° Scenes
# Project page: https://dfki-av.github.io/inpaint360gs/
#
# Copyright 2024-2026 Shaoxiang Wang
# Licensed under the Apache License, Version 2.0.
# http://www.apache.org/licenses/LICENSE-2.0
#
# This file contains original research code and modified components from the
# aforementioned projects. It is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
from argparse import ArgumentParser
from os import makedirs

import cv2
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene import Scene
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2
from utils.point_utils import create_point_cloud, get_intrinsics, ply_color_fusion
from utils.pose_utils import generate_ellipse_path


def fusion(dataset_path, model_path, name, iteration, views):
    render_path = os.path.join(model_path, name, f"ours_object_removal/iteration_{iteration}", "renders")
    depth_hole_path = os.path.join(model_path, name, f"ours_object_removal/iteration_{iteration}", "depth")
    depth_completed_path = os.path.join(model_path, name, f"ours_object_removal/iteration_{iteration}", "depth_completed")
    fused_mask_col_dep_ply_path = os.path.join(model_path, name, f"ours_object_removal/iteration_{iteration}", "fused_mask_col_dep_ply")
    fused_hole_col_dep_ply_path = os.path.join(model_path, name, f"ours_object_removal/iteration_{iteration}", "fused_hole_col_dep_ply")

    makedirs(fused_mask_col_dep_ply_path, exist_ok=True)
    makedirs(fused_hole_col_dep_ply_path, exist_ok=True)

    view = views[0]
    poses = generate_ellipse_path(views, n_frames=30, is_circle=True, circle_radius=args.circle_radius)
    virtual_poses_list = []
    for idx, pose in enumerate(tqdm(poses, desc="Prepare virtual camera poses")):
        view_tmp = copy.deepcopy(view)
        view_tmp.world_view_transform = torch.tensor(
            getWorld2View2(pose[:3, :3].T, pose[:3, 3], view.trans, view.scale)
        ).transpose(0, 1).cuda()
        view_tmp.full_proj_transform = (
            view_tmp.world_view_transform.unsqueeze(0).bmm(view.projection_matrix.unsqueeze(0))
        ).squeeze(0)
        view_tmp.camera_center = view_tmp.world_view_transform.inverse()[3, :3]
        view_tmp.image_name = f"{idx:05d}"
        view_tmp.R = pose[:3, :3].T
        view_tmp.T = pose[:3, 3]
        virtual_poses_list.append(view_tmp)

    for _, view in enumerate(tqdm(virtual_poses_list, desc="Color-Depth-Fusion progress")):
        if args.legacy_pose_rt:
            # Legacy path kept for A/B debugging.
            w2c = np.zeros((4, 4))
            w2c[:3, :3] = view.R.transpose()
            w2c[:3, 3] = view.T
            w2c[3, 3] = 1.0
            c2w = np.linalg.inv(w2c)
        else:
            # Use the same transform convention as renderer to avoid R/T transpose/sign mismatch.
            # world_view_transform is stored transposed in this codepath.
            w2c = view.world_view_transform.detach().cpu().numpy().T
            c2w = np.linalg.inv(w2c)
        intrinsics = get_intrinsics(view.image_height, view.image_width, view.FoVx, view.FoVy)

        extensions = [".jpg", ".JPG", ".png", ".PNG"]
        possible_paths = [
            os.path.join(dataset_path, "images_inpaint_unseen_virtual", view.image_name + ext)
            for ext in extensions
        ]
        inpainted_2d_color_path = next((path for path in possible_paths if os.path.exists(path)), None)
        if inpainted_2d_color_path is None:
            raise FileNotFoundError(f"File not found: {view.image_name} with extensions {extensions}")

        colors = cv2.imread(inpainted_2d_color_path).reshape(-1, 3)
        mask = cv2.imread(
            os.path.join(dataset_path, "inpaint_2d_unseen_mask_virtual", view.image_name + ".png"),
            cv2.IMREAD_GRAYSCALE,
        ).astype(bool).reshape(-1)
        depth_completed = np.load(os.path.join(depth_completed_path, view.image_name + ".npy"))
        points = create_point_cloud(depth_completed, intrinsics, c2w)
        ply_path = os.path.join(fused_mask_col_dep_ply_path, view.image_name + ".ply")
        ply_color_fusion(points, colors, ply_path, mask=mask)

        colors_hole = cv2.imread(os.path.join(render_path, view.image_name + ".png")).reshape(-1, 3)
        depth_hole = np.load(os.path.join(depth_hole_path, view.image_name + ".npy"))
        points_hole = create_point_cloud(depth_hole, intrinsics, c2w)
        ply_hole_path = os.path.join(fused_hole_col_dep_ply_path, view.image_name + ".ply")
        ply_color_fusion(points_hole, colors_hole, ply_hole_path)


def removal(dataset: ModelParams, iteration: int, pipeline: PipelineParams):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    with torch.no_grad():
        fusion(dataset.source_path, dataset.model_path, "virtual", scene.loaded_iter, scene.getTrainCameras())


if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    opt = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--legacy_pose_rt",
        action="store_true",
        help="Use legacy c2w from view.R/view.T instead of world_view_transform",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/object_removal/inpaint360/picnic.json",
        help="Path to the configuration file",
    )
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    with open(args.config_file, "r") as file:
        config = json.load(file)

    args.select_obj_id = config.get("select_obj_id")
    args.circle_radius = config.get("circle_radius")
    safe_state(args.quiet)
    removal(model.extract(args), args.iteration, pipeline.extract(args))
