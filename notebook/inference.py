# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# not ideal to put that here
os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]
os.environ["LIDRA_SKIP_INIT"] = "true"

import sys
from typing import Union, Optional, List, Callable
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
import torch
import math
import utils3d
import shutil
import subprocess
import seaborn as sns
from PIL import Image
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from copy import deepcopy
from kaolin.visualize import IpyTurntableVisualizer
from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics
import builtins

import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.model.backbone.trellis.utils import render_utils

from sam3d_objects.utils.visualization import SceneVisualizer

__all__ = ["Inference"]

# concatenate "li" + "dra" to skip the automated string replacement
if "li" + "dra" not in sys.modules:
    sys.modules["li" + "dra"] = sam3d_objects

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]


class Inference:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]] = None,
        seed: Optional[int] = None,
        pointmap=None,  # TODO(Pierre) : add pointmap type
    ) -> dict:
        # enable or disable layout model
        return self._pipeline.run(
            image,
            mask,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=True,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )


def _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitchs, rs, fovs):
    is_list = isinstance(yaws, list)
    if not is_list:
        yaws = [yaws]
        pitchs = [pitchs]
    if not isinstance(rs, list):
        rs = [rs] * len(yaws)
    if not isinstance(fovs, list):
        fovs = [fovs] * len(yaws)
    extrinsics = []
    intrinsics = []
    for yaw, pitch, r, fov in zip(yaws, pitchs, rs, fovs):
        fov = torch.deg2rad(torch.tensor(float(fov))).cuda()
        yaw = torch.tensor(float(yaw)).cuda()
        pitch = torch.tensor(float(pitch)).cuda()
        orig = (
            torch.tensor(
                [
                    torch.sin(yaw) * torch.cos(pitch),
                    torch.sin(pitch),
                    torch.cos(yaw) * torch.cos(pitch),
                ]
            ).cuda()
            * r
        )
        extr = utils3d.torch.extrinsics_look_at(
            orig,
            torch.tensor([0, 0, 0]).float().cuda(),
            torch.tensor([0, 1, 0]).float().cuda(),
        )
        intr = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
        extrinsics.append(extr)
        intrinsics.append(intr)
    if not is_list:
        extrinsics = extrinsics[0]
        intrinsics = intrinsics[0]
    return extrinsics, intrinsics


def render_video_ring(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2.0,  # radius of the ring
    pitch_deg=15,  # elevation angle above the xy-plane
    fov=40,
    **kwargs,
):
    # 1) angular positions around the ring
    yaws = torch.linspace(0, 2 * torch.pi, num_frames).tolist()

    # 2) constant pitch for every frame
    pitch = torch.full((num_frames,), torch.deg2rad(torch.tensor(pitch_deg))).tolist()

    # 3) look-at cameras centred at (0,0,0)
    extrinsics, intrinsics = _yaw_pitch_r_fov_to_extrinsics_intrinsics(
        yaws,
        pitch,
        r,
        fov,
    )

    # 4) render
    return render_utils.render_frames(
        sample,
        extrinsics,
        intrinsics,
        {"resolution": resolution, "bg_color": bg_color, "backend": "gsplat"},
        **kwargs,
    )


def render_video_flat(
    sample,
    resolution=512,
    bg_color=(0, 0, 0),
    num_frames=300,
    r=2.0,
    fov=40,
    pitch_deg=0,
    yaw_start_deg=-90,
    **kwargs,
):

    yaws = (
        torch.linspace(0, 2 * torch.pi, num_frames) + math.radians(yaw_start_deg)
    ).tolist()
    pitch = [math.radians(pitch_deg)] * num_frames

    extr, intr = _yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)

    return render_utils.render_frames(
        sample,
        extr,
        intr,
        {"resolution": resolution, "bg_color": bg_color, "backend": "gsplat"},
        **kwargs,
    )


def ready_gaussian_for_video_rendering(scene_gs, in_place=False, fix_alignment=False):
    if fix_alignment:
        scene_gs = _fix_gaussian_alignment(scene_gs, in_place=in_place)
    scene_gs = normalized_gaussian(scene_gs, in_place=fix_alignment)
    return scene_gs


def _fix_gaussian_alignment(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    device = scene_gs._xyz.device
    dtype = scene_gs._xyz.dtype
    scene_gs._xyz = (
        scene_gs._xyz
        @ torch.tensor(
            [
                [-1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            device=device,
            dtype=dtype,
        ).T
    )
    return scene_gs


def normalized_gaussian(scene_gs, in_place=False):
    if not in_place:
        scene_gs = deepcopy(scene_gs)

    orig_xyz = scene_gs._xyz
    orig_scale = scene_gs._scaling

    norm_scale = orig_scale / (orig_xyz.max(dim=0)[0] - orig_xyz.min(dim=0)[0]).max()
    norm_xyz = orig_xyz / (orig_xyz.max(dim=0)[0] - orig_xyz.min(dim=0)[0]).max()
    norm_xyz = norm_xyz - norm_xyz.min(dim=0)[0]
    scene_gs._xyz = norm_xyz
    scene_gs._scaling = norm_scale
    return scene_gs


def make_scene(*outputs, in_place=False):
    if not in_place:
        outputs = [deepcopy(output) for output in outputs]

    all_outs = []
    for output in outputs:
        # move gaussians to scene frame of reference
        PC = SceneVisualizer.object_pointcloud(
            points_local=(output["gaussian"][0]._xyz - 0.5).unsqueeze(0),
            quat_l2c=output["rotation"],
            trans_l2c=output["translation"],
            scale_l2c=output["scale"],
        )
        output["gaussian"][0]._xyz = PC.points_list()[0]
        output["gaussian"][0]._scaling *= output["scale"]
        all_outs.append(output)

    # merge gaussians
    scene_gs = all_outs[0]["gaussian"][0]
    for out in all_outs[1:]:
        out_gs = out["gaussian"][0]
        scene_gs._xyz = torch.cat([scene_gs._xyz, out_gs._xyz], dim=0)
        scene_gs._features_dc = torch.cat(
            [scene_gs._features_dc, out_gs._features_dc], dim=0
        )
        scene_gs._scaling = torch.cat([scene_gs._scaling, out_gs._scaling], dim=0)
        scene_gs._rotation = torch.cat([scene_gs._rotation, out_gs._rotation], dim=0)
        scene_gs._opacity = torch.cat([scene_gs._opacity, out_gs._opacity], dim=0)

    return scene_gs


def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    mask = load_image(path)
    mask = mask > 127
    mask = mask[..., 0]
    return mask


def load_all_masks(folder_path):
    idx = 0
    masks = []
    while os.path.exists(mask_path := os.path.join(folder_path, f"{idx}.jpg")):
        mask = load_mask(mask_path)
        masks.append(mask)
        idx += 1
    return masks


def display_image(image, masks=None):
    def imshow(image, ax):
        ax.axis("off")
        ax.imshow(image)

    grid = (1, 1) if masks is None else (2, 2)
    fig, axes = plt.subplots(*grid)
    if masks is not None:
        mask_colors = sns.color_palette("husl", len(masks))
        black_image = np.zeros_like(image[..., :3], dtype=float)  # background
        mask_display = np.copy(black_image)
        mask_union = np.zeros_like(image[..., :3])
        for i, mask in enumerate(masks):
            mask_display[mask] = mask_colors[i]
            mask_union |= mask[..., None]
        imshow(black_image, axes[0, 1])
        imshow(mask_display, axes[1, 0])
        imshow(image * mask_union, axes[1, 1])

    image_axe = axes if masks is None else axes[0, 0]
    imshow(image, image_axe)

    fig.tight_layout(pad=0)
    fig.show()


def interactive_visualizer(ply_path):
    with gr.Blocks() as demo:
        gr.Markdown("# 3D Gaussian Splatting (black-screen loading might take a while)")
        gr.Model3D(
            value=ply_path,  # splat file
            label="3D Scene",
        )
    demo.launch(share=True)
