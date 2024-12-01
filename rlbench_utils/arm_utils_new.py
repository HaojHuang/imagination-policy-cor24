from functools import reduce as funtool_reduce
from operator import mul
#from rlbench.backend.utils import Observation
import torch
from torch import nn, einsum
import torch.nn.functional as F
import numpy as np
#from rlbench.backend.utils import extract_obs
MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False
from scipy.spatial.transform import Rotation
import pyrender
import trimesh
from pyrender.trackball import Trackball
#from rlbench.demo import Demo
from typing import List
#from agent_3d.so3_utils import _so3_hopf_uniform_grid
import matplotlib.pyplot as plt
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
sys.path.append('..')
#from utils.load_saved_dataset import scene_bound

#VOXEL_S = 88

def set_task_voxel_info(task,variation,sv,fine=False):
    # the voxel_size follow xyz, which will be -z-yx in pytorch
    if not sv:
        SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
        voxel_x = 88
        voxel_y = 88
        voxel_z = 88
        voxel_size = np.asarray([voxel_x, voxel_y, voxel_z])
        range = np.asarray(SCENE_BOUNDS[3:]) - np.asarray(SCENE_BOUNDS[:3])
        res = range / voxel_size
        return np.asarray(SCENE_BOUNDS), voxel_size, res
    else:
        if variation==0:
            # SCENE_BOUNDS = [-0.274+0.2, -0.655+0.2, 0.75, 0.77-0.2+0.03475, 0.65-0.2,0.75+0.5+0.028]
            # voxel_x = 72
            # voxel_y = 96
            # voxel_z = 56
            SCENE_BOUNDS = [-0.274 + 0.2, -0.655 + 0.2, 0.75, 0.77 - 0.2, 0.65 - 0.2, 0.75 + 0.5]
            if fine:
                voxel_x = 72# + 8
                voxel_y = 96# + 8
                voxel_z = 56# + 8
            else:
                voxel_x = 72
                voxel_y = 96
                voxel_z = 56
            voxel_size = np.asarray([voxel_x, voxel_y, voxel_z])
            range_ = np.asarray(SCENE_BOUNDS[3:]) - np.asarray(SCENE_BOUNDS[:3])
            res = range_ / voxel_size
            max_res_axis = np.argmax(res)
            max_res = np.max(res)
            for i in np.arange(3):
                if i != max_res_axis:
                    SCENE_BOUNDS[i + 3] = SCENE_BOUNDS[i] + max_res * voxel_size[i]
            range = np.asarray(SCENE_BOUNDS[3:]) - np.asarray(SCENE_BOUNDS[:3])
            res = range / voxel_size
            return np.asarray(SCENE_BOUNDS), voxel_size, res

REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces',
               'task_low_dim_state', 'misc']

class Observation(object):
    """Storage for both visual and low-dimensional observations."""

    def __init__(self,
                 left_shoulder_rgb: np.ndarray,
                 left_shoulder_depth: np.ndarray,
                 left_shoulder_mask: np.ndarray,
                 left_shoulder_point_cloud: np.ndarray,
                 right_shoulder_rgb: np.ndarray,
                 right_shoulder_depth: np.ndarray,
                 right_shoulder_mask: np.ndarray,
                 right_shoulder_point_cloud: np.ndarray,
                 overhead_rgb: np.ndarray,
                 overhead_depth: np.ndarray,
                 overhead_mask: np.ndarray,
                 overhead_point_cloud: np.ndarray,
                 wrist_rgb: np.ndarray,
                 wrist_depth: np.ndarray,
                 wrist_mask: np.ndarray,
                 wrist_point_cloud: np.ndarray,
                 front_rgb: np.ndarray,
                 front_depth: np.ndarray,
                 front_mask: np.ndarray,
                 front_point_cloud: np.ndarray,
                 joint_velocities: np.ndarray,
                 joint_positions: np.ndarray,
                 joint_forces: np.ndarray,
                 gripper_open: float,
                 gripper_pose: np.ndarray,
                 gripper_matrix: np.ndarray,
                 gripper_joint_positions: np.ndarray,
                 gripper_touch_forces: np.ndarray,
                 task_low_dim_state: np.ndarray,
                 misc: dict):
        self.left_shoulder_rgb = left_shoulder_rgb
        self.left_shoulder_depth = left_shoulder_depth
        self.left_shoulder_mask = left_shoulder_mask
        self.left_shoulder_point_cloud = left_shoulder_point_cloud
        self.right_shoulder_rgb = right_shoulder_rgb
        self.right_shoulder_depth = right_shoulder_depth
        self.right_shoulder_mask = right_shoulder_mask
        self.right_shoulder_point_cloud = right_shoulder_point_cloud
        self.overhead_rgb = overhead_rgb
        self.overhead_depth = overhead_depth
        self.overhead_mask = overhead_mask
        self.overhead_point_cloud = overhead_point_cloud
        self.wrist_rgb = wrist_rgb
        self.wrist_depth = wrist_depth
        self.wrist_mask = wrist_mask
        self.wrist_point_cloud = wrist_point_cloud
        self.front_rgb = front_rgb
        self.front_depth = front_depth
        self.front_mask = front_mask
        self.front_point_cloud = front_point_cloud
        self.joint_velocities = joint_velocities
        self.joint_positions = joint_positions
        self.joint_forces = joint_forces
        self.gripper_open = gripper_open
        self.gripper_pose = gripper_pose
        self.gripper_matrix = gripper_matrix
        self.gripper_joint_positions = gripper_joint_positions
        self.gripper_touch_forces = gripper_touch_forces
        self.task_low_dim_state = task_low_dim_state
        self.misc = misc

    def get_low_dim_data(self) -> np.ndarray:
        """Gets a 1D array of all the low-dimensional obseervations.

        :return: 1D array of observations.
        """
        low_dim_data = [] if self.gripper_open is None else [[self.gripper_open]]
        for data in [self.joint_velocities, self.joint_positions,
                     self.joint_forces,
                     self.gripper_pose, self.gripper_joint_positions,
                     self.gripper_touch_forces, self.task_low_dim_state]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])


def extract_obs(obs: Observation,
                cameras,
                t: int = 0,
                prev_action=None,
                channels_last: bool = False):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
      obs.gripper_joint_positions = np.clip(
        obs.gripper_joint_positions, 0., 0.04)

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}
    # for k, v in obs_dict.items():
    #   print('raw data', k)
    robot_state = np.array([
      obs.gripper_open,
      *obs.gripper_joint_positions])
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}

    if not channels_last:
      # swap channels from last dim to 1st dim
      obs_dict = {k: np.transpose(
        v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
                  for k, v in obs_dict.items()}
    else:
      # add extra dim to depth data
      obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                  for k, v in obs_dict.items()}
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    # obs_dict['ignore_collisions'] = np.array([obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
      obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
      obs_dict['%s_camera_extrinsics' % camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
      obs_dict['%s_camera_intrinsics' % camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

    # add timestep to low_dim_state
    episode_length = 10  # TODO fix this
    time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['low_dim_state'] = np.concatenate(
      [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict

class Demo(object):

    def __init__(self, observations, random_seed=None):
        self._observations = observations
        self.random_seed = random_seed

    def __len__(self):
        return len(self._observations)

    def __getitem__(self, i):
        return self._observations[i]

    def restore_state(self):
        np.random.set_state(self.random_seed)

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    # print('==',obs.joint_velocities)
    jv = np.asarray(obs.joint_velocities, dtype=float)
    small_delta = np.allclose(jv, 0., atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def _keypoint_discovery(demo: Demo,
                        stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # if change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                       last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    print('Found %d keypoints.' % len(episode_keypoints), episode_keypoints)
    return episode_keypoints


class VoxelGrid(nn.Module):

    def __init__(self,
                 coord_bounds,
                 voxel_size,
                 device,
                 batch_size,
                 feature_size,
                 max_num_coords: int, ):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = voxel_size.reshape(-1).tolist()
        #print(self._voxel_shape,'==========')
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = torch.tensor(self._voxel_shape,
                                              device=device).unsqueeze(
            0) + 2  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(coord_bounds, dtype=torch.float,
                                          device=device).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [torch.tensor([batch_size], device=device), max_dims,
             torch.tensor([4 + feature_size], device=device)], -1).tolist()
        self._ones_max_coords = torch.ones((batch_size, max_num_coords, 1),
                                           device=device)
        self._num_coords = max_num_coords

        shape = self._total_dims_list

        self._result_dim_sizes = torch.tensor(
            [funtool_reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [
                1], device=device)
        flat_result_size = funtool_reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float,
                                         device=device)
        self._flat_output = torch.ones(flat_result_size, dtype=torch.float,
                                       device=device) * self._initial_val
        self._arange_to_max_coords = torch.arange(4 + feature_size,
                                                  device=device)
        self._flat_zeros = torch.zeros(flat_result_size, dtype=torch.float,
                                       device=device)

        self._const_1 = torch.tensor(1.0, device=device)
        self._batch_size = batch_size

        # Coordinate Bounds:
        self._bb_mins = self._coord_bounds[..., 0:3]
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - self._bb_mins
        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = self._voxel_shape_spec.int()
        self._dims_orig = dims_orig = self._voxel_shape_spec.int() - 2
        self._dims_m_one = (dims - 1).int()
        # BS x 1 x 3
        self._res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)

        self._voxel_indicy_denmominator = self._res + MIN_DENOMINATOR
        self._dims_m_one_zeros = torch.zeros_like(self._dims_m_one)

        batch_indices = torch.arange(self._batch_size, dtype=torch.int,
                                     device=device).view(self._batch_size, 1, 1)
        self._tiled_batch_indices = batch_indices.repeat(
            [1, self._num_coords, 1])

        wx = self._voxel_shape[0] + 2
        arangex = torch.arange(0, wx, dtype=torch.float, device=device)
        wy = self._voxel_shape[1] + 2
        arangey = torch.arange(0, wy, dtype=torch.float, device=device)
        wz = self._voxel_shape[2] + 2
        arangez = torch.arange(0, wz, dtype=torch.float, device=device)
        self._index_grid = torch.cat([
            arangex.view(wx, 1, 1, 1).repeat([1, wy, wz, 1]),
            arangey.view(1, wy, 1, 1).repeat([wx, 1, wz, 1]),
            arangez.view(1, 1, wz, 1).repeat([wx, wy, 1, 1])], dim=-1).unsqueeze(
            0).repeat([self._batch_size, 1, 1, 1, 1])

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(self, src: torch.Tensor, index: torch.Tensor, out: torch.Tensor,
                      dim: int = -1):
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims])
        indices_for_flat_tiled = ((indices * indices_scales).sum(
            dim=-1, keepdims=True)).view(-1, 1).repeat(
            *[1, self._voxel_feature_size])

        implicit_indices = self._arange_to_max_coords[
                           :self._voxel_feature_size].unsqueeze(0).repeat(
            *[indices_for_flat_tiled.shape[0], 1])
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates, flat_indices_for_flat,
            out=torch.zeros_like(self._flat_output))
        return flat_scatter.view(self._total_dims_list)

    def coords_to_bounding_voxel_grid(self, coords, coord_features=None,
                                      coord_bounds=None):
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins
        if coord_bounds is not None:
            bb_mins = coord_bounds[..., 0:3]
            bb_maxs = coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR)
            #print(res)
            voxel_indicy_denmominator = res + MIN_DENOMINATOR

        bb_mins_shifted = bb_mins - res  # shift back by one
        floor = torch.floor(
            (coords - bb_mins_shifted.unsqueeze(1)) / voxel_indicy_denmominator.unsqueeze(1)).int()
        voxel_indices = torch.min(floor, self._dims_m_one)
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros)

        # global-coordinate point cloud (x, y, z)
        voxel_values = coords

        # rgb values (R, G, B)
        if coord_features is not None:
            voxel_values = torch.cat([voxel_values, coord_features], -1)  # concat rgb values (B, 128, 128, 3)

        # coordinates to aggregate over
        _, num_coords, _ = voxel_indices.shape
        all_indices = torch.cat([
            self._tiled_batch_indices[:, :num_coords], voxel_indices], -1)

        # max coordinates
        voxel_values_pruned_flat = torch.cat(
            [voxel_values, self._ones_max_coords[:, :num_coords]], -1)

        # aggregate across camera views
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size))

        vox = scattered[:, 1:-1, 1:-1, 1:-1]
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (res_centre + bb_mins_shifted.unsqueeze(
                1).unsqueeze(1).unsqueeze(1))[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat([vox[..., :-1], coord_positions, vox[..., -1:]], -1)

        # occupied value
        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([
            vox[..., :-1], occupied], -1)
        # print(vox.shape)
        # print(self._index_grid.shape)
        # hard voxel-location position encoding
        return torch.cat(
            [vox[..., :-1], self._index_grid[:, :-2, :-2, :-2] / self._voxel_d,
             vox[..., -1:]], -1)


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)

def _preprocess_inputs(replay_sample,CAMERAS):
    obs, pcds = [], []
    for n in CAMERAS:
        rgb = stack_on_channel(replay_sample['%s_rgb' % n])
        pcd = stack_on_channel(replay_sample['%s_point_cloud' % n])
        rgb = _norm_rgb(rgb)
        obs.append([rgb, pcd])  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds

def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0

def point_to_voxel_index(
        point: np.ndarray,
        voxel_size: np.ndarray,
        coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = voxel_size - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / voxel_size
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    #print('voxel_indicy',voxel_indicy)
    return voxel_indicy

def voxel_index_to_point(
        voxel_idx: np.ndarray,
        voxel_size: np.ndarray,
        coord_bounds: np.ndarray):
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / voxel_size
    trans = voxel_idx*res + bb_mins + res/2
    return trans

def _compute_initial_camera_pose(scene):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    centroid = scene.centroid
    scale = scene.scale
    # if scale == 0.0:
    #     scale = DEFAULT_SCENE_SCALE
    scale = 4.0
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    return cp


def _from_trimesh_scene(
        trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                  for name, geom in trimesh_scene.geometry.items()}
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr


def create_voxel_scene(
        voxel_grid: np.ndarray,
        q_attention: np.ndarray = None,
        highlight_coordinate: np.ndarray = None,
        highlight_gt_coordinate: np.ndarray = None,
        highlight_alpha: float = 1.0,
        voxel_size: float = 0.1,
        show_bb: bool = False,
        alpha: float = 0.5):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    rgb = np.concatenate([(v[:, :, :, 3:6] + 1) / 2.0, alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = (q > 0.75)
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate([
            q, np.zeros_like(q), np.zeros_like(q),
            np.clip(q, 0, 1)], axis=-1)
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha]

    if highlight_gt_coordinate is not None:
        x, y, z = highlight_gt_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 0.0, 1.0, highlight_alpha]

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0))
    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform)
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    # if show_bb:
    #     assert d == h == w
    #     _create_bounding_box(scene, voxel_size, d)
    return scene




def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def visualise_voxel(voxel_grid: np.ndarray,
                    q_attention: np.ndarray = None,
                    highlight_coordinate: np.ndarray = None,
                    highlight_gt_coordinate: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = False,
                    voxel_size: float = 0.1,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = False,
                    alpha: float = 0.5,
                    render_gripper=False,
                    gripper_pose=None,
                    gripper_mesh_scale=1.0):
    scene = create_voxel_scene(
        voxel_grid, q_attention, highlight_coordinate, highlight_gt_coordinate,
        highlight_alpha, voxel_size,
        show_bb, alpha)
    gripper_trimesh = trimesh.load('./meshes/hand.dae', force='mesh')
    #gripper_trimesh = as_mesh(gripper_trimesh)
    gripper_trimesh.vertices *= gripper_mesh_scale
    radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
    gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                      color_map='winter')
    scene.add_geometry(gripper_trimesh,transform=np.array(gripper_pose))
    frame = trimesh.creation.axis(axis_length=2)
    scene.add_geometry(frame)
    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=1920, viewport_height=1080, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width / r.viewport_height)
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)

        if render_gripper:
            gripper_trimesh = trimesh.load('../meshes/hand.dae', force='mesh')
            gripper_trimesh.vertices *= gripper_mesh_scale
            radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
            gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                              color_map='winter')
            gripper_mesh = pyrender.Mesh.from_trimesh(gripper_trimesh, poses=np.array([gripper_pose]), smooth=False)
            s.add(gripper_mesh)

        color, depth = r.render(s)
        return color.copy()



def pyt_pose_2_sim(pose,voxel_size,scene_bounds,idx=1,N=180,so3_points=None):
    trans_indice = np.asarray(pose[0])
    angle_idx = pose[idx]
    trans_indice = reindex_predic_for_sim(trans_indice,voxel_size=voxel_size)
    continu_trans = voxel_index_to_point(trans_indice,voxel_size,coord_bounds=scene_bounds)
    if so3_points is None:
        so3_points = _so3_hopf_uniform_grid(N)
    ori = so3_points[angle_idx]
    return continu_trans, ori

def pyt_place_pose_2_sim(pose0,pose1,voxel_size,scene_bounds,N=180,so3_points=None,p_idx=1,q_idx=1):
    trans_indice = np.asarray(pose1[0])
    angle_idx0 = pose0[p_idx]
    angle_idx1 = pose1[q_idx]
    trans_indice = reindex_predic_for_sim(trans_indice,voxel_size=voxel_size)
    continu_trans = voxel_index_to_point(trans_indice,voxel_size,coord_bounds=scene_bounds)
    if so3_points is None:
        so3_points = _so3_hopf_uniform_grid(N)
    ori_pick = so3_points[angle_idx0]
    #print(so3_points.as_matrix().shape, angle_idx1)
    ori = so3_points[angle_idx1]*ori_pick
    return continu_trans, ori


def visualise_voxel_paired_action(voxel_grid: np.ndarray,
                    q_attention: np.ndarray = None,
                    highlight_coordinate: np.ndarray = None,
                    highlight_gt_coordinate: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = False,
                    voxel_size: float = 0.1,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = False,
                    alpha: float = 0.5,
                    render_gripper=False,
                    gripper_pose=None,
                    gripper_pose2 = None,
                    gripper_mesh_scale=1.0):
    scene = create_voxel_scene(
        voxel_grid, q_attention, highlight_coordinate, highlight_gt_coordinate,
        highlight_alpha, voxel_size,
        show_bb, alpha)
    # if show:
    #     scene.show()
    #gripper mesh for the pick pose
    gripper_trimesh = trimesh.load('../meshes/hand.dae',force='mesh')
    #gripper_trimesh = as_mesh(gripper_trimesh)
    gripper_trimesh.vertices *= gripper_mesh_scale
    radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
    gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                      color_map='winter')

    scene.add_geometry(gripper_trimesh, transform=np.array(gripper_pose))

    #mesh for the place pose
    gripper_trimesh2 = trimesh.load('../meshes/hand.dae', force='mesh')
    gripper_trimesh2.vertices *= gripper_mesh_scale
    gripper_trimesh2.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                      color_map='Reds')
    scene.add_geometry(gripper_trimesh2, transform=np.array(gripper_pose2))
    frame = trimesh.creation.axis(axis_length=2)
    scene.add_geometry(frame)

    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=1920, viewport_height=1080, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width / r.viewport_height)
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)

        if render_gripper:
            gripper_trimesh = trimesh.load('../meshes/hand.dae', force='mesh')
            gripper_trimesh.vertices *= gripper_mesh_scale
            radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
            gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                              color_map='winter')
            gripper_mesh = pyrender.Mesh.from_trimesh(gripper_trimesh, poses=np.array([gripper_pose]), smooth=False)
            s.add(gripper_mesh)

        color, depth = r.render(s)
        return color.copy()


def create_voxel_scene_4_dim(
        voxel_grid: np.ndarray,
        q_attention: np.ndarray = None,
        highlight_coordinate: np.ndarray = None,
        highlight_gt_coordinate: np.ndarray = None,
        highlight_alpha: float = 1.0,
        voxel_size: float = 0.1,
        show_bb: bool = False,
        alpha: float = 0.5,
        normalized=True):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    #v[:,:,:,:3] = 2*(v[:,:,:,:3]-0.5)
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    # if normalized:
    #     rgb = np.concatenate([(v[:, :, :, :3]+1.0)/2, alpha], axis=-1)
    # else:

    rgb = np.concatenate([v[:, :, :, :3], alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = (q > 0.75)
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate([
            q, np.zeros_like(q), np.zeros_like(q),
            np.clip(q, 0, 1)], axis=-1)
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)
    #print(occupancy.shape)
    # plot the center
    sx, sy, sz = occupancy.shape
    #occupancy[sx//2, sy//2, sz//2] = True
    #rgb[sx//2, sy//2, sz//2] = [0.0, 1.0, 0.0, highlight_alpha]

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha]
        pass

    if highlight_gt_coordinate is not None:
        x, y, z = highlight_gt_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 0.0, 1.0, highlight_alpha]
        pass

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0))


    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform)
    #print(trimesh_voxel_grid.scale,'scale')
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    # if show_bb:
    #     assert d == h == w
    #     _create_bounding_box(scene, voxel_size, d)
    return scene

def create_voxel_scene_q(
        voxel_grid: np.ndarray,
        q_attention: np.ndarray = None,
        highlight_coordinate: np.ndarray = None,
        highlight_gt_coordinate: np.ndarray = None,
        highlight_pick: np.ndarray = None,
        highlight_alpha: float = 1.0,
        voxel_size: float = 0.1,
        show_bb: bool = False,
        alpha: float = 0.8,
        normalized=True):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    if normalized:
        rgb = np.concatenate([(v[:, :, :, :3]+1.0)/2, alpha], axis=-1)
    else:
        rgb = np.concatenate([v[:, :, :, :3], alpha], axis=-1)

    if q_attention is not None:
        assert len(q_attention.shape)==3
        # print('11')
        # print(q_attention.shape)
        #q = np.max(q_attention, 0)


        q = q_attention
        q = q_attention / np.max(q_attention)
        #print(q.shape)

        #show_q = np.logical_and(0.035<q, q< 0.038)
        #show_q = (q>0.8)
        #print(show_q.sum())

        #occupancy = (show_q + occupancy).astype(bool)

        #occupancy = show_q.astype(bool)
        q = np.expand_dims(q, -1)  # Max q can be is 0.9
        print(q.shape)
        #print(q)

        q_rgb = np.concatenate([
            np.abs(q),np.abs(q), np.abs(q),
            np.clip(q, 0, 1)], axis=-1)

        q_rgb = np.concatenate([
             np.zeros_like(q), np.abs(q), np.abs(q), np.zeros_like(q)+0.2 ], axis=-1)
        print(q.shape)
        #rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)
        rgb = q_rgb
    #print(occupancy.shape)
    # plot the center
    # sx, sy, sz = occupancy.shape
    # occupancy[sx//2, sy//2, sz//2] = True
    # rgb[sx//2, sy//2, sz//2] = [0.0, 1.0, 0.0, highlight_alpha]

    # plot pick location
    if highlight_coordinate is not None:
        x, y, z = highlight_pick
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 1.0, 0.0, highlight_alpha]

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [1.0, 0.0, 0.0, highlight_alpha]

    if highlight_gt_coordinate is not None:
        x, y, z = highlight_gt_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 0.0, 1.0, highlight_alpha]

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0))
    trimesh_voxel_grid = trimesh.voxel.VoxelGrid(
        encoding=occupancy, transform=transform)
    #print(trimesh_voxel_grid.scale,'scale')
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    # if show_bb:
    #     assert d == h == w
    #     _create_bounding_box(scene, voxel_size, d)
    return scene



def vis_voxel_q(input, qattention=None,highlight_coordinate=None,highlight_gt_coordinate=None,highlight_pick=None):
    '''
    signal: numpy 5 D array --> reindex to xyz --> render with trimesh
    '''
    secen_bounds, voxel_size, resolution = set_task_voxel_info('stack_blocks', sv=True,variation=0)
    if qattention is not None:
        assert len(qattention.shape)==5
        qattention = reindex_output_for_simulator(qattention)
        qattention = qattention[0][0]
    if highlight_gt_coordinate is not None:
        highlight_gt_coordinate = reindex_predic_for_sim(highlight_gt_coordinate,voxel_size)
    if highlight_coordinate is not None:
        highlight_coordinate = reindex_predic_for_sim(highlight_coordinate, voxel_size)
    if highlight_pick is not None:
        highlight_pick = reindex_predic_for_sim(highlight_pick, voxel_size)

    signal = reindex_output_for_simulator(input)
    scene = create_voxel_scene_q(signal[0],qattention,highlight_coordinate,highlight_gt_coordinate,highlight_pick)
    frame = trimesh.creation.axis(axis_length=2)
    #scene.add_geometry(frame)
    scene.show()


def vis_voxel(signal,reindex=True):
    '''
    signal: numpy 5 D array --> reindex to xyz --> render with trimesh
    '''
    if reindex:
        signal = reindex_output_for_simulator(signal)

    scene = create_voxel_scene_4_dim(signal[0],normalized=False)
    frame = trimesh.creation.axis(axis_length=2)
    scene.add_geometry(frame)
    scene.show()

def vis_kernel(signal, qattention = None, reindex=True):
    '''
    signal: numpy 5 D array --> reindex to xyz --> render with trimesh
    '''
    if reindex:
        signal = reindex_output_for_simulator(signal)
        assert len(qattention.shape)==5
        qattention = reindex_output_for_simulator(qattention)
        qattention = qattention[0][0]
    scene = create_voxel_scene_q(signal[0], qattention)
    #scene = create_voxel_scene_4_dim(signal[0])
    #frame = trimesh.creation.axis(axis_length=2)
    #scene.add_geometry(frame)
    scene.show()

def visualise_samples(sample,
                      bounds,
                      voxel_s,
                      N=180,
                      res = None,
                    q_attention: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = True,
                    voxel_size: float = 0.045,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = False,
                    alpha: float = 0.5,
                    delta=False,
                    so3_points = None,
                    p_idx =1,
                    q_idx =1,
                    normalized=True,
                    ):

    voxel_grid = sample[0]
    #voxel_grid[:,:3,:,:,:] = (voxel_grid[:,:3,:,:,:]-0.5)*2
    voxel_grid = reindex_output_for_simulator(voxel_grid)
    # voxel_grid[:,-1,1,1,1] =1.
    # voxel_grid[:,-1,1,20,-1] =1
    voxel_grid = voxel_grid[0]
    p0 = sample[1][0]

    if type(p0) is not np.array:
        p0 = np.asarray(p0)
    idx0 = sample[1][p_idx]

    p1 = sample[2][0]

    if type(p1) is not np.array:
        p1 = np.asarray(p1)
    idx1 = sample[2][q_idx]

    #print(p0, p1)
    p0 = reindex_predic_for_sim(p0,voxel_size=voxel_s)
    p1 = reindex_predic_for_sim(p1,voxel_size=voxel_s)
    #print(p0,p1)
    #print(voxel_grid.shape)
    continu_p0 = voxel_index_to_point(p0,voxel_s,bounds)
    continu_p1 = voxel_index_to_point(p1,voxel_s,bounds)
    #print('continus_po',continu_p0)
    #print('continus_p1',continu_p1)

    if so3_points is None:
        so3_points = _so3_hopf_uniform_grid(N=N)
    #print(idx0,idx1)
    rot0 = so3_points[idx0]
    rot1 = so3_points[idx1]
    #rot1 is the delta

    if delta==True:
        rot1 = rot1*rot0

    quat0 = rot0.as_quat()
    quat1 = rot1.as_quat()

    voxel_scale = voxel_size*(1./res) # # of voxel per 1 meters
    print(res,1./res,'===')
    gripper_pos_mat0 = get_gripper_render_pose(voxel_scale,bounds[:3],continu_p0,quat0)
    gripper_pos_mat1 = get_gripper_render_pose(voxel_scale,bounds[:3],continu_p1,quat1)
    #print(gripper_pos_mat0)

    scene = create_voxel_scene_4_dim(
        voxel_grid, q_attention, p1, p0,
        highlight_alpha, voxel_size,
        show_bb, alpha,normalized=normalized)

    gripper_mesh_scale = voxel_size*100
    #gripper mesh for the pick pose
    gripper_trimesh = trimesh.load('./meshes/hand.dae',force='mesh')
    #gripper_trimesh = as_mesh(gripper_trimesh)
    gripper_trimesh.vertices *= gripper_mesh_scale
    radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
    gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                      color_map='winter')

    scene.add_geometry(gripper_trimesh, transform=np.array(gripper_pos_mat0))

    #mesh for the place pose
    gripper_trimesh2 = trimesh.load('./meshes/hand.dae', force='mesh')
    gripper_trimesh2.vertices *= gripper_mesh_scale
    gripper_trimesh2.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                      color_map='Reds')
    scene.add_geometry(gripper_trimesh2, transform=np.array(gripper_pos_mat1))
    frame = trimesh.creation.axis(axis_length=1)
    scene.add_geometry(frame)

    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=1920, viewport_height=1080, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width / r.viewport_height)
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)

        color, depth = r.render(s)
        return color.copy()


def visualise_samples_unchange(sample,
                      bounds,
                      voxel_s,
                      N=180,
                      res = None,
                    q_attention: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = True,
                    voxel_size: float = 0.045,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = False,
                    alpha: float = 0.5,
                    delta=False,
                    so3_points = None,
                    p_idx =1,
                    q_idx =1,
                    normalized=True,
                    sample_signal=None,
                    ):
    voxel_grid = sample[0]
    #voxel_grid = voxel_grid.numpy()
    voxel_grid = voxel_grid[0]
    p0 = sample[1][0]
    if type(p0) is not np.array:
        p0 = np.asarray(p0)
    idx0 = sample[1][p_idx]

    p1 = sample[2][0]
    if type(p1) is not np.array:
        p1 = np.asarray(p1)
    idx1 = sample[2][q_idx]
    #print(p0, p1)
    p0 = reindex_predic_for_sim(p0,voxel_size=voxel_s)
    p1 = reindex_predic_for_sim(p1,voxel_size=voxel_s)
    #print(p0,p1)
    #print(voxel_grid.shape)

    continu_p0 = voxel_index_to_point(p0,voxel_s,bounds)
    continu_p1 = voxel_index_to_point(p1,voxel_s,bounds)
    #print('continus_po',continu_p0)
    #print('continus_p1',continu_p1)
    if so3_points is None:
        so3_points = _so3_hopf_uniform_grid(N=N)
    #print(idx0,idx1)
    rot0 = so3_points[idx0]
    rot1 = so3_points[idx1]
    #rot1 is the delta
    if delta==True:
        rot1 = rot1*rot0

    quat0 = rot0.as_quat()
    quat1 = rot1.as_quat()

    voxel_scale = voxel_size*(1./res) # # of voxel per 1 meters
    print(res,1./res,'===')
    gripper_pos_mat0 = get_gripper_render_pose(voxel_scale,bounds[:3],continu_p0,quat0)
    gripper_pos_mat1 = get_gripper_render_pose(voxel_scale,bounds[:3],continu_p1,quat1)
    #print(gripper_pos_mat0)

    p1 = None
    p0 = None
    q_attention = None
    scene = create_voxel_scene_4_dim(
        voxel_grid, q_attention, p1, p0,
        highlight_alpha, voxel_size,
        show_bb, alpha,normalized=normalized)

    gripper_mesh_scale = voxel_size*100
    #gripper mesh for the pick pose
    gripper_trimesh = trimesh.load('./meshes/hand.dae',force='mesh')
    #gripper_trimesh = as_mesh(gripper_trimesh)
    gripper_trimesh.vertices *= gripper_mesh_scale
    radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
    gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                      color_map='winter')

    #scene.add_geometry(gripper_trimesh, transform=np.array(gripper_pos_mat0))

    #mesh for the place pose
    gripper_trimesh2 = trimesh.load('./meshes/hand.dae', force='mesh')
    gripper_trimesh2.vertices *= gripper_mesh_scale
    gripper_trimesh2.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                      color_map='Reds')
    #scene.add_geometry(gripper_trimesh2, transform=np.array(gripper_pos_mat1))
    frame = trimesh.creation.axis(axis_length=2)
    scene.add_geometry(frame)

    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=1920, viewport_height=1080, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width / r.viewport_height)
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)

        color, depth = r.render(s)
        return color.copy()


def visualise_pick(sample,
                      bounds,
                      voxel_reso,
                      N=180,
                    q_attention: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = True,
                    voxel_size: float = 0.045,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = False,
                    alpha: float = 0.5,
                    delta=False,
                    ):
    # show the
    voxel_grid = sample[0]
    voxel_grid = reindex_output_for_simulator(voxel_grid)
    voxel_grid = voxel_grid[0]
    p0 = sample[1][0]
    if type(p0) is not np.array:
        p0 = np.asarray(p0)
    idx0 = sample[1][1]

    # p1 = sample[2][0]
    # if type(p1) is not np.array:
    #     p1 = np.asarray(p1)
    # idx1 = sample[2][1]
    #print(p0, p1)
    p0 = reindex_predic_for_sim(p0,voxel_size=voxel_reso)
    #p1 = reindex_predic_for_sim(p1,voxel_size=voxel_reso)
    #print(p0,p1)
    #print(voxel_grid.shape)
    continu_p0 = voxel_index_to_point(p0,voxel_reso,bounds)
    #continu_p1 = voxel_index_to_point(p1,voxel_reso,bounds)
    #print('continus_po',continu_p0)
    #print('continus_p1',continu_p1)
    so3_points = _so3_hopf_uniform_grid(N=N)
    #print(idx0,idx1)
    rot0 = so3_points[idx0]
    #rot1 = so3_points[idx1]
    #rot1 is the delta
    # if delta==True:
    #     rot1 = rot1*rot0

    quat0 = rot0.as_quat()
    #quat1 = rot1.as_quat()

    voxel_scale = voxel_size*voxel_reso
    gripper_pos_mat0 = get_gripper_render_pose(voxel_scale,bounds[:3],continu_p0,quat0)
    #gripper_pos_mat1 = get_gripper_render_pose(voxel_scale,bounds[:3],continu_p1,quat1)
    #print(gripper_pos_mat0)

    scene = create_voxel_scene_4_dim(
        voxel_grid, q_attention, None, p0,
        highlight_alpha, voxel_size,
        show_bb, alpha)
    gripper_mesh_scale = voxel_size*100
    #gripper mesh for the pick pose
    gripper_trimesh = trimesh.load('../meshes/hand.dae',force='mesh')
    #gripper_trimesh = as_mesh(gripper_trimesh)
    gripper_trimesh.vertices *= gripper_mesh_scale
    radii = np.linalg.norm(gripper_trimesh.vertices - gripper_trimesh.center_mass, axis=1)
    gripper_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
                                                                      color_map='winter')

    scene.add_geometry(gripper_trimesh, transform=np.array(gripper_pos_mat0))

    # mesh for the place pose
    # gripper_trimesh2 = trimesh.load('../meshes/hand.dae', force='mesh')
    # gripper_trimesh2.vertices *= gripper_mesh_scale
    # gripper_trimesh2.visual.vertex_colors = trimesh.visual.interpolate(radii * gripper_mesh_scale,
    #                                                                   color_map='Reds')
    # scene.add_geometry(gripper_trimesh2, transform=np.array(gripper_pos_mat1))
    frame = trimesh.creation.axis(axis_length=2)
    scene.add_geometry(frame)

    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=1920, viewport_height=1080, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width / r.viewport_height)
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)

        color, depth = r.render(s)
        return color.copy()

def get_gripper_render_pose(voxel_scale, scene_bound_origin, continuous_trans, continuous_quat):
    # finger tip to gripper offset
    offset = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0.1 * voxel_scale],
                       [0, 0, 0, 1]])
    #print(continuous_trans,scene_bound_origin,'=====')
    # scale and translate by origin
    translation = (continuous_trans - scene_bound_origin)* voxel_scale
    mat = np.eye(4, 4)
    mat[:3, :3] = Rotation.from_quat(
        [continuous_quat[0], continuous_quat[1], continuous_quat[2], continuous_quat[3]]).as_matrix()
    offset_mat = np.matmul(mat, offset)
    #offset_mat = np.matmul(mat, offset)
    mat[:3, 3] = translation - offset_mat[:3, 3]
    return mat


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


# from agent_3d.so3_utils import _so3_hopf_uniform_grid, rotation_error, nearest_rotmat


# def quaternion_to_discrete_idx_bilateral(quaternion, N=180):
#
#     rot_matrix = Rotation.from_quat(quaternion)
#     R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
#     rot_matrix2 = (rot_matrix * R).as_matrix()
#     rot_matrix1 = rot_matrix.as_matrix()
#     rot_matrix1 = torch.from_numpy(rot_matrix1).to(torch.float).unsqueeze(dim=0)
#     rot_matrix2 = torch.from_numpy(rot_matrix2).to(torch.float).unsqueeze(dim=0)
#     so3_points = _so3_hopf_uniform_grid(N=N)
#     so3_points = torch.from_numpy(so3_points.as_matrix()).to(torch.float)
#     id1 = nearest_rotmat(rot_matrix1, so3_points)
#     rot_error1 = rotation_error(rot_matrix1, so3_points[id1].unsqueeze(dim=0))
#     id2 = nearest_rotmat(rot_matrix2, so3_points)
#     rot_error2 = rotation_error(rot_matrix2, so3_points[id2].unsqueeze(dim=0))
#     rot_error = torch.cat((rot_error1,rot_error2),dim=0).squeeze(dim=0)
#     rot_idx = torch.cat((id1,id2),dim=0)
#     #print(rot_error)
#     min_idx = torch.argmax(-rot_error)
#     #print(min_idx)
#     min_idx = rot_idx[min_idx].item()
#     # print(min_idx)
#     # print(rot_idx)
#     return min_idx

# def quaternion_to_discrete_idx(quaternion, N=180):
#     rot_matrix = Rotation.from_quat(quaternion)
#     #R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
#     #rot_matrix2 = (rot_matrix * R).as_matrix()
#     rot_matrix1 = rot_matrix.as_matrix()
#     rot_matrix1 = torch.from_numpy(rot_matrix1).to(torch.float).unsqueeze(dim=0)
#     #rot_matrix2 = torch.from_numpy(rot_matrix2).to(torch.float).unsqueeze(dim=0)
#     so3_points = _so3_hopf_uniform_grid(N=N)
#     so3_points = torch.from_numpy(so3_points.as_matrix()).to(torch.float)
#     id1 = nearest_rotmat(rot_matrix1, so3_points)
#     #rot_error1 = rotation_error(rot_matrix1, so3_points[id1].unsqueeze(dim=0))
#     #id2 = nearest_rotmat(rot_matrix2, so3_points)
#     #rot_error2 = rotation_error(rot_matrix2, so3_points[id2].unsqueeze(dim=0))
#     #rot_error = torch.cat((rot_error1,rot_error2),dim=0).squeeze(dim=0)
#     #rot_idx = torch.cat((id1,id2),dim=0)
#     #print(rot_error)
#     #min_idx = torch.argmax(-rot_error)
#     #print(min_idx)
#     #min_idx = rot_idx[min_idx].item()
#     # print(min_idx)
#     # print(rot_idx)
#     return id1.item()

def quaternion_to_discrete_idx_faster(quaternion, N=180, so3_points=None,device=torch.device('cpu')):
    rot_matrix = Rotation.from_quat(quaternion)
    #R = Rotation.from_rotvec(np.pi * np.r_[0.0, 0.0, 1.0])
    #rot_matrix2 = (rot_matrix * R).as_matrix()
    rot_matrix1 = rot_matrix.as_matrix()
    rot_matrix1 = torch.from_numpy(rot_matrix1).to(torch.float).unsqueeze(dim=0).to(device)
    #rot_matrix2 = torch.from_numpy(rot_matrix2).to(torch.float).unsqueeze(dim=0)
    if so3_points is None:
        so3_points = _so3_hopf_uniform_grid(N=N)

    so3_points = torch.from_numpy(so3_points.as_matrix()).to(torch.float).to(device)
    id1 = nearest_rotmat(rot_matrix1, so3_points)
    return id1.item()


# def delta_to_discrete_idx(quaternionp, quaternionq, N=180, so3_points=None):
#
#     rot_matrix1 = Rotation.from_quat(quaternionp)
#     rot_matrix2 = Rotation.from_quat(quaternionq)
#     R = rot_matrix2*rot_matrix1.inv()
#     R = R.as_matrix()
#     #print(R)
#     R = torch.from_numpy(R).to(torch.float).unsqueeze(dim=0)
#     if not so3_points:
#         so3_points = _so3_hopf_uniform_grid(N=N)
#     so3_points = torch.from_numpy(so3_points.as_matrix()).to(torch.float)
#     idx = nearest_rotmat(R, so3_points)
#     #print(so3_points[idx])
#     #rot_error = rotation_error(R, so3_points[idx].unsqueeze(dim=0))
#     return idx.item()

def delta_to_discrete_idx_faster(quaternionp, quaternionq, so3_points=None, N=180, device=torch.device('cpu')):

    rot_matrix1 = Rotation.from_quat(quaternionp)
    rot_matrix2 = Rotation.from_quat(quaternionq)
    R = rot_matrix2*rot_matrix1.inv()
    R = R.as_matrix()
    #print(R)
    R = torch.from_numpy(R).to(torch.float).unsqueeze(dim=0).to(device)
    if so3_points is None:
        so3_points = _so3_hopf_uniform_grid(N=N)

    so3_points = torch.from_numpy(so3_points.as_matrix()).to(torch.float).to(device)
    idx = nearest_rotmat(R, so3_points)
    #print(so3_points[idx])
    #rot_error = rotation_error(R, so3_points[idx].unsqueeze(dim=0))
    return idx.item()


# def quaternion_to_discrete_euler(quaternion, resolution):
#     euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
#     #print('===', euler)
#     assert np.min(euler) >= 0 and np.max(euler) <= 360
#     disc = np.around((euler / resolution)).astype(int)
#     disc[disc == int(360 / resolution)] = 0
#     return disc
#
#
# def discrete_euler_to_quaternion(discrete_euler, resolution):
#     euluer = (discrete_euler * resolution) - 180
#     return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()




def _get_action(
        obs_tp1: Observation,
        # obs_tm1: Observation,
        rlbench_scene_bounds, # metric 3D bounds of the scene
        voxel_sizes,
        # rotation_resolution: int =5,
        # crop_augmentation: bool,
        N: int=180,
        so3_points = None,
        device = torch.device('cpu')
        ):
    #print(obs_tp1.gripper_pose)
    quat = normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    idx = quaternion_to_discrete_idx_faster(quat,N=N,so3_points=so3_points,device=device)
    # disc_rot = quaternion_to_discrete_euler(quat, rotation_resolution)
    # print('disc_rot',disc_rot)
    # attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    voxel_sizes = voxel_sizes.reshape(-1,3)
    for depth, vox_size in enumerate(voxel_sizes): # only single voxelization-level is used in PerAct
        index = point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        #print(bounds[3:])
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    #rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    #print(grip)
    #rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])

    return trans_indicies, idx, quat, np.concatenate([obs_tp1.gripper_pose, np.array([grip])]), attention_coordinates


def reindex_input_for_pytorch(voxel,normalized=False,repeat=False):
    '''
    voxel: B x 10 x XYZ numpy.array
    return: B x 4 x -Z-YX numpy.array
    '''
    rgb = voxel[:,3:6,:,:,:]
    #print('rgb', rgb,rgb.min(), rgb.max())
    if normalized:
        rgb = rgb/2 + 0.5
    #print('rgb', rgb,rgb.min(), rgb.max())
    occ = voxel[:,-1:,:,:,:]
    #print('ocuu',occ.max(), occ.min())
    if repeat:
        occ = occ.repeat(3, 1)
        #print(np.absolute(occ[:,0,:,:,:]-occ[:,1,:,:,:]).sum())
    voxel = np.concatenate((rgb,occ),axis=1)
    #XYZ->ZYX
    voxel = np.transpose(voxel,(0,1,4,3,2))
    #ZYX -> -ZYX
    voxel = np.flip(voxel,axis=2)
    #-ZYX -> -Z-YX
    voxel = np.flip(voxel,axis=3)
    return voxel


def reindex_output_for_simulator_torch(tensor):
    '''
    tensor: BCDHW or BC -Z-YX numpy.array
    return BCXYZ numpy.array
    '''
    # -Z-YX -> Z-Yx
    tensor = torch.flip(tensor,[2])
    #Z-YX -> ZYX
    tensor = torch.flip(tensor,[3])
    #ZYX ->XYZ
    tensor = tensor.permute(0,1,4,3,2)
    return tensor

def reindex_output_for_simulator(tensor):
    '''
    tensor: BCDHW or BC -Z-YX numpy.array
    return BCXYZ numpy.array
    '''
    # -Z-YX -> Z-Yx
    tensor = np.flip(tensor,axis=2)
    #Z-YX -> ZYX
    tensor = np.flip(tensor,axis=3)
    #ZYX ->XYZ
    tensor = np.transpose(tensor,(0,1,4,3,2))
    return tensor

def reindex_label_for_pytorch(trans_idx,voxel_size):
    '''
    trans_idx: xyz numpy.array to -z-yx
    '''

    trans_idx = trans_idx.reshape(3,)
    trans_idx_copy = np.empty(3,dtype=int)
    trans_idx_copy[0] = voxel_size[2] -1 - trans_idx[-1]
    trans_idx_copy[1] = voxel_size[1] -1 - trans_idx[1]
    trans_idx_copy[2] = trans_idx[0]

    return trans_idx_copy

def reindex_predic_for_sim(trans_idx,voxel_size):
    '''
    it's flip instead of -
    trans_idx: -z-yx numpy.array --> xyz
    '''

    trans_idx = trans_idx.reshape(3,)
    trans_idx_copy = np.empty(3,dtype=int)
    trans_idx_copy[0] = trans_idx[-1]
    trans_idx_copy[1] = voxel_size[1] -1 - trans_idx[1]
    trans_idx_copy[2] = voxel_size[2] -1 - trans_idx[0]

    return trans_idx_copy


def flat_pcd_image(obs_dict,CAMERAS):
    #print(obs_dict)
    obs = []
    pcds = []
    masks = []
    others = {'demo':True}
    others.update(obs_dict)
    # print(list(others.keys()))
    for cname in CAMERAS:
        rgb = '%s_rgb' % cname
        pcd = '%s_point_cloud' % cname
        mask = '%s_mask' % cname
        depth = '%s_depth' % cname
        extrinsics = '%s_camera_extrinsics' % cname
        intrinsics = '%s_camera_intrinsics' % cname
        rgb_data = torch.from_numpy(others[rgb]).to(torch.float).unsqueeze(dim=0).unsqueeze(dim=0)
        #print(rgb_data.shape)
        ##
        rgb_data = (rgb_data.float() / 255.0)
        # import matplotlib.pyplot as plt
        # print(rgb_data.shape)
        # plt.imshow(rgb_data[0,0,:,:,:].permute(1,2,0).numpy())
        # plt.show()
        ##
        rgb_data = stack_on_channel(rgb_data)
        #print(rgb_data.shape)
        # rgb_data = _norm_rgb(rgb_data)

        pcd_data = stack_on_channel(torch.from_numpy(others[pcd]).to(torch.float).unsqueeze(dim=0).unsqueeze(dim=0))
        obs.append([rgb_data,pcd_data])
        pcds.append(pcd_data)
        mask_data = torch.from_numpy(others[mask].astype(np.float32)).to(torch.long).unsqueeze(dim=0).unsqueeze(dim=0)
        ###
        # print(mask_data.shape)
        # plt.imshow(mask_data[0,0,0,:,:].numpy())
        # plt.show()
        ###
        mask_data = stack_on_channel(mask_data)
        masks.append(mask_data)

    bs = obs[0][0].shape[0]
    pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcds], 1)

    image_features = [o[0] for o in obs]
    feat_size = image_features[0].shape[1]
    flat_imag_features = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in image_features], 1)
    flat_mask = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, feat_size) for p in masks], 1)
    
    return pcd_flat, flat_imag_features,flat_mask, pcds

def plot_keypoints(demo,episode_keypoints,CAMERAS,camera_name='front'):
    fig,axes = plt.subplots(ncols=len(episode_keypoints),figsize=(15, 5))
    for kp_idx, kp in enumerate(episode_keypoints):
        obs_dict = extract_obs(demo._observations[kp], CAMERAS, t=kp)
        rgb_name = "{}_rgb".format(camera_name)
        rgb = np.transpose(obs_dict[rgb_name], (1, 2, 0))
        axes[kp_idx].imshow(rgb)
        axes[kp_idx].axis('off')
        axes[kp_idx].set_title("%s | step %s | keypoint %s " % (rgb_name,kp, kp_idx))
    plt.show()

def get_action_from_obs(obs):
    #print(obs.gripper_pose)
    trans = obs.gripper_pose[:3]
    quat = normalize_quaternion(obs.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    #gripper_open = obs.gripper_pose[-1]
    ori = Rotation.from_quat(np.asarray([quat[0],quat[1],quat[2],quat[3]]))
    return trans, ori

def preprocess(input_pyt):
    #rgbo, already shift to [0,1]
    channel_mean = [0.5052, 0.5022, 0.4971, 0.0205]
    channel_std = [0.0390, 0.0203, 0.0189, 0.1277]
    channel_mean = torch.as_tensor(channel_mean).reshape(1,4,1,1,1).to(input_pyt.device)
    channel_std = torch.as_tensor(channel_std).reshape(1, 4, 1, 1, 1).to(input_pyt.device)
    input_pyt = (input_pyt-channel_mean)/channel_std
    return input_pyt
