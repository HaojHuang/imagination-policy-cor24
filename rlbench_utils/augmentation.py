import numpy as np
import torch
from rlbench_utils.arm_utils_new import point_to_voxel_index, normalize_quaternion, quaternion_to_discrete_idx_faster,delta_to_discrete_idx_faster
from pytorch3d import transforms as torch3d_tf

# there is a difference between voxel_index_to_point
def rand_dist(size, min=-1.0, max=1.0):
    dist = (max-min) * torch.rand(size) + min
    # z trans should contain the table
    dist[:,-1] = torch.abs(dist[:,-1])
    #print(dist,'shift proportion')
    return dist

def rand_discrete(size, min=0, max=1):
    if min == max:
        return torch.zeros(size)
    return torch.randint(min, max+1, size)


def perturb_se3(pcd,
                trans_shift_4x4,
                rot_shift_4x4,
                action_gripper_4x4,
                bounds):
    """ Perturb point clouds with given transformation.
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: peturbed point clouds
    """
    # baatch bounds if necessary
    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        num_points = p_shape[-1] * p_shape[-2]

        #action_trans_3x1 = action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        trans_shift_3x1 = trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)

        # flatten point cloud
        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

        # shift points to have action_gripper pose as the origin
        p_flat_4x1_action_origin[:, :3, :] = p_flat

        # apply rotation
        perturbed_p_flat_4x1_action_origin = torch.bmm(rot_shift_4x4, p_flat_4x1_action_origin)

        # apply bounded translations
        # bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        # bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        # bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        #action_then_trans_3x1 = origin + trans_shift_3x1
        action_then_trans_3x1 = trans_shift_3x1
        # action_then_trans_3x1_x = torch.clamp(action_then_trans_3x1[:, 0],
        #                                       min=bounds_x_min, max=bounds_x_max)
        # action_then_trans_3x1_y = torch.clamp(action_then_trans_3x1[:, 1],
        #                                       min=bounds_y_min, max=bounds_y_max)
        # action_then_trans_3x1_z = torch.clamp(action_then_trans_3x1[:, 2],
        #                                       min=bounds_z_min, max=bounds_z_max)
        # action_then_trans_3x1 = torch.stack([action_then_trans_3x1_x,
        #                                      action_then_trans_3x1_y,
        #                                      action_then_trans_3x1_z], dim=1)

        # shift back the origin
        # print(perturbed_p_flat_4x1_action_origin.shape)
        # print(action_trans_3x1.shape)
        # print('====')
        perturbed_p_flat_3x1 = perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1
        #print(action_then_trans_3x1,'===')
        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)
    return perturbed_pcd


def apply_se3_augmentation(pcd,
                           action_gripper_pose,
                           action_gripper_pose_place,
                           bounds,
                           layer,
                           trans_aug_range,
                           rot_aug_range,
                           rot_aug_resolution,
                           voxel_size,
                           device,
                           N=180,
                           fineN=6000):
    """ Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    #print(pcd[0].shape)
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose PICK
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat((action_gripper_pose[:, 6].unsqueeze(1),
                                          action_gripper_pose[:, 3:6]), dim=1)

    action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    # 4x4 matrix of keyframe action gripper pose PlACE
    action_gripper_trans_place = action_gripper_pose_place[:, :3]
    action_gripper_quat_wxyz_place = torch.cat((action_gripper_pose_place[:, 6].unsqueeze(1),
                                                action_gripper_pose_place[:, 3:6]), dim=1)

    action_gripper_rot_place = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz_place)
    action_gripper_4x4_place = identity_4x4.detach().clone()
    action_gripper_4x4_place[:, :3, :3] = action_gripper_rot_place
    action_gripper_4x4_place[:, 0:3, 3] = action_gripper_trans_place

    action_trans = torch.zeros(len(action_gripper_pose), 3)
    perturbed_trans_pick = torch.full_like(action_trans, -1.)
    perturbed_trans_place = torch.full_like(action_trans, -1.)

    #perturbed_rot_grip = torch.full_like(action_rot_grip, -1.)
    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans_pick < 2) or torch.any(perturbed_trans_place< 2) or \
            torch.any(perturbed_trans_pick.reshape(-1) > torch.as_tensor(voxel_size)-2) or torch.any(perturbed_trans_place.reshape(-1) > torch.as_tensor(voxel_size)-2) :
        # print('aug voxel size',torch.as_tensor(voxel_size))
        # print('aug pick',perturbed_trans_pick)
        # print('aug place', perturbed_trans_place)
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            raise Exception('Failing to perturb action and keep it within bounds.')

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device=device)
        trans_shift = trans_range * rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = rand_discrete((bs, 1),
                                   min=-roll_aug_steps,
                                   max=roll_aug_steps) * np.deg2rad(rot_aug_resolution)
        pitch = rand_discrete((bs, 1),
                                    min=-pitch_aug_steps,
                                    max=pitch_aug_steps) * np.deg2rad(rot_aug_resolution)
        yaw = rand_discrete((bs, 1),
                                  min=-yaw_aug_steps,
                                  max=yaw_aug_steps) * np.deg2rad(rot_aug_resolution)

        # print('roll',roll)
        # print('pitch',pitch)
        # print('yaw',yaw)
        # print('trans shift',trans_shift)

        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(torch.cat((roll, pitch, yaw), dim=1), "XYZ")
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rot_shift_3x3_ = torch3d_tf.euler_angles_to_matrix(torch.cat((-yaw, pitch, roll), dim=1), "XYZ")
        # rot_shift_4x4_ = identity_4x4.detach().clone()
        # rot_shift_4x4_[:, :3, :3] = rot_shift_3x3_

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(rot_shift_4x4, action_gripper_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        perturbed_action_gripper_4x4_place = torch.bmm(rot_shift_4x4,action_gripper_4x4_place)
        perturbed_action_gripper_4x4_place[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(perturbed_action_gripper_4x4[:, :3, :3])
        perturbed_action_quat_xyzw = torch.cat([perturbed_action_quat_wxyz[:, 1:],
                                                perturbed_action_quat_wxyz[:, 0].unsqueeze(1)],
                                               dim=1).cpu().numpy()

        perturbed_action_trans_place = perturbed_action_gripper_4x4_place[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz_place = torch3d_tf.matrix_to_quaternion(perturbed_action_gripper_4x4_place[:, :3, :3])
        perturbed_action_quat_xyzw_place = torch.cat([perturbed_action_quat_wxyz_place[:, 1:],
                                                      perturbed_action_quat_wxyz_place[:, 0].unsqueeze(1)],
                                                     dim=1).cpu().numpy()



        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies,trans_indicies_place = [],[]
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()
            perturbed_action_trans_p = perturbed_action_trans[b]
            perturbed_action_trans_q = perturbed_action_trans_place[b]

            trans_idx = point_to_voxel_index(perturbed_action_trans_p, voxel_size, bounds_np)
            trans_indicies.append(trans_idx.tolist())
            trans_idx_place = point_to_voxel_index(perturbed_action_trans_q, voxel_size, bounds_np)
            trans_indicies_place.append(trans_idx_place.tolist())

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans_pick = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        perturbed_trans_place = torch.from_numpy(np.array(trans_indicies_place)).to(device=device)


    # only consider the orientation here
    rot_pick_indicies, rot_place_indicies = [], []
    for b in range(bs):
        # #quat = perturbed_action_quat_xyzw[b]
        quat = normalize_quaternion(perturbed_action_quat_xyzw[b])
        if quat[-1] < 0:
            quat = -quat
        # print('quat', quat)
        #
        quat_place = normalize_quaternion(perturbed_action_quat_xyzw_place[b])
        if quat_place[-1] < 0:
            quat_place = -quat_place

        #delta_idx = delta_to_discrete_idx(quat, quat_place)
        #disc_rot = quaternion_to_discrete_idx(quat)
        #rot_pick_indicies.append(disc_rot)
        #rot_place_indicies.append(delta_idx)

    #rot_pick_indicies = torch.from_numpy(np.array(rot_pick_indicies)).to(device=device)
    #rot_place_indicies = torch.from_numpy(np.array(rot_place_indicies)).to(device=device)
    #print('perturbed_trans_pick',trans_indicies)
    # action_pick_indices = rot_pick_indicies
    # action_place_indices = rot_place_indicies

    action_trans = perturbed_trans_pick
    action_trans_place = perturbed_trans_place

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)
    return action_trans.numpy(), action_trans_place.numpy(), quat, quat_place, pcd, perturbed_action_trans_p, perturbed_action_trans_q, trans_shift_4x4,rot_shift_4x4