import numpy as np




def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp




# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs



# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


# Based on Algorithm 1 (The geodesic L2-mean on SO(3)) from Richard Hartley et al.,  Rotation averaging, IJCV, 2013
# http://users.cecs.anu.edu.au/~hongdong/rotationaveraging.pdf
def lie_log(rotations):
    Y = (rotations - rotations.swapaxes(-1, -2))/2
    y = np.zeros([*rotations.shape[:-2], 3])
    y[..., 0] = (Y[..., 2, 1] - Y[..., 1, 2])/2
    y[..., 1] = (Y[..., 0, 2] - Y[..., 2, 0])/2
    y[..., 2] = (Y[..., 1, 0] - Y[..., 0, 1])/2
    norm_y = np.linalg.norm(y, axis=-1, keepdims=True)
    rs = np.arcsin(norm_y)*y/(norm_y+np.finfo(np.float32).eps)
    return rs

def lie_exp(angle_axises):
    thetas = np.linalg.norm(angle_axises, axis=-1, keepdims=True)
    vs = angle_axises/(thetas + np.finfo(np.float32).eps)
    vs_cross = np.zeros([*vs.shape[:-1], 3, 3])
    vs_cross[..., 2, 1] = vs[..., 0]
    vs_cross[..., 1, 2] = -vs[..., 0]
    vs_cross[..., 0, 2] = vs[..., 1]
    vs_cross[..., 2, 0] = -vs[..., 1]
    vs_cross[..., 1, 0] = vs[..., 2]
    vs_cross[..., 0, 1] = -vs[..., 2]
    thetas = thetas[..., np.newaxis]
    exp_angle_axises = np.eye(3) + np.sin(thetas)*vs_cross + (1 - np.cos(thetas)) * (vs_cross @ vs_cross)
    return exp_angle_axises

def average_rotations(list_rotations):
    nb_it_max = 5
    number_rotations = list_rotations.shape[0]
    if number_rotations == 1:
        return list_rotations[0]

    R = list_rotations[0]
    for i in range(nb_it_max):
        rs = lie_log(R.swapaxes(-1, -2) @ list_rotations)
        r_mean = rs.mean(axis=0)
        R = R @ lie_exp(r_mean)
    return R


def average_poses(list_poses):
    avg_poses = np.zeros(list_poses.shape[1:])
    avg_poses[..., 0:3, 3] = list_poses[..., 0:3, 3].mean(axis=0)
    avg_poses[..., 0:3, 0:3] = average_rotations(list_poses[..., 0:3, 0:3])
    avg_poses[..., 3, 3] = 1
    return avg_poses


def local_poses_to_odometry(local_poses):
    num_frames = local_poses.shape[0]
    num_poses = local_poses.shape[1]

    extend_poses = np.concatenate([local_poses[:, :int(num_poses/2), ...],
                                   np.eye(4)[np.newaxis, np.newaxis, :].repeat(num_frames, axis=0),
                                   local_poses[:, int(num_poses/2):, ...]], axis=1)
    for i in range(num_poses, 0, -1):
        extend_poses[:, i, ...] = np.linalg.inv(extend_poses[:, i-1, ...]) @ extend_poses[:, i, ...]
    #extend_poses[:, 0, ...] = np.eye(4)[np.newaxis, np.newaxis, :].repeat(num_frames, axis=0)

    fused_poses = np.zeros([num_frames + num_poses, 4, 4])
    fused_poses[0, ...] = np.eye(4)
    for i in range(1, num_frames + num_poses):
        fused_poses[i, ...] = average_poses(np.array([extend_poses[i-j, j, ...]
                                                      for j in range(1, num_poses+1) if 0 <= i-j < num_frames]))

    odometry = []
    odometry.append(fused_poses[0, ...])
    for i in range(1, num_frames + num_poses):
        odometry.append(np.dot(odometry[i-1], fused_poses[i, ...]))
    return np.array(odometry)


# from https://github.com/ClementPinard/SfmLearner-Pytorch
def compute_pose_error(gt_poses, pred_poses):
    RE = 0
    snippet_length = gt_poses.shape[0]
    scale_factor = np.sum(gt_poses[:, :3, -1] * pred_poses[:, :3, -1])/np.sum(pred_poses[:, :3, -1] ** 2)
    ATE = np.linalg.norm((gt_poses[:, :3, -1] - scale_factor * pred_poses[:, :3, -1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt_poses, pred_poses):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:3, :3] @ np.linalg.inv(pred_pose[:3, :3])
        s = np.linalg.norm([R[0, 1]-R[1, 0],
                            R[1, 2]-R[2, 1],
                            R[0, 2]-R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return ATE/snippet_length, RE/snippet_length
