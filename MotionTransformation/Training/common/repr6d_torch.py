"""
calculations and conversions with 6d representation of rotations, operates on torch tensors
"""

import torch
import common.quaternion_torch as tquat

def euler2mat(euler, order=[0, 1, 2], degrees=False):
    """
    From Perplexity
    
    Converts Euler angles to rotation matrices.
    Args:
        euler: (..., 3) tensor, last dim = [angle_x, angle_y, angle_z]
        order: list/tuple of 3 indices (0:x, 1:y, 2:z),
               e.g. [0,1,2]='xyz', [2,1,0]='zyx'
        degrees: bool, whether to interpret input as degrees.
    Returns:
        mat: (..., 3, 3) rotation matrices
    """
    if degrees:
        euler = torch.deg2rad(euler)

    B = euler.shape[:-1]
    mats = []

    # X
    x = euler[..., 0]
    Rx = torch.zeros(B + (3, 3), dtype=euler.dtype, device=euler.device)
    Rx[..., 0, 0] = 1
    Rx[..., 1, 1] = torch.cos(x)
    Rx[..., 1, 2] = -torch.sin(x)
    Rx[..., 2, 1] = torch.sin(x)
    Rx[..., 2, 2] = torch.cos(x)
    mats.append(Rx)

    # Y
    y = euler[..., 1]
    Ry = torch.zeros(B + (3, 3), dtype=euler.dtype, device=euler.device)
    Ry[..., 0, 0] = torch.cos(y)
    Ry[..., 0, 2] = torch.sin(y)
    Ry[..., 1, 1] = 1
    Ry[..., 2, 0] = -torch.sin(y)
    Ry[..., 2, 2] = torch.cos(y)
    mats.append(Ry)

    # Z
    z = euler[..., 2]
    Rz = torch.zeros(B + (3, 3), dtype=euler.dtype, device=euler.device)
    Rz[..., 0, 0] = torch.cos(z)
    Rz[..., 0, 1] = -torch.sin(z)
    Rz[..., 1, 0] = torch.sin(z)
    Rz[..., 1, 1] = torch.cos(z)
    Rz[..., 2, 2] = 1
    mats.append(Rz)

    # Compose in order (extrinsic: left-to-right, intrinsic: right-to-left)
    R = mats[order[0]]
    R = torch.matmul(R, mats[order[1]])
    R = torch.matmul(R, mats[order[2]])
    return R

def mat2euler(mat, order=[0, 1, 2], degrees=False):
    """
    From Perplexity
    
    Converts rotation matrices to Euler angles using specified axis index order.
    Supports 'xyz' only ([0,1,2]); for others, logic must be extended.
    Args:
        mat: (..., 3, 3) tensor
        order: list, e.g. [0,1,2] for 'xyz'
        degrees: bool
    Returns:
        euler: (..., 3) tensor [a0,a1,a2]
    """
    assert order == [0, 1, 2], "Only [0,1,2] ('xyz') order supported in this implementation"
    # 'xyz' convention (right-handed, extrinsic)
    sy = torch.sqrt(mat[..., 0, 0] ** 2 + mat[..., 1, 0] ** 2)
    singular = sy < 1e-6

    x = torch.atan2(mat[..., 2, 1], mat[..., 2, 2])
    y = torch.atan2(-mat[..., 2, 0], sy)
    z = torch.atan2(mat[..., 1, 0], mat[..., 0, 0])

    # singularity handling (gimbal lock)
    x = torch.where(singular, torch.atan2(-mat[..., 1, 2], mat[..., 1, 1]), x)
    z = torch.where(singular, torch.zeros_like(z), z)
    euler = torch.stack([x, y, z], dim=-1)
    if degrees:
        euler = torch.rad2deg(euler)
    return euler

def repr6d2mat(repr):
    """
    from paper: GANimator (tested)
    """
    
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return mat

def mat2repr6d(mat):
    """
    From Perplexity
    Converts rotation matrices to 6D rotation representations.
    Args:
        mat: (..., 3, 3) torch tensor (rotation matrices)
    Returns:
        repr6d: (..., 6) torch tensor
    """
    # Use the first two columns as per Zhou et al. 2019
    return mat[..., :2, :].reshape(*mat.shape[:-2], 6)

def quat2repr6d(quat):
    """
    from paper: GANimator (tested)
    """
    
    mat = tquat.quat2mat(quat)
    res = mat[..., :2, :]
    res = res.reshape(res.shape[:-2] + (6, ))
    return res

def repr6d2quat(repr):
    """
    from paper: GANimator (tested)
    """
    
    x = repr[..., :3]
    y = repr[..., 3:]
    x = x / x.norm(dim=-1, keepdim=True)
    z = torch.cross(x, y)
    z = z / z.norm(dim=-1, keepdim=True)
    y = torch.cross(z, x)
    res = [x, y, z]
    res = [v.unsqueeze(-2) for v in res]
    mat = torch.cat(res, dim=-2)
    return tquat.mat2quat(mat)

def interpolate_6d(input, size):
    """
    from paper: GANimator
    
    :param input: (batch_size, n_channels, length)
    :param size: required output size for temporal axis
    :return:
    """
    batch = input.shape[0]
    length = input.shape[-1]
    input = input.reshape((batch, -1, 6, length))
    input = input.permute(0, 1, 3, 2)     # (batch_size, n_joint, length, 6)
    input_q = repr6d2quat(input)
    idx = torch.tensor(list(range(size)), device=input_q.device, dtype=torch.float) / size * (length - 1)
    idx_l = torch.floor(idx)
    t = idx - idx_l
    idx_l = idx_l.long()
    idx_r = idx_l + 1
    t = t.reshape((1, 1, -1))
    res_q = tquat.slerp(input_q[..., idx_l, :], input_q[..., idx_r, :], t, unit=True)
    res = quat2repr6d(res_q)  # shape = (batch_size, n_joint, t, 6)
    res = res.permute(0, 1, 3, 2)
    res = res.reshape((batch, -1, size))
    return res

    