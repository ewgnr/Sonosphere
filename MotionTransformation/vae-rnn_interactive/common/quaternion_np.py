"""
quaternion operations and conversions that operate on numpy arrays
representation: w, x, y, z
"""

import torch
import numpy as np
import common.quaternion_torch as tquat 

def mag(q):
    """
    return magnitude of quaternion (tested)
    """

    assert q.shape[-1] == 4
    
    return np.linalg.norm(q, axis=-1, keepdims=True)

def conj(q):
    """
    returns conjugate of quaternion (tested)
    """
    
    assert q.shape[-1] == 4
    
    return np.concatenate((q[..., :1], q[..., -3:] * -1), axis=-1)

def inv(q):
    """
    returns inverse of quaternion (tested)
    """
    
    assert q.shape[-1] == 4
    
    return conj(q) / mag(q)

def normalize(q):
    """
    returns normalised quaternion (tested)
    """
    assert q.shape[-1] == 4
    
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def mul(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return tquat.mul(q, r).numpy()

def rot(q, v):
    """
    Rotate a 3D vector by the rotation stored in the quaternion (tested)
    """

    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return tquat.rot(q, v).numpy()

def mat2quat(R):
    """
    from paper: Ganimator (tested)
    but adapted for numpy instead of torch
    """

    # The rotation matrix must be orthonormal

    w2 = (1 + R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2])
    x2 = (1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])
    y2 = (1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])
    z2 = (1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])

    yz = (R[..., 1, 2] + R[..., 2, 1])
    xz = (R[..., 2, 0] + R[..., 0, 2])
    xy = (R[..., 0, 1] + R[..., 1, 0])

    wx = (R[..., 2, 1] - R[..., 1, 2])
    wy = (R[..., 0, 2] - R[..., 2, 0])
    wz = (R[..., 1, 0] - R[..., 0, 1])

    w = np.empty_like(x2)
    x = np.empty_like(x2)
    y = np.empty_like(x2)
    z = np.empty_like(x2)

    flagA = (R[..., 2, 2] < 0) * (R[..., 0, 0] > R[..., 1, 1])
    flagB = (R[..., 2, 2] < 0) * (R[..., 0, 0] <= R[..., 1, 1])
    flagC = (R[..., 2, 2] >= 0) * (R[..., 0, 0] < -R[..., 1, 1])
    flagD = (R[..., 2, 2] >= 0) * (R[..., 0, 0] >= -R[..., 1, 1])

    x[flagA] = np.sqrt(x2[flagA])
    w[flagA] = wx[flagA] / x[flagA]
    y[flagA] = xy[flagA] / x[flagA]
    z[flagA] = xz[flagA] / x[flagA]

    y[flagB] = np.sqrt(y2[flagB])
    w[flagB] = wy[flagB] / y[flagB]
    x[flagB] = xy[flagB] / y[flagB]
    z[flagB] = yz[flagB] / y[flagB]

    z[flagC] = np.sqrt(z2[flagC])
    w[flagC] = wz[flagC] / z[flagC]
    x[flagC] = xz[flagC] / z[flagC]
    y[flagC] = yz[flagC] / z[flagC]

    w[flagD] = np.sqrt(w2[flagD])
    x[flagD] = wx[flagD] / w[flagD]
    y[flagD] = wy[flagD] / w[flagD]
    z[flagD] = wz[flagD] / w[flagD]

    res = [w, x, y, z]
    res = [np.expand_dims(z, axis=-1) for z in res]

    return np.concatenate(res, axis=-1) / 2

def quat2mat(q):
    """
    from paper: Ganimator
    
    Convert (w, x, y, z) quaternions to 3x3 rotation matrix (tested)
    :param quats: quaternions of shape (..., 4)
    :return:  rotation matrices of shape (..., 3, 3)
    """
    qw = q[..., 0]
    qx = q[..., 1]
    qy = q[..., 2]
    qz = q[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m

def quat2euler(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return tquat.quat2euler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return tquat.quat2euler(q, order, epsilon).numpy()

def slerp(q0, q1, t=0.5, unit=True):
    """
    tested
    :param q0: shape = (*, n)
    :param q1: shape = (*, n)
    :param t: shape = (*)
    :param unit: If q0 and q1 are unit vectors
    :return: res: shape = (*, n)
    """
    eps = 1e-8
    if not unit:
        q0_n = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
        q1_n = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    else:
        q0_n = q0
        q1_n = q1
    omega = np.arccos((q0_n * q1_n).sum(axis=-1).clip(-1, 1))
    dom = np.sin(omega)

    flag = dom < eps

    res = np.empty_like(q0_n)
    t_t = np.expand_dims(t[flag], axis=-1)
    res[flag] = (1 - t_t) * q0_n[flag] + t_t * q1_n[flag]

    flag = ~ flag

    t_t = t[flag]
    d_t = dom[flag]
    va = np.sin((1 - t_t) * omega[flag]) / d_t
    vb = np.sin(t_t * omega[flag]) / d_t
    res[flag] = (np.expand_dims(va, axis=-1) * q0_n[flag] + np.expand_dims(vb, axis=-1) * q1_n[flag])
    return res
