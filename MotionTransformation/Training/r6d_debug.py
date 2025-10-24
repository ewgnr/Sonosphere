import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np
import json

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer
import common.repr6d_torch as repr6d_torch

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

"""
Compute Unit
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings

important: the skeleton needs to be identical in all mocap recordings
"""


# Example: XSens Mocap Recording
mocap_file_path = "data/mocap"
mocap_files = ["Muriel_Embodied_Machine_variation.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_loss_weights_file = None

"""
Visualization Settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0


"""
Load Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

all_mocap_data = []

for mocap_file in mocap_files:
    
    print("process file ", mocap_file)
    
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        bvh_data = bvh_tools.load(mocap_file_path + "/" + mocap_file)
        mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        fbx_data = fbx_tools.load(mocap_file_path + "/" + mocap_file)
        mocap_data = mocap_tools.fbx_to_mocap(fbx_data)[0] # first skeleton only
        
    mocap_data["skeleton"]["offsets"] *= mocap_pos_scale
    mocap_data["motion"]["pos_local"] *= mocap_pos_scale
    
    # set x and z offset of root joint to zero
    mocap_data["skeleton"]["offsets"][0, 0] = 0.0 
    mocap_data["skeleton"]["offsets"][0, 2] = 0.0 
    
    rot_sequence = mocap_data["rot_sequence"]
    
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        mocap_data["motion"]["rot_6d"] = repr6d_torch.quat2repr6d( torch.from_numpy(mocap_data["motion"]["rot_local"]) ).numpy()

    all_mocap_data.append(mocap_data)

# retrieve mocap properties

mocap_data = all_mocap_data[0]
joint_count = mocap_data["motion"]["rot_6d"].shape[1]
joint_dim = mocap_data["motion"]["rot_6d"].shape[2]
pose_dim = joint_count * joint_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

# set joint loss weigths 

if mocap_loss_weights_file is not None:
    with open(mocap_loss_weights_file) as f:
        joint_loss_weights = json.load(f)
        joint_loss_weights = joint_loss_weights["joint_loss_weights"]
else:
    joint_loss_weights = [1.0]
    joint_loss_weights *= joint_count
    
    
# inference and rendering 

poseRenderer = PoseRenderer(edge_list)

def safe_slerp(q1, q2, t):
    # q1, q2: (..., 4)
    # t: interpolation parameter
    dot = (q1 * q2).sum(axis=-1, keepdims=True)
    q2_fixed = np.where(dot < 0, -q2, q2)
    return slerp(q1, q2_fixed, t)

def consistent_quaternion_seq(seq1, seq2):
    """
    Enforces sign consistency for two quaternion sequences for blending.

    Both seq1 and seq2 should be np.ndarray of shape (L, J, 4).

    The function makes sure:
    - Each joint's quaternions are consistent over frames in seq1.
    - Each joint's quaternions are consistent over frames in seq2.
    - The last frame of seq1 for each joint is sign-consistent with the first frame of seq2.

    Returns:
        seq1_out, seq2_out : np.ndarray, with sign-consistent quaternions.
    """
    seq1_out = np.copy(seq1)
    seq2_out = np.copy(seq2)
    L1, J, _ = seq1_out.shape
    L2 = seq2_out.shape[0]

    # Enforce consistency within seq1
    for j in range(J):
        for i in range(1, L1):
            if np.dot(seq1_out[i, j], seq1_out[i-1, j]) < 0:
                seq1_out[i, j] *= -1

    # Enforce consistency within seq2
    for j in range(J):
        for i in range(1, L2):
            if np.dot(seq2_out[i, j], seq2_out[i-1, j]) < 0:
                seq2_out[i, j] *= -1

    # Enforce consistency between seq1's last frame and seq2's first frame (for each joint)
    for j in range(J):
        if np.dot(seq1_out[-1, j], seq2_out[0, j]) < 0:
            seq2_out[:, j, :] *= -1

    return seq1_out, seq2_out

def forward_kinematics_r6d(rotations_r6d, root_positions):
    """
    Args:
        rotations_r6d: (N, L, J, 6) tensor of 6D joint rotations
        root_positions: (N, L, 3) tensor
        offsets: (J, 3) numpy array (or tensor)
        parents: list of parent indices for each joint
        children: list of children indices for each joint
    Returns:
        positions_world: (N, L, J, 3) tensor of world joint positions
    """
    
    assert len(rotations_r6d.shape) == 4
    assert rotations_r6d.shape[-1] == 6
    
    rotations_quat = repr6d_torch.repr6d2quat(rotations_r6d)
    
    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations_quat.shape[0], rotations_quat.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations_quat[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations_quat[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def forward_kinematics_quat(rotations_quat, root_positions):
    """
    Args:
        rotations_quat: (N, L, J, 4) tensor of quaternion joint rotations
        root_positions: (N, L, 3) tensor
        offsets: (J, 3) numpy array (or tensor)
        parents: list of parent indices for each joint
        children: list of children indices for each joint
    Returns:
        positions_world: (N, L, J, 3) tensor of world joint positions
    """
    
    assert len(rotations_quat.shape) == 4
    assert rotations_quat.shape[-1] == 4

    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations_quat.shape[0], rotations_quat.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations_quat[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations_quat[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def export_sequence_anim_r6d(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, 6))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics_r6d(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)
  
def export_sequence_anim_quat(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, 4))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics_quat(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)  

orig_sequence_r6d = all_mocap_data[0]["motion"]["rot_6d"].astype(np.float32)
orig_sequence_quat = all_mocap_data[0]["motion"]["rot_local"].astype(np.float32)

seq_start = 1000
seq_length = 2000

export_sequence_anim_r6d(orig_sequence_r6d[seq_start:seq_start+seq_length], "orig_r6d.gif".format(seq_start, seq_length))
export_sequence_anim_quat(orig_sequence_quat[seq_start:seq_start+seq_length], "orig_quat.gif".format(seq_start, seq_length))
