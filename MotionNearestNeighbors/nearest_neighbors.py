"""
Motion Nearest Neighbors
"""

"""
Imports
"""

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
import motion_analysis as ma

import torch
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer

from matplotlib import pyplot as plt
import numpy as np
import json

"""
Mocap Settings
"""

mocap_file_path = "../../Data/Mocap/"
mocap_files = ["Daniel_ChineseRoom_Take1_50fps.fbx"]
mocap_valid_frame_ranges = [ [ 400, 24600 ] ]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_joint_weight_file = None
mocap_body_weight = 60

"""
Analysis Settings
"""

mocap_excerpt_length = 80 # 80
mocap_excerpt_offset = 40 # 40
mocap_smooth_length = 25 # 25

motion_feature_names = ["bsphere", "space_effort"]
motion_feature_average = True # average motion features in the time domain

"""
Visualisation Settings
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
    
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
        
    mocap_data["motion"]["pos_world"], mocap_data["motion"]["rot_world"] = mocap_tools.local_to_world(mocap_data["motion"]["rot_local"], mocap_data["motion"]["pos_local"], mocap_data["skeleton"])

    all_mocap_data.append(mocap_data)

# retrieve mocap properties

mocap_data = all_mocap_data[0]
joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = mocap_data["motion"]["rot_local"].shape[2]
pose_dim = joint_count * joint_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

# retrieve joint weight percentages

with open(mocap_joint_weight_file) as fh:
    mocap_joint_weight_percentages = json.load(fh)
mocap_joint_weight_percentages = mocap_joint_weight_percentages["jointWeights"]

mocap_joint_weight_percentages = np.array(mocap_joint_weight_percentages)
mocap_joint_weight_percentages_total = np.sum(mocap_joint_weight_percentages)
joint_weights = mocap_joint_weight_percentages * mocap_body_weight / 100.0

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

"""
Gather Mocap Exerpts
"""

mocap_pos_excerpts = []
mocap_rot_excerpts = []

for mocap_index, mocap_data in enumerate(all_mocap_data):
    mocap_rot = mocap_data["motion"]["rot_local"]
    mocap_pos = mocap_data["motion"]["pos_world"] / 100.0 # meters
    
    frame_range_start = mocap_valid_frame_ranges[mocap_index][0]
    frame_range_end = mocap_valid_frame_ranges[mocap_index][1]
    
    for seq_excerpt_start in np.arange(frame_range_start, frame_range_end - mocap_excerpt_length, mocap_excerpt_offset):
        
        #print("valid: start ", frame_range_start, " end ", frame_range_end, " exc: start ", seq_excerpt_start, " end ", (seq_excerpt_start + mocap_excerpt_length) )
        
        mocap_pos_excerpt =  mocap_pos[seq_excerpt_start:seq_excerpt_start + mocap_excerpt_length]
        mocap_pos_excerpts.append(mocap_pos_excerpt)
        
        mocap_rot_excerpt =  mocap_rot[seq_excerpt_start:seq_excerpt_start + mocap_excerpt_length]
        mocap_rot_excerpts.append(mocap_rot_excerpt)
        
mocap_pos_excerpts = np.stack(mocap_pos_excerpts, axis=0)
mocap_rot_excerpts = np.stack(mocap_rot_excerpts, axis=0)

mocap_excerpt_count = mocap_pos_excerpts.shape[0]
    
"""
Compute Motion Features
"""

motion_features = {}

motion_features["rot_local"] = mocap_rot_excerpts
motion_features["pos_world_m"] = mocap_pos_excerpts

# pos world smooth
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    pos_world = motion_features["pos_world_m"][mocap_excerpt_index]
    pos_world_smooth = ma.smooth(pos_world, mocap_smooth_length)
    features.append(pos_world_smooth)
features = np.stack(features, axis=0)
motion_features["pos_world_smooth"] = features

#pos_scalar
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    pos_world_smooth = motion_features["pos_world_smooth"][mocap_excerpt_index]
    pos_scalar = ma.scalar(pos_world_smooth, "norm")
    features.append(pos_scalar)
features = np.stack(features, axis=0)
motion_features["pos_scalar"] = features

#vel_world
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    pos_world_smooth = motion_features["pos_world_smooth"][mocap_excerpt_index]
    vel_world = ma.derivative(pos_world_smooth, 1.0 / mocap_fps)
    features.append(vel_world)
features = np.stack(features, axis=0)
motion_features["vel_world"] = features

#vel_world_smooth
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    vel_world = motion_features["vel_world"][mocap_excerpt_index]
    vel_world_smooth = ma.smooth(vel_world, mocap_smooth_length)
    features.append(vel_world_smooth)
features = np.stack(features, axis=0)
motion_features["vel_world_smooth"] = features
    
#vel_world_scalar
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    vel_world_smooth = motion_features["vel_world_smooth"][mocap_excerpt_index]
    vel_world_scalar = ma.scalar(vel_world_smooth, "norm")
    features.append(vel_world_scalar)
features = np.stack(features, axis=0)
motion_features["vel_world_scalar"] = features

#accel_world
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    vel_world_smooth = motion_features["vel_world_smooth"][mocap_excerpt_index]
    accel_world = ma.derivative(vel_world_smooth, 1.0 / mocap_fps)
    features.append(accel_world)
features = np.stack(features, axis=0)
motion_features["accel_world"] = features

#accel_world_smooth
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    accel_world = motion_features["accel_world"][mocap_excerpt_index]
    accel_world_smooth = ma.smooth(accel_world, mocap_smooth_length)
    features.append(accel_world_smooth)
features = np.stack(features, axis=0)
motion_features["accel_world_smooth"] = features

#accel_world_scalar
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    accel_world_smooth = motion_features["accel_world_smooth"][mocap_excerpt_index]
    accel_world_scalar = ma.scalar(accel_world_smooth, "norm")
    features.append(accel_world_scalar)
features = np.stack(features, axis=0)
motion_features["accel_world_scalar"] = features

#jerk_world
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    accel_world_smooth = motion_features["accel_world_smooth"][mocap_excerpt_index]
    jerk_world = ma.derivative(accel_world_smooth, 1.0 / mocap_fps)
    features.append(jerk_world)
features = np.stack(features, axis=0)
motion_features["jerk_world"] = features

#jerk_world_smooth
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    jerk_world = motion_features["jerk_world"][mocap_excerpt_index]
    jerk_world_smooth = ma.smooth(jerk_world, mocap_smooth_length)
    features.append(jerk_world_smooth)
features = np.stack(features, axis=0)
motion_features["jerk_world_smooth"] = features

#jerk_world_scalar
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    jerk_world_smooth = motion_features["jerk_world_smooth"][mocap_excerpt_index]
    jerk_world_scalar = ma.scalar(jerk_world_smooth, "norm")
    features.append(jerk_world_scalar)
features = np.stack(features, axis=0)
motion_features["jerk_world_scalar"] = features

#qom
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    vel_world_scalar = motion_features["vel_world_scalar"][mocap_excerpt_index]
    qom = ma.quantity_of_motion(vel_world_scalar, joint_weights)
    features.append(qom)
features = np.stack(features, axis=0)
motion_features["qom"] = features

#bbox
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    pos_world = motion_features["pos_world_m"][mocap_excerpt_index]
    bbox = ma.bounding_box(pos_world)
    features.append(bbox)
features = np.stack(features, axis=0)
motion_features["bbox"] = features

#bsphere
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    pos_world = motion_features["pos_world_m"][mocap_excerpt_index]
    bsphere = ma.bounding_sphere(pos_world)
    features.append(bsphere)
features = np.stack(features, axis=0)
motion_features["bsphere"] = features

#weight_effort
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    vel_world_scalar = motion_features["vel_world_scalar"][mocap_excerpt_index]
    weight_effort = ma.weight_effort(vel_world_scalar, joint_weights, mocap_smooth_length)
    features.append(weight_effort)
features = np.stack(features, axis=0)
motion_features["weight_effort"] = features

#space_effort
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    pos_world = motion_features["pos_world_m"][mocap_excerpt_index]
    space_effort = ma.space_effort_v2(pos_world, joint_weights, mocap_smooth_length)
    features.append(space_effort)
features = np.stack(features, axis=0)
motion_features["space_effort"] = features

#time_effort
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    accel_world_scalar = motion_features["accel_world_scalar"][mocap_excerpt_index]
    time_effort = ma.time_effort(accel_world_scalar, joint_weights, mocap_smooth_length)
    features.append(time_effort)
features = np.stack(features, axis=0)
motion_features["time_effort"] = features

#flow_effort
features = []
for mocap_excerpt_index in range(mocap_excerpt_count):
    jerk_world_scalar = motion_features["jerk_world_scalar"][mocap_excerpt_index]
    flow_effort = ma.flow_effort(jerk_world_scalar, joint_weights, mocap_smooth_length)
    features.append(flow_effort)
features = np.stack(features, axis=0)
motion_features["flow_effort"] = features

"""
Normalise Motion Features
"""

for motion_feature_name in list(motion_features.keys()):
    
    #print(motion_feature_name)
    
    motion_feature = motion_features[motion_feature_name]
    
    #print("motion_feature s ", motion_feature.shape)
    
    if motion_feature_average == True:
        motion_feature = np.mean(motion_feature, axis=1, keepdims=True)
        #print("motion_feature avg s ", motion_feature.shape)
    
    shape_orig = motion_feature.shape
    
    motion_feature = motion_feature.reshape(shape_orig[0] * shape_orig[1], -1)
    
    #print("motion_feature 2 s ", motion_feature.shape)
    
    motion_feature_mean = np.mean(motion_feature, axis=0, keepdims=True)
    motion_feature_std = np.std(motion_feature, axis=0, keepdims=True)
    
    #print("motion_feature_mean s ", motion_feature_mean.shape)
    #print("motion_feature_std s ", motion_feature_std.shape)
    
    motion_feature_norm = (motion_feature - motion_feature_mean) / motion_feature_std
    
    #print("motion_feature_norm s ", motion_feature_norm.shape)
    
    motion_feature_norm = motion_feature_norm.reshape(shape_orig[0], -1)
    
    #print("motion_feature_norm 2 s ", motion_feature_norm.shape)
    
    motion_features[motion_feature_name + "_norm"] = motion_feature_norm
    
"""
Prepare Motion Blending
"""

# create rotation envelope for blending motions excerpts into gen_motion
motion_excerpt_overlap = mocap_excerpt_length - mocap_excerpt_offset
rot_envelope_part1 = np.linspace(0.0, 1.0, motion_excerpt_overlap)
rot_envelope_part2 = np.ones([mocap_excerpt_offset])
rot_envelope = np.concatenate([rot_envelope_part1, rot_envelope_part2], axis=0)

"""
rot_envelope_part1 = np.repeat(np.expand_dims(np.linspace(0.0, 1.0, motion_excerpt_overlap), 1), joint_count, axis=1)
rot_envelope_part2 = np.ones([mocap_excerpt_offset, joint_count])
rot_envelope = np.concatenate([rot_envelope_part1, rot_envelope_part2], axis=0)
"""

#plt.plot(rot_envelope)

def motion_blend(target_motion, motion_excerpt, start_pose_index):

    #print("motion_excerpt s ", motion_excerpt.shape, " target_motion s ", target_motion.shape, )

    for epI in range(mocap_excerpt_length):
        tpI = epI + start_pose_index
        slerp_value = rot_envelope[epI]
        
        #print("epI ", epI, " tpI ", tpI, " slerp ", slerp_value)
        
        for jI in range(joint_count):
            
            quat1 = target_motion[tpI, jI]
            quat2 = motion_excerpt[epI , jI]
            quat_blend = slerp(quat1, quat2, slerp_value)
            
            """
            if jI == 1:
                print("jI ", jI, " q1 ", quat1, " q2 ", quat2, " qb ", quat_blend)
            """
            
            target_motion[tpI, jI] = quat_blend

"""
Find Nearest Neighbors
"""

# gather all motion excerpts and concatenate motion features
motion_features_proc = []
for motion_feature_name in motion_feature_names:
    motion_norm_feature_name = motion_feature_name + "_norm"
    motion_feature = motion_features[motion_norm_feature_name]
    
    #print("name ", motion_norm_feature_name, " shape ", motion_feature.shape)
    
    motion_features_proc.append(motion_feature)
    
motion_features_proc = np.concatenate(motion_features_proc, axis=1)
motion_proc = np.copy(mocap_rot_excerpts)

#motion_proc.shape

# prepare empty motion to copy motions corresponding to nearest features into
nn_element_count = motion_features_proc.shape[0]
gen_motion_pose_count = 2 * mocap_excerpt_length + (nn_element_count - 1) * mocap_excerpt_offset
gen_motion = np.array([1.0, 0.0, 0.0, 0.0])
gen_motion = np.reshape(gen_motion, [1, 1, 4])
gen_motion = np.repeat(gen_motion, joint_count, axis=1)
gen_motion = np.repeat(gen_motion, gen_motion_pose_count, axis=0)

# select first motion feature to begin search with
nn_current_index = 0
nn_current_motion = motion_proc[nn_current_index]
nn_current_feature = motion_features_proc[nn_current_index]
nn_current_feature = np.expand_dims(nn_current_feature, 0)

#print("gen_motion s ", gen_motion.shape)

# add first motion excerpt to gen motion
motion_blend(gen_motion, nn_current_motion, 0)

# iterate through all neighbors
pI = mocap_excerpt_offset

while nn_element_count > 0:
    
    print("remaining neighbors ", nn_element_count)
    
    # search nearest element
    nn_distances = np.linalg.norm(motion_features_proc - nn_current_feature, axis=1)
    k = 2
    nn_indices = nn_distances.argsort()[:k]

    # replace current element with nearest element
    nn_previous_index = nn_current_index
    nn_current_index = nn_indices[1]
    nn_current_motion = motion_proc[nn_current_index]
    nn_current_feature = motion_features_proc[nn_current_index]
    nn_current_feature = np.expand_dims(nn_current_feature, 0)
    
    # blend motion corresponding to current element into gen motion
    motion_blend(gen_motion, nn_current_motion, pI)
    
    # remove previous element
    if nn_previous_index == 0:
        motion_proc = np.copy(motion_proc[nn_previous_index + 1:])
        motion_features_proc = np.copy(motion_features_proc[nn_previous_index + 1:])
    elif nn_previous_index == motion_proc.shape[0] - 1:
        motion_proc = np.copy(motion_proc[:nn_previous_index])
        motion_features_proc = np.copy(motion_features_proc[:nn_previous_index])
    else:
        motion_proc = np.copy(np.concatenate([motion_proc[:nn_previous_index], motion_proc[nn_previous_index + 1:]], axis=0))
        motion_features_proc = np.copy(np.concatenate([motion_features_proc[:nn_previous_index], motion_features_proc[nn_previous_index + 1:]], axis=0))

    nn_element_count -= 1
    pI += mocap_excerpt_offset
    
"""
Rendering and Export
"""

print(gen_motion.shape)

poseRenderer = PoseRenderer(edge_list)

def forward_kinematics(rotations, root_positions):
    """
    Perform forward kinematics using the given trajectory and local rotations.
    Arguments (where N = batch size, L = sequence length, J = number of joints):
     -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
     -- root_positions: (N, L, 3) tensor describing the root joint positions.
    """

    assert len(rotations.shape) == 4
    assert rotations.shape[-1] == 4
    
    toffsets = torch.tensor(offsets)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def export_sequence_anim(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, joint_dim))
    
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(torch.float32)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3))).to(torch.float32)
    
    skel_sequence = forward_kinematics(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_sequence_bvh(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]

    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_local"] = pose_sequence
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])

    pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
    
    bvh_tools.write(pred_bvh, file_name)

def export_sequence_fbx(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    
    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_local"] = pose_sequence
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])
    
    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    
    fbx_tools.write(pred_fbx, file_name)

# export gen motion
export_sequence_anim(gen_motion, "results/nearest_neighbors.gif")
export_sequence_bvh(gen_motion,  "results/nearest_neighbors.bvh")
export_sequence_fbx(gen_motion,  "results/nearest_neighbors.fbx")


