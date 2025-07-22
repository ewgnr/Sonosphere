import numpy as np

from common import utils
from common import bvh_tools as bvh
from common import mocap_tools as mocap
from matplotlib import pyplot as plt
import numpy as np

"""
mocap_data_path = "D:/Data/mocap/ZacharyChant_Music_Improvisation.bvh"
mocap_valid_frame_ranges = [ [ 1500, 29180 ] ]
mocap_fps = 50
mocap_excerpt_length = 200
mocap_excerpt_offset = 50
"""

"""
load mocap data
"""

"""
bvh_tools = bvh.BVH_Tools()
mocap_tools = mocap.Mocap_Tools()

bvh_data = bvh_tools.load(mocap_data_path)
mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
mocap_data["motion"]["pos_world"], mocap_data["motion"]["rot_world"] = mocap_tools.local_to_world(mocap_data["motion"]["rot_local"], mocap_data["motion"]["pos_local"], mocap_data["skeleton"])

joint_count = mocap_data["motion"]["rot_local"].shape[1]
joint_dim = mocap_data["motion"]["rot_local"].shape[2]
offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

joint_weights = np.array([1.0] * joint_count) # todo: get proper joint weights

torso_joint_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
left_arm_joint_indices = [9, 10, 11, 12, 13]
right_arm_joint_indices = [14, 15, 16, 17, 18]
left_leg_indices = [19, 20, 21, 22, 23]
right_leg_indices = [24, 25, 26, 27, 28]

torso_joint_weights = joint_weights[torso_joint_indices]
left_arm_joint_weights = joint_weights[left_arm_joint_indices]
right_arm_joint_weights = joint_weights[right_arm_joint_indices]
left_leg_weights = joint_weights[left_leg_indices]
right_leg_weights = joint_weights[right_leg_indices]
"""

"""
plot orginal data
"""

"""
mocap_data["motion"]["pos_world"].shape

plt.plot(mocap_data["motion"]["pos_world"][:, 0, 0])
"""

"""
smooth data
"""

# code from: https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

def smooth_1d(data_1d, window_length, window_type="hanning"):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is padded with zeros at both ends
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    """
    
    if data_1d.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if data_1d.size < window_length:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_length<3:
        return data_1d

    if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    #print("data_1d s ", data_1d.shape)
    
    pad_left = np.flip(data_1d[:(window_length - 1)//2])
    pad_right = np.flip(data_1d[-(window_length - 1)//2:])
    
    data_padded=np.concatenate((pad_left, data_1d, pad_right))
    
    #print("data_padded s ", data_padded.shape)
    
    #print(len(s))
    if window_type == 'flat': #moving average
        window=np.ones(window_length,'d')
    else:
        window=eval('np.'+window_type+'(window_length)')

    data_smooth=np.convolve(window/window.sum(),data_padded,mode='valid')
    
    #print("data_smooth s ", data_smooth.shape)
    
    return data_smooth

def smooth(data, window_length, window_type="hanning"):
    """
    helper function for multidimensional data
    """
    
    orig_shape = data.shape

    data = np.reshape(data, (orig_shape[0], -1))
    data_dim = data.shape[1]
    data_smooth = [ smooth_1d(data[:, d], window_length, window_type) for d in range(data_dim) ]
    data_smooth = np.stack(data_smooth, axis=1)
    data_smooth = np.reshape(data_smooth, orig_shape)
    
    return data_smooth
    
def derivative(data, time_interval):
    """
    compute first order differences
    """
    
    deriv = (data[1:,...] - data[:-1,...]) / time_interval
    
    deriv = np.concatenate((np.zeros_like(deriv[0:1,...]), deriv))
    
    return deriv

def scalar(data, mode="average"):
    """
    compute scalar value
    mode: "average", "norm", "min", "max"
    """
    
    if not mode in ['average', 'norm', 'min', 'max']:
        raise ValueError("mode is on of 'average', 'norm', 'min', 'max'")
    

    if mode == "average":
        data_scalar = np.mean(data, axis=-1, keepdims=True)
    if mode == "norm":
        data_scalar = np.linalg.norm(data, axis=-1, keepdims=True)
    elif mode == "min":
        data_scalar = np.min(data, axis=-1, keepdims=True)
    elif mode == "max":
        data_scalar = np.max(data, axis=-1, keepdims=True)
        
    return data_scalar

def curvature(accelerations, velocities, scalar_velocities):
    """
    input:
        accelerations: T X J X 3
        velocities: T X J X 3
        scalar velocities: T X J X 1
    output:
        curv: T X 1
    """
    
    accelerations = np.reshape(accelerations, (-1, 3))
    velocities = np.reshape(velocities, (-1, 3))
    scalar_velocities = np.reshape(scalar_velocities, (-1, 1))
    
    cross = np.cross(accelerations, velocities)
    cross_lengths = np.linalg.norm(cross, axis=-1)
    curv = cross_lengths / np.reshape(scalar_velocities, (-1))
    
    curv = np.reshape(curv, (-1, 1))
    
    return curv

def quantity_of_motion(scalar_velocities, joint_weights):
    """
    input: 
        scalar velocities: T X J X 1
        joint_weights: J
    output:
        qom: T X 1
    """
    
    seq_length = scalar_velocities.shape[0]
    
    scalar_velocities = np.squeeze(scalar_velocities, axis=-1)
    
    div = 1.0 / np.sum(joint_weights)
    
    qom = np.zeros(seq_length)
    
    for i in range(0, seq_length):
        qom[i] = np.sum(scalar_velocities[i] * joint_weights) * div
    
    qom = np.reshape(qom, (-1, 1))
    
    return qom

def bounding_box(positions):
    """
    input:
        positions: T x J x 3
    output:
        bbox: T x 2 x 3
    """
    
    seq_length = positions.shape[0]
    
    bbox_min_x = np.min(positions[:, :, 0], axis=1, keepdims=True)
    bbox_min_y = np.min(positions[:, :, 1], axis=1, keepdims=True)
    bbox_min_z = np.min(positions[:, :, 2], axis=1, keepdims=True)
    
    bbox_max_x = np.max(positions[:, :, 0], axis=1, keepdims=True)
    bbox_max_y = np.max(positions[:, :, 1], axis=1, keepdims=True)
    bbox_max_z = np.max(positions[:, :, 2], axis=1, keepdims=True)

    bbox = np.concatenate((bbox_min_x, bbox_min_y, bbox_min_z, bbox_max_x, bbox_max_y, bbox_max_z), axis=1)
    bbox = np.reshape(bbox, (seq_length, 2, 3))

    return bbox

def bounding_sphere(positions):
    """
    input:
        positions: T x J x 3
    output:
        sphere: T x 4
    """

    seq_length = positions.shape[0]
    joint_count = positions.shape[1]

    positions_mean = np.mean(positions, axis=1)
    
    bsphere_center = positions_mean
    
    positions_mean = np.repeat(np.expand_dims(positions_mean, axis=1), joint_count, axis=1)
    
    joint_distances = positions - positions_mean  
    
    joint_distances = np.linalg.norm(joint_distances, axis=2)

    max_distances = np.max(joint_distances, axis=1)
    
    bsphere_radius = np.reshape(max_distances, (seq_length, 1))

    bsphere = np.concatenate((bsphere_center, bsphere_radius), axis=1)

    return bsphere
    
def filter(data, indices):
    """
    filter data
    indices: joint indices to keep
    """
    
    return np.copy(data[:, indices, :])
    
def weight_effort(scalar_velocities, joint_weights, window_length):
    
    seq_length = scalar_velocities.shape[0]
    
    #print("scalar_velocities1 s ", scalar_velocities.shape)
    
    scalar_velocities = np.reshape(scalar_velocities, (seq_length, -1))
    
    #print("scalar_velocities2 s ", scalar_velocities.shape)

    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("joint_weights s ", joint_weights.shape)
    
    efforts_sum = np.sum(scalar_velocities * scalar_velocities * joint_weights, axis=1)
    
    #print("efforts_sum s ", efforts_sum.shape)
    
    pad_left = np.flip(efforts_sum[:(window_length - 1)//2 + 1])
    pad_right = np.flip(efforts_sum[-(window_length - 1)//2:])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
        
    efforts_padded=np.concatenate((pad_left, efforts_sum, pad_right))
    
    #print("efforts_padded s ", efforts_padded.shape)
    
    efforts = np.zeros_like(efforts_sum)
    
    for i in range(0, seq_length - window_length):
        
        excerpt = efforts_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        effort_max = np.max(excerpt)
        
        #print("i ", i, " effort_max s ", effort_max.shape)
        
        efforts[i] = effort_max
    
    efforts = np.expand_dims(efforts, axis=1)
    
    return efforts
    
def time_effort(scalar_accelerations, joint_weights, window_length):
    
    seq_length = scalar_accelerations.shape[0]
    scalar_accelerations = np.reshape(scalar_accelerations, (seq_length, -1))
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("scalar_accelerations s ", scalar_accelerations.shape)

    pad_left = np.flip(scalar_accelerations[:(window_length - 1)//2 + 1, :])
    pad_right = np.flip(scalar_accelerations[-(window_length - 1)//2:, :])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
    
    accelerations_padded=np.concatenate((pad_left, scalar_accelerations, pad_right))
    
    #print("accelerations_padded s ", accelerations_padded.shape)
    
    time = np.zeros_like(scalar_accelerations)
    
    #print("time s ", time.shape)
    
    for i in range(0, seq_length - window_length):
        excerpt = accelerations_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        time[i] = np.sum(excerpt, axis=0) / window_length
        
    efforts = np.sum(time * joint_weights, axis=1)
    
    efforts = np.expand_dims(efforts, axis=1)
    
    #print("efforts s ", efforts.shape)
    
    return efforts

def flow_effort(scalar_jerks, joint_weights, window_length):
    
    seq_length = scalar_jerks.shape[0]
    scalar_jerks = np.reshape(scalar_jerks, (seq_length, -1))
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("scalar_accelerations s ", scalar_accelerations.shape)

    pad_left = np.flip(scalar_jerks[:(window_length - 1)//2 + 1, :])
    pad_right = np.flip(scalar_jerks[-(window_length - 1)//2:, :])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
    
    jerks_padded=np.concatenate((pad_left, scalar_jerks, pad_right))
    
    #print("accelerations_padded s ", accelerations_padded.shape)
    
    time = np.zeros_like(scalar_jerks)
    
    #print("time s ", time.shape)
    
    for i in range(0, seq_length - window_length):
        excerpt = jerks_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        time[i] = np.sum(excerpt, axis=0) / window_length
        
    efforts = np.sum(time * joint_weights, axis=1)
    
    efforts = np.expand_dims(efforts, axis=1)
    
    #print("efforts s ", efforts.shape)
    
    return efforts

def space_effort(scalar_positions, joint_weights, window_length):
    
    seq_length = scalar_positions.shape[0]
    scalar_positions = np.reshape(scalar_positions, (seq_length, -1))
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("scalar_positions s ", scalar_positions.shape)
    
    pad_left = np.flip(scalar_positions[:(window_length - 1)//2 + 1, :])
    pad_right = np.flip(scalar_positions[-(window_length - 1)//2:, :])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
    
    positions_padded=np.concatenate((pad_left, scalar_positions, pad_right))
    
    #print("positions_padded s ", positions_padded.shape)
    
    space = np.zeros_like(scalar_positions)
    
    for i in range(0, seq_length - window_length):
        excerpt = positions_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        start_end_dist = np.absolute(excerpt[0, ...] - excerpt[-1, ...])
        
        #print("start_end_dist s ", start_end_dist.shape)
        
        cum_dist = np.sum(np.absolute(excerpt[1:, ...] - excerpt[:-1, ...]), axis=0)
        
        #print("cum_dist s ", cum_dist.shape)
        
        space[i] = cum_dist / start_end_dist
        
    efforts = np.sum(space * joint_weights, axis=1)
    
    return efforts
    
def space_effort_v2(positions, joint_weights, window_length):
    
    seq_length = positions.shape[0]
    joint_weights = np.reshape(joint_weights, (1, -1))
    
    #print("positions s ", positions.shape)
    
    pad_left = np.flip(positions[:(window_length - 1)//2 + 1, :, :])
    pad_right = np.flip(positions[-(window_length - 1)//2:, :, :])
    
    #print("pad_left s ", pad_left.shape)
    #print("pad_right s ", pad_right.shape)
    
    positions_padded=np.concatenate((pad_left, positions, pad_right))
    
    #print("positions_padded s ", positions_padded.shape)
    
    space = np.zeros(positions.shape[:-1])
    
    #print("space s ", space.shape)
    
    for i in range(0, seq_length - window_length):
        excerpt = positions_padded[i:i+window_length]
        
        #print("i ", i, " excerpt s ", excerpt.shape)
        
        start_end_diff = excerpt[-1, ...] - excerpt[0, ...]
        start_end_length = np.linalg.norm(start_end_diff, axis=-1) 
        start_end_dir = start_end_diff / np.expand_dims(start_end_length, axis=1)
        start_end_dir = np.repeat(np.expand_dims(start_end_dir, axis=0), window_length - 1, axis=0)
        start_end_dir = np.reshape(start_end_dir, (-1, 3))
        
        #print("start_end_diff s ", start_end_diff.shape)
        #print("start_end_length s ", start_end_length.shape)
        #print("start_end_dir s ", start_end_dir.shape)
        
        step_diffs = excerpt[1:, ...] - excerpt[:-1, ...]

        #print("step_diffs s ", step_diffs.shape)
        
        step_lengths = np.linalg.norm(step_diffs, axis=-1) 
        
        #print("step_lengths s ", step_lengths.shape)
        
        step_dirs = step_diffs / np.expand_dims(step_lengths + 0.0001, axis=2)
        
        #print("step_dirs s ", step_dirs.shape)
        
        step_dirs = np.reshape(step_dirs, (-1, 3))
        
        #print("step_dirs s ", step_dirs.shape)
        
        step_dots = np.einsum('ij,ij->i',step_dirs, start_end_dir)
        
        #print("step_dots s ", step_dots.shape)
        
        step_dots = np.reshape(step_dots, (window_length  - 1, -1))
        
        #print("step_dots s ", step_dots.shape)
        
        #step_dots = np.dot(step_dirs, start_end_dir)
        step_deviations = (step_dots * -1.0 + 1.0) * 0.5
        step_deviations *= step_lengths
        
        #print("step_deviations s ", step_deviations.shape)
        
        cum_deviation = np.sum(step_deviations, axis=0)
        
        #print("cum_deviation s ", cum_deviation.shape)
        
        space[i] = cum_deviation
    
    #print("space s ", space.shape)
        
    efforts = np.sum(space * joint_weights, axis=1)
    
    efforts = np.expand_dims(efforts, axis=1)
    
    return efforts

"""
mocap_data["motion"]["pos_world_m"] = mocap_data["motion"]["pos_world"] / 100.0
mocap_data["motion"]["pos_world_smooth"] = smooth(mocap_data["motion"]["pos_world_m"], 25)
mocap_data["motion"]["pos_scalar"] = scalar(mocap_data["motion"]["pos_world_smooth"], "norm")

plt.plot(mocap_data["motion"]["pos_world_m"][:100, 0, 0])
plt.plot(mocap_data["motion"]["pos_world_smooth"][:100, 0, 0])
plt.plot(mocap_data["motion"]["pos_scalar"][:100, 0, 0])

mocap_data["motion"]["vel_world"] = derivative(mocap_data["motion"]["pos_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["vel_world_smooth"] = smooth(mocap_data["motion"]["vel_world"], 25)
mocap_data["motion"]["vel_world_scalar"] = scalar(mocap_data["motion"]["vel_world_smooth"], "norm")

plt.plot(mocap_data["motion"]["vel_world"][:100, 0, 0])
plt.plot(mocap_data["motion"]["vel_world_smooth"][:100, 0, 0])
plt.plot(mocap_data["motion"]["vel_world_scalar"][:100, 0, 0])

mocap_data["motion"]["accel_world"] = derivative(mocap_data["motion"]["vel_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["accel_world_smooth"] = smooth(mocap_data["motion"]["accel_world"], 25)
mocap_data["motion"]["accel_world_scalar"] = scalar(mocap_data["motion"]["accel_world_smooth"], "norm")

plt.plot(mocap_data["motion"]["accel_world"][:100, 0, 0])
plt.plot(mocap_data["motion"]["accel_world_smooth"][:100, 0, 0])
plt.plot(mocap_data["motion"]["accel_world_scalar"][:100, 0, 0])

mocap_data["motion"]["jerk_world"] = derivative(mocap_data["motion"]["accel_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["jerk_world_smooth"] = smooth(mocap_data["motion"]["jerk_world"], 25)
mocap_data["motion"]["jerk_world_scalar"] = scalar(mocap_data["motion"]["jerk_world_smooth"], "norm")

plt.plot(mocap_data["motion"]["jerk_world"][:100, 0, 0])
plt.plot(mocap_data["motion"]["jerk_world_smooth"][:100, 0, 0])
plt.plot(mocap_data["motion"]["jerk_world_scalar"][:100, 0, 0])


mocap_data["motion"]["curvature"] = curvature(mocap_data["motion"]["accel_world_smooth"], mocap_data["motion"]["vel_world_smooth"], mocap_data["motion"]["vel_world_scalar"])
mocap_data["motion"]["qom"] = quantity_of_motion(mocap_data["motion"]["vel_world_scalar"], joint_weights)

plt.plot(mocap_data["motion"]["curvature"])

mocap_data["motion"]["weight_effort"] = weight_effort(mocap_data["motion"]["vel_world_scalar"], joint_weights, 11)
mocap_data["motion"]["time_effort"] = time_effort(mocap_data["motion"]["accel_world_scalar"], joint_weights, 11)
mocap_data["motion"]["flow_effort"] = flow_effort(mocap_data["motion"]["jerk_world_scalar"], joint_weights, 11)
mocap_data["motion"]["space_effort"] = space_effort_v2(mocap_data["motion"]["pos_world_smooth"], joint_weights, 11)

plt.plot(mocap_data["motion"]["weight_effort"])
plt.plot(mocap_data["motion"]["time_effort"])
plt.plot(mocap_data["motion"]["flow_effort"])
plt.plot(mocap_data["motion"]["space_effort"])
"""


