"""
Motion Clustering
"""

"""
Imports
"""

import motion_analysis
import motion_model
import motion_synthesis
import motion_sender
import motion_gui
import motion_control


import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict
import networkx as nx
import scipy.linalg as sclinalg
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import DBSCAN

import os, sys, time, subprocess
import numpy as np
import math
import json

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix

"""
Settings
"""

"""
Mocap Settings
"""

mocap_file_path = "../../Data/Mocap"
mocap_files = ["Daniel_ChineseRoom_Take1_50fps.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_joint_weight_file = "configs/joint_weights_qualisys_fbx.json"
mocap_body_weight = 60 # total body weight (in kg)

"""
Cluster Settings
"""

cluster_count = 20
cluster_random_state = 170
sequence_length = 48 # 8
sequence_overlap = 24 # 4

"""
OSC Settings
"""

osc_send_ip = "127.0.0.1"
osc_send_port = 9004

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9002

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

"""
Calculate Joint Weights
"""

# retrieve joint weight percentages

with open(mocap_joint_weight_file) as fh:
    mocap_joint_weight_percentages = json.load(fh)
mocap_joint_weight_percentages = mocap_joint_weight_percentages["jointWeights"]

# calc joint weights

mocap_joint_weight_percentages = np.array(mocap_joint_weight_percentages)
mocap_joint_weight_percentages_total = np.sum(mocap_joint_weight_percentages)
joint_weights = mocap_joint_weight_percentages * mocap_body_weight / 100.0

"""
Calculate Motion Features
"""

mocap_data["motion"]["pos_world_m"] = mocap_data["motion"]["pos_world"] / 100.0

mocap_data["motion"]["pos_world_smooth"] = motion_analysis.smooth(mocap_data["motion"]["pos_world_m"], 25)
mocap_data["motion"]["pos_scalar"] = motion_analysis.scalar(mocap_data["motion"]["pos_world_smooth"], "norm")
mocap_data["motion"]["vel_world"] = motion_analysis.derivative(mocap_data["motion"]["pos_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["vel_world_smooth"] = motion_analysis.smooth(mocap_data["motion"]["vel_world"], 25)
mocap_data["motion"]["vel_world_scalar"] = motion_analysis.scalar(mocap_data["motion"]["vel_world_smooth"], "norm")
mocap_data["motion"]["accel_world"] = motion_analysis.derivative(mocap_data["motion"]["vel_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["accel_world_smooth"] = motion_analysis.smooth(mocap_data["motion"]["accel_world"], 25)
mocap_data["motion"]["accel_world_scalar"] = motion_analysis.scalar(mocap_data["motion"]["accel_world_smooth"], "norm")
mocap_data["motion"]["jerk_world"] = motion_analysis.derivative(mocap_data["motion"]["accel_world_smooth"], 1.0 / 50.0)
mocap_data["motion"]["jerk_world_smooth"] = motion_analysis.smooth(mocap_data["motion"]["jerk_world"], 25)
mocap_data["motion"]["jerk_world_scalar"] = motion_analysis.scalar(mocap_data["motion"]["jerk_world_smooth"], "norm")
mocap_data["motion"]["qom"] = motion_analysis.quantity_of_motion(mocap_data["motion"]["vel_world_scalar"], joint_weights)
mocap_data["motion"]["bbox"] = motion_analysis.bounding_box(mocap_data["motion"]["pos_world_m"])
mocap_data["motion"]["bsphere"] = motion_analysis.bounding_sphere(mocap_data["motion"]["pos_world_m"])
mocap_data["motion"]["weight_effort"] = motion_analysis.weight_effort(mocap_data["motion"]["vel_world_scalar"], joint_weights, 25)
mocap_data["motion"]["space_effort"] = motion_analysis.space_effort_v2(mocap_data["motion"]["pos_world_m"], joint_weights, 25)
mocap_data["motion"]["time_effort"] = motion_analysis.time_effort(mocap_data["motion"]["accel_world_scalar"], joint_weights, 25)
mocap_data["motion"]["flow_effort"] = motion_analysis.flow_effort(mocap_data["motion"]["jerk_world_scalar"], joint_weights, 25)

"""
Create Clustering Model
"""

mocap_features = {"qom": mocap_data["motion"]["qom"],
                  "bsphere": mocap_data["motion"]["bsphere"], 
                  "weight_effort": mocap_data["motion"]["weight_effort"],
                  "space_effort": mocap_data["motion"]["space_effort"],
                  "time_effort": mocap_data["motion"]["time_effort"],
                  "flow_effort": mocap_data["motion"]["flow_effort"]}


motion_model.config["mocap_data"] = mocap_data["motion"]["rot_local"]
motion_model.config["features_data"] = mocap_features
motion_model.config["mocap_window_length"] = sequence_length
motion_model.config["mocap_window_offset"] = sequence_overlap
motion_model.config["cluster_method"] = "kmeans"
motion_model.config["cluster_count"] = cluster_count
motion_model.config["cluster_random_state"] = cluster_random_state

clustering = motion_model.createModel(motion_model.config) 

"""
Setup Motion Synthesis
"""

motion_synthesis.config["skeleton"] = mocap_data["skeleton"]
motion_synthesis.config["model"] = clustering
motion_synthesis.config["seq_window_length"] = sequence_length
motion_synthesis.config["seq_window_overlap"] = sequence_overlap

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)

"""
Create OSC Sender
"""

motion_sender.config["ip"] = osc_send_ip
motion_sender.config["port"] = osc_send_port

osc_sender = motion_sender.OscSender(motion_sender.config)

"""
Create GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable


"""
Create OSC Control
"""


motion_control.config["synthesis"] = synthesis
motion_control.config["model"] = clustering
motion_control.config["gui"] = gui
motion_control.config["ip"] = osc_receive_ip
motion_control.config["port"] = osc_receive_port

osc_control = motion_control.MotionControl(motion_control.config)

"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

#osc_control.stop()
