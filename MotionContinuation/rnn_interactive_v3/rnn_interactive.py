"""
Simple sequence continuation model based on an LSTM neural network
This version works with motion capture data that stores joint rotations and recorded in BVH or FBX format
"""

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

import os, sys, time, subprocess
import numpy as np
import math
import pickle
from time import sleep

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_file_path = "../../Data/mocap"
mocap_files = [
    "Dream_ChiaraAndreasMaximilian_Take1.50fps.fbx",
    "HarmonicOrbit_SelinaAlinaSergejNico_Take1_50fps.fbx",
    "InterfaceDance_MagdaPascal_Take2_50fps.fbx",
    "Shopping_SelinHabibaNicole_Take2_50fps.fbx",
    "MHOM_Ephraim_Take2_50fps.fbx",
    "ChineseRoom_Daniel_Take1_50fps.fbx"
    ]
mocap_valid_frame_ranges = [ 
    [ [ 629, 18900 ] ],
    [ [ 450, 23500 ] ],
    [ [ 750, 19200 ] ],
    [ [ 600, 15000 ] ], 
    [ [ 460, 19255 ] ], 
    [ [ 500, 23742 ] ] 
    ]
mocap_pos_scale = 1.0
mocap_fps = 50

"""
Weight Settings
"""

weight_file_path = "../rnn"
weight_files = [
    "results_Dream_ChiaraAndreasMaximilian_Take1/weights/rnn_weights_epoch_400",
    "results_HarmonicOrbit_SelinaAlinaSergejNico_Take1/weights/rnn_weights_epoch_400",
    "results_InterfaceDance_MagdaPascal_Take2/weights/rnn_weights_epoch_400",
    "results_Shopping_SelinHabibaNicole_Take2/weights/rnn_weights_epoch_400",
    "results_MHOM_Ephraim_Take2/weights/rnn_weights_epoch_400",
    "results_ChineseRoom_Daniel_Take1/weights/rnn_weights_epoch_400"
    ]


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

    all_mocap_data.append(mocap_data)

all_pose_sequences = []

for mocap_data in all_mocap_data:
    
    pose_sequence = mocap_data["motion"]["rot_local"].astype(np.float32)
    all_pose_sequences.append(pose_sequence)

joint_count = all_pose_sequences[0].shape[1]
joint_dim = all_pose_sequences[0].shape[2]
pose_dim = joint_count * joint_dim


"""
Load Model
"""

motion_model.config["input_length"] = 64
motion_model.config["data_dim"] = pose_dim
motion_model.config["node_dim"] = 512
motion_model.config["layer_count"] = 2
motion_model.config["device"] = device
motion_model.config["weights_path"] = weight_file_path + "/" + weight_files[0]

model = motion_model.createModel(motion_model.config) 


"""
Setup Motion Synthesis
"""

synthesis_config  = motion_synthesis.config
synthesis_config["skeleton"] = all_mocap_data[0]["skeleton"]
synthesis_config["model"] = model
synthesis_config["seq_length"] = motion_model.config["input_length"]
synthesis_config["orig_sequences"] = all_pose_sequences
synthesis_config["orig_seq_index"] = 0
synthesis_config["device"] = motion_model.config["device"] 
synthesis_config["weight_files"] = [ weight_file_path + "/" + weight_file for weight_file in weight_files ]

synthesis = motion_synthesis.MotionSynthesis(synthesis_config)


"""
OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9004

osc_sender = motion_sender.OscSender(motion_sender.config)


"""
GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["synthesis"] = synthesis
motion_gui.config["sender"] = osc_sender
motion_gui.config["update_interval"] = 1.0 / mocap_fps

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

"""
OSC Control
"""

motion_control.config["motion_seq"] = pose_sequence
motion_control.config["synthesis"] = synthesis
motion_control.config["gui"] = gui
motion_control.config["ip"] = "0.0.0.0"
motion_control.config["port"] = 9002

osc_control = motion_control.MotionControl(motion_control.config)

"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()


osc_control.stop()
