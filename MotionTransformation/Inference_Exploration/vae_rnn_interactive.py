"""
Motion Autoencoder (RNN Version)
"""

"""
Imports
"""

from matplotlib import pyplot as plt

import motion_model
import motion_mapping
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

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, qfix
from common.quaternion_np import slerp
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""

mocap_file = "../../../Data/Mocap/Daniel_ChineseRoom_Take1_50fps.fbx"
mocap_pos_scale = 1.0
mocap_fps = 50


"""
Model Settings
"""

latent_dim = 32
sequence_length = 64
ae_rnn_layer_count = 2
ae_rnn_layer_size = 512
ae_dense_layer_sizes = [ 512 ]

ae_encoder_weights_file = "../../../Data/Models/MotionTransformation/All/weights/encoder_weights_epoch_600"
ae_decoder_weights_file = "../../../Data/Models/MotionTransformation/All/weights/decoder_weights_epoch_600"

"""
OSC Settings
"""

"""
OSC Receive Settings
"""

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9002

"""
OSC Send Settings
"""

osc_send_ip = "127.0.0.1"
osc_send_port = 9004


"""
Load Mocap Data
"""

bvh_tools = bvh.BVH_Tools()
fbx_tools = fbx.FBX_Tools()
mocap_tools = mocap.Mocap_Tools()

if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
    bvh_data = bvh_tools.load(mocap_file)
    mocap_data = mocap_tools.bvh_to_mocap(bvh_data)
elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
    fbx_data = fbx_tools.load(mocap_file)
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

pose_sequence = mocap_data["motion"]["rot_local"].astype(np.float32)

joint_count = pose_sequence.shape[1]
joint_dim = pose_sequence.shape[2]
pose_dim = joint_count * joint_dim

"""
Load Model
"""

motion_model.config["seq_length"] = sequence_length
motion_model.config["data_dim"] = pose_dim
motion_model.config["latent_dim"] = latent_dim
motion_model.config["rnn_layer_count"] = ae_rnn_layer_count
motion_model.config["rnn_layer_size"] = ae_rnn_layer_size
motion_model.config["dense_layer_sizes"] = ae_dense_layer_sizes
motion_model.config["device"] = device
motion_model.config["weights_path"] = [ae_encoder_weights_file, ae_decoder_weights_file]

encoder, decoder = motion_model.createModels(motion_model.config) 

"""
Setup Motion Mapping
"""

motion_mapping.config["model_encoder"] = encoder
motion_mapping.config["device"] = device
motion_mapping.config["pose_sequence"] = pose_sequence
motion_mapping.config["pose_sequence_length"] = sequence_length
motion_mapping.config["pose_excerpt_offset"] = 20
motion_mapping.config["n_neighbors"] = 4

mapping = motion_mapping.MotionMapping(motion_mapping.config) 

"""
Setup Motion Synthesis
"""

sequence_overlap = sequence_length // 4 * 3

motion_synthesis.config["skeleton"] = mocap_data["skeleton"]
motion_synthesis.config["model_encoder"] = encoder
motion_synthesis.config["model_decoder"] = decoder
motion_synthesis.config["device"] = device
motion_synthesis.config["pose_sequence"] = pose_sequence
motion_synthesis.config["pose_sequence_length"] = sequence_length
motion_synthesis.config["pose_sequence_overlap"] = sequence_overlap // 4

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)

"""
Create OSC Sender
"""

motion_sender.config["ip"] = osc_send_ip
motion_sender.config["port"] = osc_send_port

osc_sender = motion_sender.OscSender(motion_sender.config)


"""
Create Application
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["mapping"] = mapping
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

motion_control.config["motion_seq"] = pose_sequence
motion_control.config["synthesis"] = synthesis
motion_control.config["gui"] = gui
motion_control.config["latent_dim"] = latent_dim
motion_control.config["ip"] = osc_receive_ip
motion_control.config["port"] = osc_receive_port

osc_control = motion_control.MotionControl(motion_control.config)


"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()
