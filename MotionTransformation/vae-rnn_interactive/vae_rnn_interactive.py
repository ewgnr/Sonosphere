
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

mocap_file_path = "../../../data/mocap/"
mocap_files = ["DeanNoel_Ocean_Take1_50fps.fbx",
               "DeanNoel_Ocean_Take2_50fps.fbx",
               "MagdaPascalAmbience_Take1_50fps.fbx",
               "MagdaPascalAmbience_Take2_50fps.fbx",
               "SelinHabibaNic_Shopping_Take1_50fps.fbx",
               "SelinHabibaNic_Shopping_Take2_50fps.fbx"]
mocap_valid_frame_ranges = [ [ [ 260, 20786 ] ],
                            [ [ 515, 18942 ] ],
                            [ [ 541, 23400 ] ],
                            [ [ 498, 20012 ] ],
                            [ [ 665, 10898 ] ],
                            [ [ 572, 15092 ] ]]
mocap_pos_scale = 1.0
mocap_fps = 50

mocap_seq_window_length = 64
mocap_seq_window_overlap = 48


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

motion_model.config["seq_length"] = mocap_seq_window_length
motion_model.config["data_dim"] = pose_dim
motion_model.config["latent_dim"] = 32
motion_model.config["rnn_layer_count"] = 2
motion_model.config["rnn_layer_size"] = 512
motion_model.config["dense_layer_sizes"] = [512]
motion_model.config["device"] = device

motion_model.config["weights_path"] = ["../vae-rnn/results_Marcella_JazzFMSynth/weights/encoder_weights_epoch_600", "../vae-rnn/results_Marcella_JazzFMSynth/weights/decoder_weights_epoch_600"]

encoder, decoder = motion_model.createModels(motion_model.config) 

"""
Setup Motion Synthesis
"""

motion_synthesis.config = {
    "skeleton": all_mocap_data[0]["skeleton"],
    "model_encoder": encoder,
    "model_decoder": decoder,
    "device": device,
    "seq_window_length": mocap_seq_window_length,
    "seq_window_overlap": mocap_seq_window_overlap,
    "orig_sequences": all_pose_sequences,
    "orig_seq1_index": 0,
    "orig_seq2_index": 1
    }

synthesis = motion_synthesis.MotionSynthesis(motion_synthesis.config)

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
motion_control.config["latent_dim"] = 32
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
