"""
Motion Analysis
"""

"""
imports
"""

import analysis as ma
import motion_receiver
import motion_sender
import motion_pipeline
import motion_gui

import json
import sys

from matplotlib import pyplot as plt
import numpy as np

"""
Settings
"""

"""
Mocap Settings
"""

#mocap_config_path = "configs/xsens_fbx_config.json"
#mocap_config_path = "configs/zed34_fbx_config.json"
mocap_config_path = "configs/qualisys_fbx_config.json"

"""
OSC Receive Settings
"""

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9007

"""
OSC Send Settings
"""

osc_send_ip = "127.0.0.1"
osc_send_port = 9008

"""
Load Mocap Config
"""

with open(mocap_config_path) as json_data:
    mocap_config = json.load(json_data)
    
input_pos_data = np.zeros((len(mocap_config["joint_names"]), mocap_config["pos_dim"]), dtype=np.float32)

"""
Create OSC Receiver
"""

motion_receiver.config["ip"] = osc_receive_ip
motion_receiver.config["port"] = osc_receive_port
motion_receiver.config["data"] = [ input_pos_data ]
motion_receiver.config["messages"] = ["/mocap/*/joint/pos_world"]

osc_receiver = motion_receiver.MotionReceiver(motion_receiver.config)

"""
Create OSC Sender
"""

motion_sender.config["ip"] = osc_send_ip
motion_sender.config["port"] = osc_send_port

osc_sender = motion_sender.OscSender(motion_sender.config)

"""
Create Analysis Pipeline
"""

pipeline = motion_pipeline.MotionPipeline(osc_receiver, mocap_config)

"""
Create GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["pipeline"] = pipeline
motion_gui.config["sender"] = osc_sender

app = QtWidgets.QApplication(sys.argv)
gui = motion_gui.MotionGui(motion_gui.config)


"""
Start Application
"""

osc_receiver.start()
gui.show()
app.exec_()

#osc_receiver.stop()


