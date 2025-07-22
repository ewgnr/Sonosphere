
import os, sys, time, subprocess

import motion_player 
import motion_sender
import motion_control
import motion_gui

"""
Setup Motion Player
"""

# Example Qualisys Mocap Recording

motion_player.config = { 
    "file_name": "../Data/mocap/ChineseRoom_Daniel_Take1_50fps.fbx",
    "fps": 50,
    "hip_joint_index": 0,
    "remove_hip_offset": True,
    "joint_pos_scale": 1.2
    }

player = motion_player.MotionPlayer(motion_player.config)


"""
OSC Sender
"""

motion_sender.config["ip"] = "127.0.0.1"
motion_sender.config["port"] = 9004 # 23456

osc_sender = motion_sender.OscSender(motion_sender.config)

"""
Setup Motion GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

motion_gui.config["player"] = player
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

motion_control.config["gui"] = gui
motion_control.config["ip"] = "127.0.0.1"
motion_control.config["port"] = 9002

osc_control = motion_control.MotionControl(motion_control.config)


"""
Start Application
"""

osc_control.start()
gui.show()
app.exec_()

osc_control.stop()