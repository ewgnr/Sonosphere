import torch
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

from threading import Thread, Event
import time
from time import sleep
import datetime

import motion_synthesis

config = {"mapping": None,
          "synthesis": None,
          "sender": None,
          "update_interval": 0.02,
          "view_min": np.array([-100, -100, -100], dtype=np.float32),
          "view_max": np.array([100, 100, 100], dtype=np.float32),
          "view_ele": 90,
          "view_azi": -90,
          "view_dist": 250,
          "view_line_width": 2.0
    }

class PoseCanvasUpdater(QtCore.QObject):
    request_canvas_update = QtCore.pyqtSignal()
    
class MappingCanvas(pg.PlotWidget):
    
    def __init__(self, points2D, callback=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.points2D = points2D
        self.callback = callback
        
        self.scatter = pg.ScatterPlotItem(
            x=points2D[:, 0],
            y=points2D[:, 1],
            pen=pg.mkPen(None),
            brush=pg.mkBrush(100, 100, 255, 120),
            size=10
        )
        
        self.addItem(self.scatter)
        
        # Interactive points and encodings (red points are user-selected)
        self.click_points = []
        self.click_scatter = pg.ScatterPlotItem(size=12, brush=pg.mkBrush(255, 0, 0, 200))
        self.addItem(self.click_scatter)
        self.add_point_interval_ms = 100
        self.remove_point_interval_ms = 10
        
        self.left_move_timer = QtCore.QTimer()
        self.left_move_timer.setInterval(self.add_point_interval_ms)
        self.left_move_timer.timeout.connect(self._add_point_by_timer)
        
        self.right_move_timer = QtCore.QTimer()
        self.right_move_timer.setInterval(self.remove_point_interval_ms)
        self.right_move_timer.timeout.connect(self._remove_point_by_timer)
        self.pending_mouse_pos = None
        
        # Interaction state
        self.left_button_pressed = False
        self.middle_button_pressed = False
        self.right_button_pressed = False
        self.last_pos = None
        
    def add_click_point(self, pos):
        
        self.click_points.append({'pos': (pos[0], pos[1])})
        self.click_scatter.setData(
            [p['pos'][0] for p in self.click_points],
            [p['pos'][1] for p in self.click_points]
        )
        
    def remove_click_point(self, idx):
        
        self.click_points.pop(idx)
        self.click_scatter.setData(
            [p['pos'][0] for p in self.click_points],
            [p['pos'][1] for p in self.click_points]
        )

    def remove_click_points(self):
        self.click_points.clear()
        self.click_scatter.setData([], [])

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_button_pressed = True
            self.pending_mouse_pos = event.pos()
            self.left_move_timer.start()
            
            if self.callback: 
                self.callback("left_press", event.pos())
            
        elif event.button() == Qt.RightButton:
            self.right_button_pressed = True
            self.pending_mouse_pos = event.pos()
            self.right_move_timer.start()
            
            if self.callback: 
                self.callback("right_press", event.pos())
            
        elif event.button() == Qt.MiddleButton:
            super().mousePressEvent(event)
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.left_button_pressed = False
            self.left_move_timer.stop()
            
        elif event.button() == Qt.RightButton:
            self.right_button_pressed = False
            self.right_move_timer.stop()
        else:
            super().mouseReleaseEvent(event)
            
    def mouseMoveEvent(self, event):
        self.pending_mouse_pos = event.pos()
        
        if event.button() == Qt.MiddleButton:
            super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        """Clear points/encodings on 'C' key."""
        if event.key() == QtCore.Qt.Key_C:
            self.remove_click_points()
            #self.synthesis.clearEncodings()
        super().keyPressEvent(event)
            
    def _add_point_by_timer(self):
        if self.left_button_pressed and self.pending_mouse_pos is not None and self.callback:
            self.callback("left_move", self.pending_mouse_pos)

    def _remove_point_by_timer(self):
        if self.right_button_pressed and self.pending_mouse_pos is not None and self.callback:
            self.callback("right_move", self.pending_mouse_pos)
        
class MotionGui(QtWidgets.QWidget):
    
    def __init__(self, config):
        super().__init__()
        
        self.mapping = config["mapping"]
        self.synthesis = config["synthesis"]
        self.sender = config["sender"]
        
        self.edges = self.synthesis.edge_list
        
        self.pose_thread_interval = config["update_interval"]
        
        self.view_min = config["view_min"]
        self.view_max = config["view_max"]
        self.view_ele = config["view_ele"]
        self.view_azi = config["view_azi"]
        self.view_dist = config["view_dist"]
        self.view_line_width = config["view_line_width"]
        
        # mapping canvas
        self.mapping_canvas = MappingCanvas(self.mapping.Z_tsne, callback=self.handle_mapping_mouse)
        
        # pose canvas
        self.pose_canvas = gl.GLViewWidget()
        self.pose_canvas_lines = gl.GLLinePlotItem()
        self.pose_canvas_points = gl.GLScatterPlotItem()
        self.pose_canvas.addItem(self.pose_canvas_lines)
        self.pose_canvas.addItem(self.pose_canvas_points)
        self.pose_canvas.setCameraParams(distance=self.view_dist)
        self.pose_canvas.setCameraParams(azimuth=self.view_azi)
        self.pose_canvas.setCameraParams(elevation=self.view_ele)

        self.q_start_buttom = QtWidgets.QPushButton("start", self)
        self.q_start_buttom.clicked.connect(self.start)  
        
        self.q_stop_buttom = QtWidgets.QPushButton("stop", self)
        self.q_stop_buttom.clicked.connect(self.stop)  

        self.q_clear_buttom = QtWidgets.QPushButton("clear", self)
        self.q_clear_buttom.clicked.connect(self.clear)  
        
        self.q_button_grid = QtWidgets.QGridLayout()
        self.q_button_grid.addWidget(self.q_start_buttom,0,0)
        self.q_button_grid.addWidget(self.q_stop_buttom,0,1)
        self.q_button_grid.addWidget(self.q_clear_buttom,0,2)
        
        self.canvas_grid = QtWidgets.QGridLayout()
        self.canvas_grid.addWidget(self.mapping_canvas,0,0)
        self.canvas_grid.addWidget(self.pose_canvas,0,1)
        self.canvas_grid.setColumnStretch(0, 1)
        self.canvas_grid.setColumnStretch(1, 1)

        self.q_grid = QtWidgets.QGridLayout()
        self.q_grid.addLayout(self.canvas_grid,0,0)
        self.q_grid.addLayout(self.q_button_grid,1,0)
        
        self.q_grid.setRowStretch(0, 0)
        self.q_grid.setRowStretch(1, 0)
        
        self.setLayout(self.q_grid)

        self.setGeometry(50,50,512 * 2,612)
        self.setWindowTitle("Motion Autoencoder")

        # Signals that can be emitted 
        self.poseCanvasUpdater = PoseCanvasUpdater()
        # Update graph whenever the 'request_graph_update' signal is emitted 
        self.poseCanvasUpdater.request_canvas_update.connect(self.update_pose_plot)
        
        # Timer for continuous add (while dragging left button)
        self.add_mapping_timer = QtCore.QTimer()
        self.add_mapping_timer.setInterval(100)  # ms
        self.add_mapping_timer.timeout.connect(self.continuous_add_mapping)
        
    def start(self):
        self.pose_thread_event = Event()
        self.pose_thread = Thread(target = self.update)
        
        self.pose_thread.start()
        
    def stop(self):
        self.pose_thread_event.set()
        self.pose_thread.join()

    def clear(self):
        self.mapping_canvas.remove_click_points()
        self.synthesis.clearEncodings()
        
    def continuous_add_mapping(self):
        """Timer-driven continuous interactive addition while dragging."""
        if self.mapping_canvas.left_button_pressed and self.mapping_canvas.last_mouse_pos is not None:

            x, y = self.mapping_canvas.last_mouse_pos[0], self.mapping_canvas.last_mouse_pos[1]
            self.mapping_canvas.add_click_point([x, y])
            
            # add encoding
            encoding = self.mapping.calc_distance_based_averaged_encoding(np.array([[x, y]]))
            encoding = torch.from_numpy(encoding).unsqueeze(0).to(torch.float32).to(self.synthesis.device)
            self.synthesis.addEncoding(encoding)
        
    def handle_mapping_mouse(self, button, scene_pos):
        
        #print(f"{button} at {scene_pos}")
        
        if button == "left_press":
            
            #print("left_press")
            
            if self.mapping_canvas.sceneBoundingRect().contains(scene_pos):
                
                # convert scene pos to mouse pos
                vb = self.mapping_canvas.plotItem.vb
                mouse_point = vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                # add mouse pos to mapping canvas
                self.mapping_canvas.left_button_pressed = True
                self.mapping_canvas.last_mouse_pos = [x, y]
                
                self.mapping_canvas.add_click_point([x, y])
                
                # add encoding
                encoding = self.mapping.calc_distance_based_averaged_encoding(np.array([[x, y]]))
                encoding = torch.from_numpy(encoding).unsqueeze(0).to(torch.float32).to(self.synthesis.device)
                self.synthesis.addEncoding(encoding)

                #self.add_mapping_timer.start()
            return
        
        elif button == "left_move":
            
            if self.mapping_canvas.sceneBoundingRect().contains(scene_pos) and self.mapping_canvas.left_button_pressed:

                # convert scene pos to mouse pos
                vb = self.mapping_canvas.plotItem.vb
                mouse_point = vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                # add mouse pos to mapping canvas
                self.mapping_canvas.left_button_pressed = True
                self.mapping_canvas.last_mouse_pos = [x, y]
                
                self.mapping_canvas.add_click_point([x, y])
                
                # add encoding
                encoding = self.mapping.calc_distance_based_averaged_encoding(np.array([[x, y]]))
                encoding = torch.from_numpy(encoding).unsqueeze(0).to(torch.float32).to(self.synthesis.device)
                self.synthesis.addEncoding(encoding)
            
            return
            
        elif button == "left_release":
            
            #print("left_release")

            self.mapping_canvas.left_button_pressed = False
            #self.add_mapping_timer.stop()
            
            return
            
        elif button == "right_press":
            
            #print("right_press")
            
            if self.mapping_canvas.sceneBoundingRect().contains(scene_pos):
                
                # convert scene pos to mouse pos
                vb = self.mapping_canvas.plotItem.vb
                mouse_point = vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                # remove mouse pos from mapping canvas and encodings from synthesis
                self.mapping_canvas.right_button_pressed = True
                self.mapping_canvas.last_mouse_pos = [x, y]
                
                to_remove_indices = []
                radius=0.5
                
                for i, p in enumerate(self.mapping_canvas.click_points):
                    dx = p['pos'][0] - x
                    dy = p['pos'][1] - y
                    dist = (dx*dx + dy*dy)**0.5
                    if dist < radius:
                        to_remove_indices.append(i)
                        
                for idx in reversed(to_remove_indices):
                    self.mapping_canvas.remove_click_point(idx)
                    self.synthesis.removeEncoding(idx)

            return

        elif button == "right_move":
            
            if self.mapping_canvas.sceneBoundingRect().contains(scene_pos) and self.mapping_canvas.right_button_pressed:

                # convert scene pos to mouse pos
                vb = self.mapping_canvas.plotItem.vb
                mouse_point = vb.mapSceneToView(scene_pos)
                x, y = mouse_point.x(), mouse_point.y()
                
                # remove mouse pos from mapping canvas and encodings from synthesis
                self.mapping_canvas.right_button_pressed = True
                self.mapping_canvas.last_mouse_pos = [x, y]
                
                to_remove_indices = []
                radius=0.5
                
                for i, p in enumerate(self.mapping_canvas.click_points):
                    dx = p['pos'][0] - x
                    dy = p['pos'][1] - y
                    dist = (dx*dx + dy*dy)**0.5
                    if dist < radius:
                        to_remove_indices.append(i)
                
                for idx in reversed(to_remove_indices):
                    self.mapping_canvas.remove_click_point(idx)
                    self.synthesis.removeEncoding(idx)
                
            return
            
        elif button == "right_release":
            
            #print("right_release")

            self.mapping_canvas.right_button_pressed = False
            #self.add_mapping_timer.stop()
            
            return

                
    def update(self):
        
        while self.pose_thread_event.is_set() == False:

            start_time = time.time()            

            self.update_pred_seq()
            
            self.poseCanvasUpdater.request_canvas_update.emit() 

            self.update_osc()
            
            end_time = time.time()   
            
            #print("update time ", end_time - start_time, " interval ", self.pose_thread_interval)
            
            next_update_interval = max(self.pose_thread_interval - (end_time - start_time), 0.0)
            
            sleep(next_update_interval)

            
    def update_pred_seq(self):
        
        self.synthesis.update()       
        self.synth_pose_wpos = self.synthesis.synth_pose_wpos
        self.synth_pose_wrot = self.synthesis.synth_pose_wrot
        
    def update_osc(self):
        
        # convert from left handed bvh coordinate system to right handed standard coordinate system
        self.synth_pose_wpos_rh = np.copy(self.synth_pose_wpos)

        self.synth_pose_wpos_rh[:, 0] = self.synth_pose_wpos[:, 0] / 100.0
        self.synth_pose_wpos_rh[:, 1] = -self.synth_pose_wpos[:, 2] / 100.0
        self.synth_pose_wpos_rh[:, 2] = self.synth_pose_wpos[:, 1] / 100.0

        self.synth_pose_wrot_rh = np.copy(self.synth_pose_wrot)
        
        self.synth_pose_wrot_rh[:, 1] = self.synth_pose_wrot[:, 1]
        self.synth_pose_wrot_rh[:, 2] = -self.synth_pose_wrot[:, 3]
        self.synth_pose_wrot_rh[:, 3] = self.synth_pose_wrot[:, 2]

        
        self.sender.send("/mocap/0/joint/pos_world", self.synth_pose_wpos_rh)
        self.sender.send("/mocap/0/joint/rot_world", self.synth_pose_wrot_rh)

    def update_pose_plot(self):
        
        pose = self.synth_pose_wpos

        points_data = pose
        lines_data = pose[np.array(self.edges).flatten()]
        
        self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(1.0, 1.0, 1.0, 0.5), width=self.view_line_width)
        #self.pose_canvas_lines.setData(pos=lines_data, mode="lines", color=(0.0, 0.0, 0.0, 1.0), width=self.view_line_width)
        self.pose_canvas_points.setData(pos=pose, color=(1.0, 1.0, 1.0, 0.5))

        #self.pose_canvas.show()
        
        #print(self.pose_canvas.cameraParams())