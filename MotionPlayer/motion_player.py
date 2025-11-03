import numpy as np
import pathlib

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap

config = { 
    "file_name": "data/mocap/accumulation_fullbody_take1.bvh"
    }

class MotionPlayer():
    def __init__(self, config):

        self.file_name = config["file_name"]
        
        self.load(self.file_name)
        
    def load(self, file_name):
        
        file_suffix = pathlib.Path(self.file_name).suffix

        if(file_suffix == ".bvh" or file_suffix == ".BVH"):
            self.load_bvh(file_name)
        elif(file_suffix == ".fbx" or file_suffix == ".FBX"):
            self.load_fbx(file_name)

        self.fps = self.mocap_data["frame_rate"]
        
        rot_sequence = self.mocap_data["rot_sequence"]
        offsets = self.mocap_data["skeleton"]["offsets"]
        pos_local = self.mocap_data["motion"]["pos_local"]
        rot_local_euler = self.mocap_data["motion"]["rot_local_euler"]

        # compute local joint rotations and world joint positions
        mocap_tools = mocap.Mocap_Tools()
        
        if(file_suffix == ".bvh" or file_suffix == ".BVH"):
            self.mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(self.mocap_data["motion"]["rot_local_euler"] , self.mocap_data["rot_sequence"])
        elif(file_suffix == ".fbx" or file_suffix == ".FBX"):
            self.mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(self.mocap_data["motion"]["rot_local_euler"] , self.mocap_data["rot_sequence"])
        
        self.mocap_data["motion"]["pos_world"], self.mocap_data["motion"]["rot_world"] = mocap_tools.local_to_world(self.mocap_data["motion"]["rot_local"], self.mocap_data["motion"]["pos_local"], self.mocap_data["skeleton"])
                
        # update start, end, and play position
        self.play_frame = 0
        self.start_play_frame = 0
        self.end_play_frame = self.mocap_data["motion"]["pos_world"].shape[0] - 1

    def load_bvh(self, file_name):
        # load mocap data
        bvh_tools = bvh.BVH_Tools()
        mocap_tools = mocap.Mocap_Tools()

        bvh_data = bvh_tools.load(file_name)
        self.mocap_data = mocap_tools.bvh_to_mocap(bvh_data)

    def load_fbx(self, file_name):
        # load mocap data
        fbx_tools = fbx.FBX_Tools()
        mocap_tools = mocap.Mocap_Tools()
        
        fbx_data = fbx_tools.load(file_name)

        all_mocap_data = mocap_tools.fbx_to_mocap(fbx_data)
        self.mocap_data = all_mocap_data[0] # use only mocap data of first skeleton
        
    def update(self):

        if self.play_frame >= self.end_play_frame:
            self.play_frame = self.start_play_frame
        
        self.play_frame += 1
        
    def get_file_name(self):
        return self.file_name
        
    def get_fps(self):
        return self.fps
    
    def set_fps(self, fps):
        self.fps = fps
        
    def get_play_frame(self):
        return self.play_frame
        
    def set_play_frame(self, frame):

        if frame >= self.end_play_frame:
            frame = self.end_play_frame - 1
            
        if frame <= self.start_play_frame:
             frame = self.start_play_frame + 1
             
        self.play_frame = frame
        
    def get_start_play_frame(self):
        return self.start_play_frame
        
    def set_start_play_frame(self, frame):
        
        if frame >= self.end_play_frame:
            frame = self.end_play_frame - 1

        self.start_play_frame = frame
        
        if self.play_frame < self.start_play_frame:
            self.play_frame = self.start_play_frame
        
        #self.play_frame  = self.start_play_frame
    
    def get_end_play_frame(self):
        return self.end_play_frame
        
    def set_end_play_frame(self, frame):
        
        if frame <= self.start_play_frame:
             frame = self.start_play_frame + 1

        self.end_play_frame = frame
        
        if self.play_frame > self.end_play_frame:
            self.play_frame = self.end_play_frame
        
        #self.play_frame  = self.end_play_frame

    def get_skeleton(self):
        return self.mocap_data["skeleton"]

    def get_pose(self, pose_feature):
        
        return self.mocap_data["motion"][pose_feature][self.play_frame, ...]
