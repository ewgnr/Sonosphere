import torch
from torch import nn
import numpy as np

from common.quaternion import qmul, qrot, qnormalize_np, slerp
from common.quaternion_torch import qfix

config = {"skeleton": None,
          "model": None,
          "seq_window_length": 20,
          "seq_window_overlap": 10
          }

class MotionSynthesis():
    
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.model = config["model"]
        self.seq_window_length = config["seq_window_length"]
        self.seq_window_overlap = config["seq_window_overlap"]
        
        self.seq_window_offset = self.seq_window_length - self.seq_window_overlap
        
        self.cluster_label = 0
        self.mocap_excerpts = torch.from_numpy(self.model.get_cluster_mocap_excerpts(self.cluster_label))
        self.mocap_excerpt_count = self.mocap_excerpts.shape[0]

        self.seq_length = self.mocap_excerpts[0].shape[0]
        self.joint_count = self.mocap_excerpts[0].shape[1]
        self.joint_dim = self.mocap_excerpts[0].shape[2]
        self.pose_dim = self.joint_count * self.joint_dim
        
        self.joint_offsets = self.skeleton ["offsets"].astype(np.float32)
        self.joint_parents = self.skeleton ["parents"]
        self.joint_children = self.skeleton ["children"]
        
        self._create_edge_list()

        self.pred_pose = None
        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        
        self.mocap_excerpt_index = 0
        self.excerpt_frame_index = 0
        
    def _create_edge_list(self):
        
        self.edge_list = []
        
        for parent_joint_index in range(len(self.joint_children)):
            for child_joint_index in self.joint_children[parent_joint_index]:
                self.edge_list.append([parent_joint_index, child_joint_index])
                
    def setClusterLabel(self, label):
        
        self.mocap_excerpt_index = 0
        self.excerpt_frame_index = 0
        
        self.cluster_label = min(label, self.model.get_label_count() - 1)
        self.mocap_excerpts = torch.from_numpy(self.model.get_cluster_mocap_excerpts(self.cluster_label))
        self.mocap_excerpt_count = self.mocap_excerpts.shape[0]
        
    def selectMotionFeature(self, fleatureName):
        
        self.model.select_motion_feature(fleatureName)
        self.model.create_clusters()
        
        self.cluster_label = 0
        self.mocap_excerpt_index = 0
        self.excerpt_frame_index = 0
        
        self.mocap_excerpts = torch.from_numpy(self.model.get_cluster_mocap_excerpts(self.cluster_label))
        self.mocap_excerpt_count = self.mocap_excerpts.shape[0]
        
    def update(self):
        
        self._gen()
        
        # generate next skel pose
        self.pred_pose = self.pred_pose.reshape((-1, 4))
        self.pred_pose = nn.functional.normalize(self.pred_pose, p=2, dim=1)
        self.pred_pose = self.pred_pose.reshape((1, self.joint_count, self.joint_dim))
        
        zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32))
        zero_trajectory = zero_trajectory
        
        self.synth_pose_wpos, self.synth_pose_wrot = self._forward_kinematics(torch.unsqueeze(self.pred_pose,dim=0), zero_trajectory)
        
        self.synth_pose_wpos = self.synth_pose_wpos.detach().cpu().numpy()
        self.synth_pose_wpos = self.synth_pose_wpos.reshape((self.joint_count, 3))
        
        self.synth_pose_wrot = self.synth_pose_wrot.detach().cpu().numpy()
        self.synth_pose_wrot = self.synth_pose_wrot.reshape((self.joint_count, 4))

    def _gen(self):
        
        if self.excerpt_frame_index >= self.seq_window_length:
            self.mocap_excerpt_index += 1
            self.excerpt_frame_index  = self.seq_window_overlap

            if self.mocap_excerpt_index >= self.mocap_excerpt_count:
                self.mocap_excerpt_index = 0
        
        if self.mocap_excerpt_index == 0 and self.excerpt_frame_index < self.seq_window_offset:
            
            #print("result_seq[", rfI, "] = excerpts[", eI, "][", feI, "]")
            
            self.pred_pose = self.mocap_excerpts[self.mocap_excerpt_index][self.excerpt_frame_index]
            
        elif self.mocap_excerpt_index == self.mocap_excerpt_count - 1 and self.excerpt_frame_index  > self.seq_window_overlap:
            
            #print("result_seq[", rfI, "] = excerpts[", eI, "][", feI, "]")
            
            self.pred_pose = self.mocap_excerpts[self.mocap_excerpt_index][self.excerpt_frame_index] 
        elif self.mocap_excerpt_index < self.mocap_excerpt_count - 1:
            if self.excerpt_frame_index  < self.seq_window_offset:
                
                #print("result_seq[", rfI, "] = excerpts[", eI, "][", feI, "]")
                
                self.pred_pose = self.mocap_excerpts[self.mocap_excerpt_index][self.excerpt_frame_index]
            else:

                bI = self.excerpt_frame_index  - self.seq_window_offset
                blendValue = bI / (self.seq_window_overlap - 1)
                
                #print("result_seq[", rfI, "] = excerpts[", eI, "][", feI, "] + excerpts[", (eI+1), "][", (feI - window_offset), "] blend ", blendValue)

                for jI in range(self.joint_count):
                    quat1 = self.mocap_excerpts[self.mocap_excerpt_index][self.excerpt_frame_index , jI]
                    quat2 = self.mocap_excerpts[self.mocap_excerpt_index + 1][self.excerpt_frame_index  - self.seq_window_offset, jI]
                    quat_blend = slerp(quat1, quat2, blendValue)
                    
                    self.pred_pose[0, jI] = quat_blend
        
        self.excerpt_frame_index += 1

    def _forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4
        
        toffsets = torch.tensor(self.joint_offsets)
        
        positions_world = []
        rotations_world = []

        expanded_offsets = toffsets.expand(rotations.shape[0], rotations.shape[1], self.joint_offsets.shape[0], self.joint_offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for jI in range(self.joint_offsets.shape[0]):
            if self.joint_parents[jI] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self.joint_parents[jI]], expanded_offsets[:, :, jI]) \
                                       + positions_world[self.joint_parents[jI]])
                if len(self.joint_children[jI]) > 0:
                    rotations_world.append(qmul(rotations_world[self.joint_parents[jI]], rotations[:, :, jI]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(torch.Tensor([[[1.0, 0.0, 0.0, 0.0]]]))

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(rotations_world, dim=3).permute(0, 1, 3, 2)
        