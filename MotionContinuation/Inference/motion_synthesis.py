import torch
from torch import nn
import numpy as np

from common.quaternion import qmul, qrot, qnormalize_np, qfix
from common.quaternion_torch import slerp

config = {"skeleton": None,
          "model": None,
          "seq_length": 64,
          "orig_sequences": [],
          "orig_seq_index": 0,
          "device": "cuda"
          }

"""
test_rot = torch.ones([4], dtype=torch.float32)
test_rot.shape

test_rot

test_rot_norm = nn.functional.normalize(test_rot, p=2, dim=0)

test_rot_norm
"""

class MotionSynthesis():
    
    def __init__(self, config):
        self.skeleton = config["skeleton"]
        self.model = config["model"]
        self.seq_length = config["seq_length"]
        self.device = config["device"]
        self.orig_sequences = config["orig_sequences"]
        self.orig_seq_index = config["orig_seq_index"]
        self.orig_seq_start_frame_index = 0
        self.orig_seq_frame_count = self.seq_length
        self.orig_seq_blend_factor = 1.0
        self.seq_rand_range = 0.00 # TODO: remove this, doesn't help a bit

        self.motion_seq = torch.from_numpy(self.orig_sequences[self.orig_seq_index][self.orig_seq_start_frame_index:self.orig_seq_start_frame_index + self.orig_seq_frame_count, ...]).to(self.device)
        
        self.orig_seq_changed = False
        
        self.joint_count = self.motion_seq.shape[1]
        self.joint_dim = self.motion_seq.shape[2]
        self.pose_dim = self.joint_count * self.joint_dim

        self.joint_offsets = self.skeleton ["offsets"].astype(np.float32)
        self.joint_parents = self.skeleton ["parents"]
        self.joint_children = self.skeleton ["children"]
        
        self._create_edge_list()
        
        self.synth_pose_wpos = None
        self.synth_pose_wrot = None
        self.synth_pose_lrot = None
        
    def _create_edge_list(self):
        
        self.edge_list = []
        
        for parent_joint_index in range(len(self.joint_children)):
            for child_joint_index in self.joint_children[parent_joint_index]:
                self.edge_list.append([parent_joint_index, child_joint_index])
                
    def setOrigSeqIndex(self, index):
        
        self.orig_seq_index = min(index, len(self.orig_sequences) - 1 ) 
        self.orig_seq_changed = True
        
    def setOrigSeqStartFrameIndex(self, index):
        
        self.orig_seq_start_frame_index = min(index, self.orig_sequences[self.orig_seq_index].shape[0] - self.seq_length ) 
        self.orig_seq_changed = True
        
    def setOrigSeqFrameCount(self, count):
        
        self.orig_seq_frame_count = min(count, self.seq_length)
        self.orig_seq_changed = True
        
    def setOrigSeqBlend(self, blend):
        
        self.orig_seq_blend_factor = max(min(1.0, blend), 0.0)
        self.orig_seq_changed = True
        
    def setRandRange(self, rand):
        self.seq_rand_range = rand
                
    def changeSequence(self):
        
        orig_seq = torch.from_numpy(self.orig_sequences[self.orig_seq_index][self.orig_seq_start_frame_index:self.orig_seq_start_frame_index + self.orig_seq_frame_count, ... ]).to(self.device)
        
        if self.orig_seq_blend_factor >= 1.0:
         
            if self.orig_seq_frame_count < self.seq_length:
                #self.motion_seq =  torch.concat( (self.motion_seq[:self.seq_length - self.orig_seq_frame_count, ...], orig_seq), dim=0)
                self.motion_seq =  torch.concat( (orig_seq, self.motion_seq[:self.seq_length - self.orig_seq_frame_count, ...]), dim=0)
                
            else:
                self.motion_seq = orig_seq
        else:

            if self.orig_seq_frame_count < self.seq_length:
                
                orig_seq = orig_seq.reshape(-1, self.joint_dim)
                cur_seq  = self.motion_seq[:self.orig_seq_frame_count, ...].reshape(-1, self.joint_dim)
                blend_factor = torch.ones([orig_seq.shape[0]], dtype=torch.float32).to(self.device) * self.orig_seq_blend_factor
                
                #print("frame_count ", self.orig_seq_frame_count, " orig_seq s ", orig_seq.shape, " cur_seq s ", cur_seq.shape, " blend_factor s ", blend_factor.shape)
                
                blend_seq = slerp(cur_seq, orig_seq, blend_factor)
                blend_seq = blend_seq.reshape(-1, self.joint_count, self.joint_dim)
                
                blend_seq = torch.concat( (blend_seq, self.motion_seq[:self.seq_length - self.orig_seq_frame_count, ...]), dim=0)
                
                self.motion_seq  = blend_seq
                
            else:
                
                orig_seq = orig_seq.reshape(-1, self.joint_dim)
                cur_seq  = self.motion_seq.reshape(-1, self.joint_dim)
                blend_factor = torch.ones([orig_seq.shape[0]], dtype=torch.float32).to(self.device) * self.orig_seq_blend_factor
                
                blend_seq = slerp(cur_seq, orig_seq, blend_factor)
                blend_seq = blend_seq.reshape(-1, self.joint_count, self.joint_dim)
                
                self.motion_seq  = blend_seq

        self.orig_seq_changed = False
        
    def setJointRotation(self, joint_index, joint_rot, frame_count):
        
        #print("setJointRotation index ", joint_index, " rot ", joint_rot)
        
        """
        seq_length = self.motion_seq.shape[0]
        joint_rot = np.repeat(np.expand_dims(joint_rot, axis=0), seq_length, axis=0)
        
        joint_rot = torch.from_numpy(joint_rot).to(self.device)
        self.motion_seq[:, joint_index, :] = joint_rot
        """
        
        joint_rot = torch.from_numpy(joint_rot).to(self.device)
        
        if frame_count == 1:
            #self.motion_seq[-1, joint_index, :] = joint_rot
            self.motion_seq[-2, joint_index, :] = joint_rot
        elif frame_count > 1:
            frame_count = min(frame_count, self.seq_length)
            joint_rot = torch.unsqueeze(joint_rot, dim=0).repeat(frame_count, 1)
            self.motion_seq[:frame_count, joint_index, :] = joint_rot

    def changeJointRotation(self, joint_index, joint_rot, frame_count):
        
        #print("setJointRotation index ", joint_index, " rot ", joint_rot)
        
        """
        seq_length = self.motion_seq.shape[0]
        joint_rot = np.repeat(np.expand_dims(joint_rot, axis=0), seq_length, axis=0)
        
        joint_rot = torch.from_numpy(joint_rot).to(self.device)
        self.motion_seq[:, joint_index, :] = joint_rot
        """
        
        joint_rot = torch.from_numpy(joint_rot).to(torch.float32).to(self.device)
        joint_rot = nn.functional.normalize(joint_rot, p=2, dim=0)
        
        #print("joint_rot ", joint_rot, " frame_count ", frame_count)
        
        #print("joint_rot s ", joint_rot.shape)
        
        if frame_count == 1:
            
            #self.motion_seq[-1, joint_index, :] = qmul(self.motion_seq[-1, joint_index, :], joint_rot)
            self.motion_seq[-2, joint_index, :] = qmul(self.motion_seq[-1, joint_index, :], joint_rot)

        elif frame_count > 1:
            frame_count = min(frame_count, self.seq_length)
            joint_rot = torch.unsqueeze(joint_rot, dim=0).repeat(frame_count, 1)
            self.motion_seq[:frame_count, joint_index, :] *= joint_rot            
    
    def update(self):
        
        if self.orig_seq_changed == True:
            self.changeSequence()
            
        #self.model.eval()
        
        with torch.no_grad():
            self.pred_pose = self.model(torch.unsqueeze(self.motion_seq.reshape(-1, self.pose_dim), axis=0))
                
        # normalize pred pose
        self.pred_pose = torch.squeeze(self.pred_pose)
        self.pred_pose = self.pred_pose.reshape((-1, 4))
        self.pred_pose = nn.functional.normalize(self.pred_pose, p=2, dim=1)
        self.pred_pose = self.pred_pose.reshape((1, self.joint_count, self.joint_dim))
        
        self.synth_pose_lrot = self.pred_pose.detach().cpu().numpy()
        self.synth_pose_lrot = self.synth_pose_lrot.reshape((self.joint_count, 4))
        
        """
        # debug randomize pred pose
        rand_rot = torch.rand([self.pred_pose.shape[0], 4], dtype=torch.float32).to(self.device)
        rand_rot = nn.functional.normalize(rand_rot, p=2, dim=1)
        rand_range = torch.ones(self.pred_pose.shape[0], dtype=torch.float32).to(self.device) * self.seq_rand_range
        rand_pos = slerp(self.pred_pose, rand_rot, rand_range)
        self.pred_pose = rand_pos.reshape((1, self.joint_count, self.joint_dim))
        """
    
        # append pred pose to sequence
        self.motion_seq = torch.cat([self.motion_seq[1:,:], self.pred_pose], axis=0)
        
        # debug randomize first pose in motion seq
        if self.seq_rand_range > 0:
            rand_rot = torch.rand([self.joint_count, 4], dtype=torch.float32).to(self.device)
            rand_rot = nn.functional.normalize(rand_rot, p=2, dim=1)
            rand_range = torch.ones(self.joint_count, dtype=torch.float32).to(self.device) * self.seq_rand_range
            self.motion_seq[0] = (slerp(self.motion_seq[0], rand_rot, rand_range))

        # convert quaternion pose to position pose
        zero_trajectory = torch.tensor(np.zeros((1, 1, 3), dtype=np.float32))
        zero_trajectory = zero_trajectory.to(self.device)
            
        self.synth_pose_wpos, self.synth_pose_wrot = self._forward_kinematics(torch.unsqueeze(self.pred_pose,dim=0), zero_trajectory)

        self.synth_pose_wpos = self.synth_pose_wpos.detach().cpu().numpy()
        self.synth_pose_wpos = self.synth_pose_wpos.reshape((self.joint_count, 3))
        
        self.synth_pose_wrot = self.synth_pose_wrot.detach().cpu().numpy()
        self.synth_pose_wrot = self.synth_pose_wrot.reshape((self.joint_count, 4))
        
        #self.model.train()

                
    def _forward_kinematics(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4
        
        toffsets = torch.tensor(self.joint_offsets).to(self.device)
        
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
                    rotations_world.append(torch.Tensor([[[1.0, 0.0, 0.0, 0.0]]]).to(self.device))
                    
        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2), torch.stack(rotations_world, dim=3).permute(0, 1, 3, 2)

    
        
        

        
    
    