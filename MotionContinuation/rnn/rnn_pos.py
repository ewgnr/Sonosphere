"""
Simple sequence continuation model based on an LSTM neural network
This version is meant for motion capture data that consists of joint positions only and typically stems from a pose estimation system
"""

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
import json
import pickle

from common import utils
from common.pose_renderer import PoseRenderer

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings
"""
# important: the skeleton needs to be identical in all mocap recordings

"""
mocap_config_file = "configs/Human36M_config.json" 
mocap_file_path = "mocap/"
mocap_files = ["Mocap_class_0_time_1724065746.5842216.pkl"]
mocap_valid_frame_ranges = [ [ [ 0, 9390 ] ] ]
mocap_sensor_ids = ["/mocap/0/joint/pos3d_world", "/mocap/0/joint/visibility"]
mocap_root_joint_name = "Bottom_Torso"
mocap_fps = 30
mocap_joint_dim = 3
"""

mocap_config_file = "configs/Halpe26_config.json" 
mocap_file_path = "mocap/"
mocap_files = ["Mocap_class_0_time_1723812067.0081663.pkl"]
mocap_valid_frame_ranges = [ [ [ 0, 9390 ] ] ]
mocap_sensor_ids = ["/mocap/0/joint/pos2d_world", "/mocap/0/joint/visibility"]
mocap_root_joint_name = "Hip"
mocap_fps = 30
mocap_joint_dim = 2

"""
Model Settings
"""

rnn_layer_dim = 512
rnn_layer_count = 2

save_weights = True
load_weights = False
rnn_weights_file = "results_xSens_stocos_takes1-7/weights/rnn_weights_epoch_200"

"""
Training settings
"""

batch_size = 32
test_percentage = 0.1
seq_input_length = 64
seq_output_length = 10 # this is only used for non-teacher forcing scenarios
learning_rate = 1e-4
pos_loss_scale = 1.0
teacher_forcing_prob = 0.0
model_save_interval = 10
epochs = 200
save_history = True
joint_loss_weights = [1.0]

"""
Visualization Settings
"""

if mocap_joint_dim == 2:
    view_ele = 90.0
    view_azi = 90.0
    view_line_width = 1.0
    view_size = 4.0
else:
    view_ele = 0.0
    view_azi = 0.0
    view_line_width = 1.0
    view_size = 4.0

"""
Load mocap data
"""

# load mocap config

with open(mocap_config_file) as f:
    mocap_config = json.load(f)


def config_to_skeletondata(mocap_config):
    
    skeleton_data = {}
    skeleton_data["joints"] = mocap_config["jointNames"]
    skeleton_data["root"] = skeleton_data["joints"][0]
    skeleton_data["parents"] = mocap_config["jointParents"]
    skeleton_data["children"] = mocap_config["jointChildren"]
    
    return skeleton_data

def recording_to_motiondata(mocap_recording, skeleton_data, mocap_sensor_ids):
    
    joint_count = len(skeleton_data["joints"])
    
    # gather sensor values
    motion_data = {}
    
    sensor_ids = mocap_recording["sensor_ids"]
    sensor_values = mocap_recording["sensor_values"]
    
    for sensor_id in mocap_sensor_ids:

        #print("sensor_id ", sensor_id)
        motion_data[sensor_id]  = [ sensor_values [vI] for vI in range(len(sensor_values)) if sensor_ids[vI].endswith(sensor_id) ]
        motion_data[sensor_id] = np.array(motion_data[sensor_id], dtype=np.float32)
        motion_data[sensor_id] = np.reshape(motion_data[sensor_id], (motion_data[sensor_id].shape[0], joint_count, -1))

    return motion_data

skeleton_data = config_to_skeletondata(mocap_config)

all_motion_data = []

for mocap_file in mocap_files:
    
    print("process file ", mocap_file)
    
    with open(mocap_file_path + "/" + mocap_file, "rb") as f:
        mocap_recording = pickle.load(f)
        
        motion_data = recording_to_motiondata(mocap_recording, skeleton_data, mocap_sensor_ids)
        
        all_motion_data.append(motion_data)
        
# retrieve mocap properties

joint_count = len(skeleton_data["joints"])
joint_dim = mocap_joint_dim
pose_dim = joint_count * joint_dim

parents = skeleton_data["parents"]
children = skeleton_data["children"]

joint_loss_weights = joint_loss_weights * joint_count

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

# set root position to zero
mocap_root_joint_index = skeleton_data["joints"].index(mocap_root_joint_name)

for motion_data in all_motion_data:

    if joint_dim == 3:
        joint_pos = motion_data["/mocap/0/joint/pos3d_world"]
    else:
        joint_pos = motion_data["/mocap/0/joint/pos2d_world"]
        
    root_pos = joint_pos[:, mocap_root_joint_index:mocap_root_joint_index+1, :]
    
    joint_pos_root_zero = joint_pos - root_pos
    
    motion_data["/mocap/0/joint/pos_root_zero"] = joint_pos_root_zero

# calculate pose normalisation values
pose_sequence_all = []
for motion_data in all_motion_data:
    pose_sequence = motion_data["/mocap/0/joint/pos_root_zero"]
    pose_sequence_all.append(pose_sequence)
    
pose_sequence_all = np.concatenate(pose_sequence_all, axis=0)
#pose_sequence_all = np.reshape(pose_sequence_all, (-1, pose_dim))

pose_mean = np.mean(pose_sequence_all, axis=0).flatten()
pose_std = np.std(pose_sequence_all, axis=0).flatten()

"""
Create Dataset
"""

X = []
y = []

for i, motion_data in enumerate(all_motion_data):
    
    print("mocap ", mocap_files[i])
    
    pose_sequence = motion_data["/mocap/0/joint/pos_root_zero"]
    pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))
    
    print("shape ", pose_sequence.shape)
    
    valid_frame_ranges = mocap_valid_frame_ranges[i]
    
    for valid_frame_range in valid_frame_ranges:
        
        frame_range_start = valid_frame_range[0]
        frame_range_end = valid_frame_range[1]
        
        print("frame range from ", frame_range_start, " to ", frame_range_end)
        
        for pI in np.arange(frame_range_start, frame_range_end - seq_input_length - seq_output_length - 1):

            X_sample = pose_sequence[pI:pI+seq_input_length]
            X.append(X_sample.reshape((seq_input_length, pose_dim)))
            
            Y_sample = pose_sequence[pI+seq_input_length:pI+seq_input_length+seq_output_length]  
            y.append(Y_sample.reshape((seq_output_length, pose_dim)))

X = np.array(X)
y = np.array(y)

X = torch.from_numpy(X).to(torch.float32)
y = torch.from_numpy(y).to(torch.float32)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, ...], self.y[idx, ...]

full_dataset = SequenceDataset(X, y)

X_item, y_item = full_dataset[0]

print("X_item s ", X_item.shape)
print("y_item s ", y_item.shape)

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X_batch, y_batch = next(iter(train_loader))

print("X_batch s ", X_batch.shape)
print("y_batch s ", y_batch.shape)

"""
Reccurent Model
"""

class Reccurent(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_count):
        super(Reccurent, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_count = layer_count
        self.output_dim = output_dim
            
        rnn_layers = []
        
        rnn_layers.append(("rnn", nn.LSTM(self.input_dim, self.hidden_dim, self.layer_count, batch_first=True)))
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        dense_layers = []
        dense_layers.append(("dense", nn.Linear(self.hidden_dim, self.output_dim)))
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
    
    def forward(self, x):
        x, (_, _) = self.rnn_layers(x)
        
        x = x[:, -1, :] # only last time step 
        x = self.dense_layers(x)
        
        return x

rnn = Reccurent(pose_dim, rnn_layer_dim, pose_dim, rnn_layer_count).to(device)
print(rnn)

# test Reccurent model

batch_x, _ = next(iter(train_loader))
batch_x = batch_x.to(device)

print(batch_x.shape)

test_y2 = rnn(batch_x)

print(test_y2.shape)

if load_weights == True:
    rnn.load_state_dict(torch.load(rnn_weights_file))

"""
Training
"""

optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.336) # reduce the learning every 20 epochs by a factor of 10

# pose mean and std

pose_mean = torch.tensor(pose_mean).reshape(1, 1, -1).to(device)
pose_std = torch.tensor(pose_std).reshape(1, 1, -1).to(device)

# joint loss weights

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32)
joint_loss_weights = joint_loss_weights.reshape(1, 1, -1).to(device)

def pos_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    #print("pos_loss")
    #print("y s ", y.shape)
    #print("yhat s ", yhat.shape)
    
    # reshape into batch_size x sequence_length x joint_count x joint_dim
    _y = y.reshape(y.shape[0], y.shape[1], joint_count, joint_dim )
    _yhat = yhat.reshape(y.shape[0], y.shape[1], joint_count, joint_dim )
    
    #print("_y s ", _y.shape)
    #print("_yhat s ", _yhat.shape)

    _pos_diff = torch.norm((_y - _yhat), dim=3)
    
    #print("_pos_diff s ", _pos_diff.shape)
    #print("joint_loss_weights s ", joint_loss_weights.shape)
    
    _pos_diff_weighted = _pos_diff * joint_loss_weights
    
    _loss = torch.mean(_pos_diff_weighted)

    return _loss


# position loss function
def loss(y, yhat):
    _pos_loss = pos_loss(y, yhat)
    
    _total_loss = 0.0
    _total_loss += _pos_loss * pos_loss_scale
    
    return _total_loss, _pos_loss

def train_step(pose_sequences, target_poses, teacher_forcing):
    
    rnn.train()

    #print("ar_train_step")    
    #print("teacher_forcing ", teacher_forcing)
    #print("pose_sequences s ", pose_sequences.shape)
    #print("target_poses s ", target_poses.shape)

    #_input_poses = pose_sequences.detach().clone()    
    _input_poses = pose_sequences  
    _input_poses_norm = (_input_poses - pose_mean) / pose_std
    _input_poses_norm = torch.nan_to_num(_input_poses_norm)
    
    target_poses_norm = (target_poses - pose_mean) / pose_std
    target_poses_norm = torch.nan_to_num(target_poses_norm)
    
    output_poses_length = target_poses_norm.shape[1]
    
    #print("output_poses_length ", output_poses_length)
    
    _pred_poses_norm_for_loss = []
    _target_poses_norm_for_loss = []
    
    for o_i in range(1, output_poses_length):
        
        #print("_input_poses_norm s ", _input_poses_norm.shape)
        
        _pred_poses_norm = rnn(_input_poses_norm)
        _pred_poses_norm = torch.unsqueeze(_pred_poses_norm, axis=1)
        
        #print("_pred_poses s ", _pred_poses.shape)
        
        _target_poses_norm = target_poses_norm[:,o_i,:].detach().clone()
        _target_poses_norm = torch.unsqueeze(_target_poses_norm, axis=1)

        #print("_target_poses_norm s ", _target_poses_norm.shape)
        
        _pred_poses_norm_for_loss.append(_pred_poses_norm)
        _target_poses_norm_for_loss.append(_target_poses_norm)
        
        # shift input pose seqeunce one pose to the right
        # remove pose from beginning input pose sequence
        # detach necessary to avoid error concerning running backprob a second time
        _input_poses_norm = _input_poses_norm[:, 1:, :].detach().clone()
        _target_poses_norm = _target_poses_norm.detach().clone()
        _pred_poses_norm = _pred_poses_norm.detach().clone()
        
        # add predicted or target pose to end of input pose sequence
        if teacher_forcing == True:
            _input_poses_norm = torch.concat((_input_poses_norm, _target_poses_norm), axis=1)
        else:
            #_pred_poses = torch.reshape(_pred_poses, (_pred_poses.shape[0], 1, joint_count, joint_dim))
            _input_poses_norm = torch.cat((_input_poses_norm, _pred_poses_norm), axis=1)
            
        #print("_input_poses s ", _input_poses.shape)

        
        #print("_input_poses 2 s ", _input_poses.shape)
        
    _pred_poses_norm_for_loss = torch.cat(_pred_poses_norm_for_loss, dim=1)
    _target_poses_norm_for_loss = torch.cat(_target_poses_norm_for_loss, dim=1)
    
    #print("_pred_poses_for_loss 2 s ", _pred_poses_for_loss.shape)
    #print("_target_poses_for_loss 2 s ", _target_poses_for_loss.shape)
    
    _loss, _pos_loss = loss(_target_poses_norm_for_loss, _pred_poses_norm_for_loss) 
    
    # Backpropagation
    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    #print("_ar_loss_total mean s ", _ar_loss_total.shape)
    
    #return _ar_loss, _ar_norm_loss, _ar_quat_loss
    
    return _loss, _pos_loss

def test_step(pose_sequences, target_poses, teacher_forcing):
    
    rnn.eval()

    #print("ar_train_step")    
    #print("teacher_forcing ", teacher_forcing)
    #print("pose_sequences s ", pose_sequences.shape)
    #print("target_poses s ", target_poses.shape)

    #_input_poses = pose_sequences.detach().clone()    
    _input_poses = pose_sequences  
    _input_poses_norm = (_input_poses - pose_mean) / pose_std
    _input_poses_norm = torch.nan_to_num(_input_poses_norm)
    
    target_poses_norm = (target_poses - pose_mean) / pose_std
    target_poses_norm = torch.nan_to_num(target_poses_norm)
    
    output_poses_length = target_poses_norm.shape[1]
    
    #print("output_poses_length ", output_poses_length)
    
    _pred_poses_norm_for_loss = []
    _target_poses_norm_for_loss = []
    
    with torch.no_grad():
    
        for o_i in range(1, output_poses_length):
            
            #print("_input_poses_norm s ", _input_poses_norm.shape)
            
            _pred_poses_norm = rnn(_input_poses_norm)
            _pred_poses_norm = torch.unsqueeze(_pred_poses_norm, axis=1)
            
            #print("_pred_poses s ", _pred_poses.shape)
            
            _target_poses_norm = target_poses_norm[:,o_i,:].detach().clone()
            _target_poses_norm = torch.unsqueeze(_target_poses_norm, axis=1)
    
            #print("_target_poses_norm s ", _target_poses_norm.shape)
            
            _pred_poses_norm_for_loss.append(_pred_poses_norm)
            _target_poses_norm_for_loss.append(_target_poses_norm)
            
            # shift input pose seqeunce one pose to the right
            # remove pose from beginning input pose sequence
            # detach necessary to avoid error concerning running backprob a second time
            _input_poses_norm = _input_poses_norm[:, 1:, :].detach().clone()
            _target_poses_norm = _target_poses_norm.detach().clone()
            _pred_poses_norm = _pred_poses_norm.detach().clone()
            
            # add predicted or target pose to end of input pose sequence
            if teacher_forcing == True:
                _input_poses_norm = torch.concat((_input_poses_norm, _target_poses_norm), axis=1)
            else:
                #_pred_poses = torch.reshape(_pred_poses, (_pred_poses.shape[0], 1, joint_count, joint_dim))
                _input_poses_norm = torch.cat((_input_poses_norm, _pred_poses_norm), axis=1)
                
            #print("_input_poses s ", _input_poses.shape)
    
            
            #print("_input_poses 2 s ", _input_poses.shape)
            
        _pred_poses_norm_for_loss = torch.cat(_pred_poses_norm_for_loss, dim=1)
        _target_poses_norm_for_loss = torch.cat(_target_poses_norm_for_loss, dim=1)
        
        #print("_pred_poses_for_loss 2 s ", _pred_poses_for_loss.shape)
        #print("_target_poses_for_loss 2 s ", _target_poses_for_loss.shape)
        
        _loss, _pos_loss = loss(_target_poses_norm_for_loss, _pred_poses_norm_for_loss) 
    
    #print("_ar_loss_total mean s ", _ar_loss_total.shape)
    
    #return _ar_loss, _ar_norm_loss, _ar_quat_loss
    
    rnn.train()
    
    return _loss, _pos_loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []
    loss_history["pos"] = []

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []
        _pos_loss_per_epoch = []

        for train_batch in train_dataloader:
            input_pose_sequences = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)
            
            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            
            _loss, _pos_loss = train_step(input_pose_sequences, target_poses, use_teacher_forcing)
            
            _loss = _loss.detach().cpu().numpy()
            _pos_loss = _pos_loss.detach().cpu().numpy()
            
            _train_loss_per_epoch.append(_loss)
            _pos_loss_per_epoch.append(_pos_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))
        _pos_loss_per_epoch = np.mean(np.array(_pos_loss_per_epoch))

        _test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            input_pose_sequences = train_batch[0].to(device)
            target_poses = train_batch[1].to(device)
            
            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            
            _loss, _ = test_step(input_pose_sequences, target_poses, use_teacher_forcing)
            
            _loss = _loss.detach().cpu().numpy()
            
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(rnn.state_dict(), "results/weights/rnn_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        loss_history["pos"].append(_pos_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} pos {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, _pos_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/rnn_history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/rnn_history_{}.png".format(epochs))

# save model weights
torch.save(rnn.state_dict(), "results/weights/rnn_weights_epoch_{}".format(epochs))

# inference and rendering 
poseRenderer = PoseRenderer(edge_list)

def export_sequence_anim(pose_sequence, file_name):

    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, joint_dim))
    
    if joint_dim == 2: # add third axis to pose sequence
        pose_sequence = np.concatenate([pose_sequence, np.zeros((pose_count, joint_count, 1))], axis=2)
    
    skel_sequence = pose_sequence

    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_sequence_pkl(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, pose_dim))

    export_dict = {}
    export_dict["class_id"] = 0
    if joint_dim == 2:
        export_dict["sensor_ids"] = ["/mocap/0/joint/pos2d_world"] * pose_count
    else:
        export_dict["sensor_ids"] = ["/mocap/0/joint/pos3d_world"] * pose_count

    # this should be a list of tuples, but I guess a list of lists is also fine
    export_dict["sensor_values"] = pose_sequence.tolist()
    export_dict["time_stamps"] = (np.arange(0, pose_count, 1) * (1.0 / mocap_fps)).tolist()
    
    with open(file_name, "wb") as f:
        pickle.dump(export_dict, f)
    
def create_pred_sequence(pose_sequence, pose_count):
    
    rnn.eval()

    start_seq = pose_sequence
    start_seq = torch.from_numpy(start_seq).to(device)
    start_seq = torch.reshape(start_seq, (seq_input_length, pose_dim))
    
    next_seq = start_seq

    pred_poses = []
    
    for i in range(pose_count):
        
        with torch.no_grad():
            
            next_seq_norm = (torch.unsqueeze(next_seq, axis=0) - pose_mean) / pose_std
            next_seq_norm = torch.nan_to_num(next_seq_norm)
            
            #print("next_seq_norm s ", next_seq_norm.shape)
            
            pred_pose_norm = rnn(next_seq_norm)
            
            #print("pred_pose_norm s ", pred_pose_norm.shape)
            
            pred_pose_norm = torch.unsqueeze(pred_pose_norm, axis=0)
            pred_pose = pred_pose_norm * pose_std + pose_mean
            pred_pose = torch.squeeze(pred_pose, axis=0)
            
            #print("pred_pose s ", pred_pose.shape)

        pred_poses.append(pred_pose)
    
        #print("next_seq s ", next_seq.shape)
        #print("pred_pose s ", pred_pose.shape)

        next_seq = torch.cat([next_seq[1:,:], pred_pose], axis=0)

    pred_poses = torch.cat(pred_poses, dim=0)
    pred_poses = pred_poses.reshape((pose_count, joint_count, joint_dim))
    
    pred_poses = pred_poses.detach().cpu().numpy()

    rnn.train()
    
    return pred_poses

# create original sequence

seq_index = 0
seq_start = 1000
seq_length = 1000

orig_sequence = all_motion_data[seq_index]["/mocap/0/joint/pos_root_zero"].astype(np.float32)

export_sequence_anim(orig_sequence[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.gif".format(seq_start, seq_length))
export_sequence_pkl(orig_sequence[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.pkl".format(seq_start, seq_length))

# create predicted sequence

seq_index = 0
seq_start = 1000
seq_length = 1000

orig_sequence = all_motion_data[seq_index]["/mocap/0/joint/pos_root_zero"].astype(np.float32)
pred_sequence = create_pred_sequence(orig_sequence[seq_start:seq_start+seq_input_length], seq_length)

export_sequence_anim(pred_sequence, "results/anims/pred_sequence_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_pkl(pred_sequence, "results/anims/pred_sequence_epoch_{}_seq_start_{}_length_{}.pkl".format(epochs, seq_start, seq_length))
