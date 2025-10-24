"""
same as vae_rnn.py but with 6d representation for joint rotations
"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np
import json

from common import utils
from common import bvh_tools as bvh
from common import fbx_tools as fbx
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer
import common.repr6d_torch as repr6d_torch

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

"""
Compute Unit
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Mocap Settings

important: the skeleton needs to be identical in all mocap recordings
"""

# Example: XSens Mocap Recording
mocap_file_path = "data/mocap"
mocap_files = ["Muriel_Embodied_Machine_variation.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_loss_weights_file = None


"""
# Example: ZED Mocap Recording
mocap_file_path = "../../../Data/Mocap/Zed/Daniel/Solos/fbx_30hz"
mocap_files = ["daniel_zed_solo1.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 30
mocap_loss_weights_file = "data/configs/zed_body34_joint_loss_weights.json"
"""

"""
# Example: Captury Mocap Recording
mocap_file_path = "../../../Data/Mocap/Captury/MotionBank/Solos/fbx_50hz"
mocap_files = ["zachary_music_improvisation.fbx"]
mocap_pos_scale = 0.1
mocap_fps = 50
mocap_loss_weights_file = None
"""

"""
# Example: XSens Mocap Recording
mocap_file_path = "../../../Data/Mocap/XSens/Stocos/Solos/fbx_50hz"
mocap_files = ["Muriel_Embodied_Machine_variation.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_loss_weights_file = None
"""

"""
# Example: Qualisys Mocap Recording
mocap_file_path = "../../../Data/Mocap/Qualisys/Stocos/Solos/fbx_50hz"
mocap_files = ["polytopia_fullbody_take2.fbx"]
mocap_pos_scale = 1.0
mocap_fps = 50
mocap_loss_weights_file = "data/configs/qualisys_with_hands_joint_loss_weights.json"
"""

"""
Model Settings
"""

latent_dim = 32
sequence_length = 64
ae_rnn_layer_count = 2
ae_rnn_layer_size = 512
ae_dense_layer_sizes = [ 512 ]

save_models = False
save_tscript = False
save_weights = True

# load model weights
load_weights = False
encoder_weights_file = "results/weights/encoder_weights_epoch_600"
decoder_weights_file = "results/weights/decoder_weights_epoch_600"

"""
Training Settings
"""

sequence_offset = 2 # when creating sequence excerpts, each excerpt is offset from the previous one by this value
batch_size = 64
test_percentage  = 0.2
ae_learning_rate = 1e-4
ae_norm_loss_scale = 0.1
ae_pos_loss_scale = 0.1
ae_quat_loss_scale = 1.0
ae_kld_loss_scale = 0.0 # will be calculated
kld_scale_cycle_duration = 100
kld_scale_min_const_duration = 20
kld_scale_max_const_duration = 20
min_kld_scale = 0.0
max_kld_scale = 0.1

epochs = 600
model_save_interval = 50
save_history = True

"""
Visualization Settings
"""

view_ele = 90.0
view_azi = -90.0
view_line_width = 1.0
view_size = 4.0

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
    
    rot_sequence = mocap_data["rot_sequence"]
    
    if mocap_file.endswith(".bvh") or mocap_file.endswith(".BVH"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat_bvh(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    elif mocap_file.endswith(".fbx") or mocap_file.endswith(".FBX"):
        mocap_data["motion"]["rot_local"] = mocap_tools.euler_to_quat(mocap_data["motion"]["rot_local_euler"], mocap_data["rot_sequence"])
    
    mocap_data["motion"]["rot_6d"] = repr6d_torch.quat2repr6d( torch.from_numpy(mocap_data["motion"]["rot_local"]) ).numpy()

    all_mocap_data.append(mocap_data)

# retrieve mocap properties

mocap_data = all_mocap_data[0]
joint_count = mocap_data["motion"]["rot_6d"].shape[1]
joint_dim = mocap_data["motion"]["rot_6d"].shape[2]
pose_dim = joint_count * joint_dim

offsets = mocap_data["skeleton"]["offsets"].astype(np.float32)
parents = mocap_data["skeleton"]["parents"]
children = mocap_data["skeleton"]["children"]

# create edge list
def get_edge_list(children):
    edge_list = []

    for parent_joint_index in range(len(children)):
        for child_joint_index in children[parent_joint_index]:
            edge_list.append([parent_joint_index, child_joint_index])
    
    return edge_list

edge_list = get_edge_list(children)

# set joint loss weigths 

if mocap_loss_weights_file is not None:
    with open(mocap_loss_weights_file) as f:
        joint_loss_weights = json.load(f)
        joint_loss_weights = joint_loss_weights["joint_loss_weights"]
else:
    joint_loss_weights = [1.0]
    joint_loss_weights *= joint_count

"""
Create Dataset
"""

# gather pose sequence excerpts

pose_sequence_excerpts = []

for mocap_data in all_mocap_data:
    pose_sequence = mocap_data["motion"]["rot_6d"]
    pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))
    
    frame_range_start = 0
    frame_range_end = pose_sequence.shape[0]
    
    for seq_excerpt_start in np.arange(frame_range_start, frame_range_end - sequence_length, sequence_offset):
        #print("valid: start ", frame_range_start, " end ", frame_range_end, " exc: start ", seq_excerpt_start, " end ", (seq_excerpt_start + sequence_length) )
        pose_sequence_excerpt =  pose_sequence[seq_excerpt_start:seq_excerpt_start + sequence_length]
        pose_sequence_excerpts.append(pose_sequence_excerpt)
    
pose_sequence_excerpts = np.array(pose_sequence_excerpts, dtype=np.float32)

# create dataset

sequence_excerpts_count = pose_sequence_excerpts.shape[0]

class SequenceDataset(Dataset):
    def __init__(self, sequence_excerpts):
        self.sequence_excerpts = sequence_excerpts
    
    def __len__(self):
        return self.sequence_excerpts.shape[0]
    
    def __getitem__(self, idx):
        return self.sequence_excerpts[idx, ...]
        

full_dataset = SequenceDataset(pose_sequence_excerpts)
dataset_size = len(full_dataset)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
Create Models
"""

# create encoder model

class Encoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Encoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_count = rnn_layer_count
        self.rnn_layer_size = rnn_layer_size 
        self.dense_layer_sizes = dense_layer_sizes
    
        # create recurrent layers
        rnn_layers = []
        rnn_layers.append(("encoder_rnn_0", nn.LSTM(self.pose_dim, self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # create dense layers
        
        dense_layers = []
        
        dense_layers.append(("encoder_dense_0", nn.Linear(self.rnn_layer_size, self.dense_layer_sizes[0])))
        dense_layers.append(("encoder_dense_relu_0", nn.ReLU()))
        
        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("encoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("encoder_dense_relu_{}".format(layer_index), nn.ReLU()))
            
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create final dense layers
            
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        
    def forward(self, x):
        
        #print("x 1 ", x.shape)
        
        x, (_, _) = self.rnn_layers(x)
        
        #print("x 2 ", x.shape)
        
        x = x[:, -1, :] # only last time step 
        
        #print("x 3 ", x.shape)
        
        x = self.dense_layers(x)
        
        #print("x 3 ", x.shape)
        
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        #print("mu s ", mu.shape, " lvar s ", log_var.shape)
    
        return mu, std
    
encoder = Encoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_dense_layer_sizes).to(device)

print(encoder)

if load_weights and encoder_weights_file:
    encoder.load_state_dict(torch.load(encoder_weights_file, map_location=device))
    
# create decoder model

class Decoder(nn.Module):
    def __init__(self, sequence_length, pose_dim, latent_dim, rnn_layer_count, rnn_layer_size, dense_layer_sizes):
        super(Decoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.pose_dim = pose_dim
        self.latent_dim = latent_dim
        self.rnn_layer_size = rnn_layer_size
        self.rnn_layer_count = rnn_layer_count
        self.dense_layer_sizes = dense_layer_sizes

        # create dense layers
        dense_layers = []
        
        dense_layers.append(("decoder_dense_0", nn.Linear(latent_dim, self.dense_layer_sizes[0])))
        dense_layers.append(("decoder_relu_0", nn.ReLU()))

        dense_layer_count = len(self.dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            dense_layers.append(("decoder_dense_{}".format(layer_index), nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index])))
            dense_layers.append(("decoder_dense_relu_{}".format(layer_index), nn.ReLU()))
 
        self.dense_layers = nn.Sequential(OrderedDict(dense_layers))
        
        # create rnn layers
        rnn_layers = []

        rnn_layers.append(("decoder_rnn_0", nn.LSTM(self.dense_layer_sizes[-1], self.rnn_layer_size, self.rnn_layer_count, batch_first=True)))
        
        self.rnn_layers = nn.Sequential(OrderedDict(rnn_layers))
        
        # final output dense layer
        final_layers = []
        
        final_layers.append(("decoder_dense_{}".format(dense_layer_count), nn.Linear(self.rnn_layer_size, self.pose_dim)))
        
        self.final_layers = nn.Sequential(OrderedDict(final_layers))
        
    def forward(self, x):
        #print("x 1 ", x.size())
        
        # dense layers
        x = self.dense_layers(x)
        #print("x 2 ", x.size())
        
        # repeat vector
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, sequence_length, 1)
        #print("x 3 ", x.size())
        
        # rnn layers
        x, (_, _) = self.rnn_layers(x)
        #print("x 4 ", x.size())
        
        # final time distributed dense layer
        x_reshaped = x.contiguous().view(-1, self.rnn_layer_size)  # (batch_size * sequence, input_size)
        #print("x 5 ", x_reshaped.size())
        
        yhat = self.final_layers(x_reshaped)
        #print("yhat 1 ", yhat.size())
        
        yhat = yhat.contiguous().view(-1, self.sequence_length, self.pose_dim)
        #print("yhat 2 ", yhat.size())

        return yhat

ae_dense_layer_sizes_reversed = ae_dense_layer_sizes.copy()
ae_dense_layer_sizes_reversed.reverse()

decoder = Decoder(sequence_length, pose_dim, latent_dim, ae_rnn_layer_count, ae_rnn_layer_size, ae_dense_layer_sizes_reversed).to(device)

print(decoder)

if load_weights and decoder_weights_file:
    decoder.load_state_dict(torch.load(decoder_weights_file, map_location=device))
    
"""
Training
"""

def calc_kld_scales():
    
    kld_scales = []

    for e in range(epochs):
        
        cycle_step = e % kld_scale_cycle_duration
        
        #print("cycle_step ", cycle_step)


        if cycle_step < kld_scale_min_const_duration:
            kld_scale = min_kld_scale
            kld_scales.append(kld_scale)
        elif cycle_step > kld_scale_cycle_duration - kld_scale_max_const_duration:
            kld_scale = max_kld_scale
            kld_scales.append(kld_scale)
        else:
            lin_step = cycle_step - kld_scale_min_const_duration
            kld_scale = min_kld_scale + (max_kld_scale - min_kld_scale) * lin_step / (kld_scale_cycle_duration - kld_scale_min_const_duration - kld_scale_max_const_duration)
            kld_scales.append(kld_scale)
            
    return kld_scales

kld_scales = calc_kld_scales()

ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=ae_learning_rate)
ae_scheduler = torch.optim.lr_scheduler.StepLR(ae_optimizer, step_size=100, gamma=0.316) # reduce the learning every 100 epochs by a factor of 10

mse_loss = nn.MSELoss()
cross_entropy = nn.BCELoss()

# joint loss weights
joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32)
joint_loss_weights = joint_loss_weights.reshape(1, 1, -1).to(device)

# KL Divergence

def variational_loss(mu, std):
    #returns the variational loss from arguments mean and standard deviation std
    #see also: see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #https://arxiv.org/abs/1312.6114
    vl=-0.5*torch.mean(1+ 2*torch.log(std)-mu.pow(2) -(std.pow(2)))
    return vl
   
def variational_loss2(mu, std):
    #returns the variational loss from arguments mean and standard deviation std
    #alternative: mean squared distance from ideal mu=0 and std=1:
    vl=torch.mean(mu.pow(2)+(1-std).pow(2))
    return vl

def reparameterize(mu, std):
    z = mu + std*torch.randn_like(std)
    return z

def ae_norm_loss(yhat):
    """
    Computes 6D rotation normalization loss:
      - encourages both columns to have unit length
      - encourages orthogonality between them

    Args:
        yhat: (..., J*6) tensor (predicted 6D rep for all joints in a batch)
    Returns:
        loss: scalar tensor
    """
    # Reshape to (..., J, 6)
    yhat = yhat.view(-1, yhat.shape[-1] // 6, 6)
    x_raw = yhat[..., :3]
    y_raw = yhat[..., 3:6]
    # Unit length loss
    x_norm = torch.norm(x_raw, dim=-1, keepdim=True)
    y_norm = torch.norm(y_raw, dim=-1, keepdim=True)
    loss_norm = ((x_norm - 1)**2).mean() + ((y_norm - 1)**2).mean()
    # Orthogonality loss
    dot = (x_raw * y_raw).sum(dim=-1, keepdim=True)
    loss_ortho = (dot ** 2).mean()
    return loss_norm + loss_ortho

def forward_kinematics(rotations_r6d, root_positions):
    """
    Args:
        rotations_r6d: (N, L, J, 6) tensor of 6D joint rotations
        root_positions: (N, L, 3) tensor
        offsets: (J, 3) numpy array (or tensor)
        parents: list of parent indices for each joint
        children: list of children indices for each joint
    Returns:
        positions_world: (N, L, J, 3) tensor of world joint positions
    """
    
    assert len(rotations_r6d.shape) == 4
    assert rotations_r6d.shape[-1] == 6
    
    rotations_quat = repr6d_torch.repr6d2quat(rotations_r6d)
    
    toffsets = torch.tensor(offsets).to(device)
    
    positions_world = []
    rotations_world = []

    expanded_offsets = toffsets.expand(rotations_quat.shape[0], rotations_quat.shape[1], offsets.shape[0], offsets.shape[1])

    # Parallelize along the batch and time dimensions
    for jI in range(offsets.shape[0]):
        if parents[jI] == -1:
            positions_world.append(root_positions)
            rotations_world.append(rotations_quat[:, :, 0])
        else:
            positions_world.append(qrot(rotations_world[parents[jI]], expanded_offsets[:, :, jI]) \
                                   + positions_world[parents[jI]])
            if len(children[jI]) > 0:
                rotations_world.append(qmul(rotations_world[parents[jI]], rotations_quat[:, :, jI]))
            else:
                # This joint is a terminal node -> it would be useless to compute the transformation
                rotations_world.append(None)

    return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

def ae_pos_loss(y, yhat):
    # y and yhat shapes: batch_size, seq_length, pose_dim

    """
    # normalize tensors
    _yhat = yhat.view(-1, 4)

    _yhat_norm = nn.functional.normalize(_yhat, p=2, dim=1)
    _y_rot = y.view((y.shape[0], y.shape[1], -1, 4))
    _yhat_rot = _yhat.view((y.shape[0], y.shape[1], -1, 4))
    """
    
    _y_rot = y.view((y.shape[0], y.shape[1], -1, 6))
    _yhat_rot = yhat.view((y.shape[0], y.shape[1], -1, 6))
    
    zero_trajectory = torch.zeros((y.shape[0], y.shape[1], 3), dtype=torch.float32, requires_grad=True).to(device)

    _y_pos = forward_kinematics(_y_rot, zero_trajectory)
    _yhat_pos = forward_kinematics(_yhat_rot, zero_trajectory)

    _pos_diff = torch.norm((_y_pos - _yhat_pos), dim=3)
    
    #print("_pos_diff s ", _pos_diff.shape)
    
    _pos_diff_weighted = _pos_diff * joint_loss_weights
    
    _loss = torch.mean(_pos_diff_weighted)

    return _loss

def ae_rot_loss(y, yhat):
    """
    Computes MSE between true and predicted joint rotation matrices from 6D reps.
    Args:
        y: (..., J*6) ground truth 6D rotations
        yhat: (..., J*6) predicted 6D rotations
    Returns:
        loss: scalar tensor
    """
    # Reshape to (..., J, 6)
    J = y.shape[-1] // 6
    y = y.view(-1, J, 6)
    yhat = yhat.view(-1, J, 6)

    # Convert to rotmats: (..., J, 3, 3)
    y_rotmat = repr6d_torch.repr6d2mat(y)   # [N,J,3,3]
    yhat_rotmat = repr6d_torch.repr6d2mat(yhat) # [N,J,3,3]
    
    # Mean squared error over matrices
    loss = torch.mean((y_rotmat - yhat_rotmat)**2)
    return loss

# autoencoder loss function
def ae_loss(y, yhat, mu, std):
    # function parameters
    # y: encoder input
    # yhat: decoder output (i.e. reconstructed encoder input)
    # disc_fake_output: discriminator output for encoder generated prior
    
    _norm_loss = ae_norm_loss(yhat)
    _pos_loss = ae_pos_loss(y, yhat)
    _rot_loss = ae_rot_loss(y, yhat)

    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
    
    _total_loss = 0.0
    _total_loss += _norm_loss * ae_norm_loss_scale
    _total_loss += _pos_loss * ae_pos_loss_scale
    _total_loss += _rot_loss * ae_quat_loss_scale
    _total_loss += _ae_kld_loss * ae_kld_loss_scale
    
    return _total_loss, _norm_loss, _pos_loss, _rot_loss, _ae_kld_loss

def ae_train_step(target_poses):
    
    #print("train step target_poses ", target_poses.shape)
 
    # let autoencoder preproduce target_poses (decoder output) and also return encoder output
    encoder_output = encoder(target_poses)

    encoder_output_mu = encoder_output[0]
    encoder_output_std = encoder_output[1]
    mu = torch.tanh(encoder_output_mu)
    std = torch.abs(torch.tanh(encoder_output_std)) + 0.00001
    decoder_input = reparameterize(mu, std)
    
    pred_poses = decoder(decoder_input)
    
    #print("pred_poses s ", pred_poses.shape)
    
    _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_kld_loss = ae_loss(target_poses, pred_poses, mu, std) 

    #print("_ae_pos_loss ", _ae_pos_loss)
    
    # Backpropagation
    ae_optimizer.zero_grad()
    _ae_loss.backward()
    
    #torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.01)
    #torch.nn.utils.clip_grad_norm(decoder.parameters(), 0.01)

    ae_optimizer.step()
    
    return _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_kld_loss

def ae_test_step(target_poses):
    with torch.no_grad():
        # let autoencoder preproduce target_poses (decoder output) and also return encoder output
        encoder_output = encoder(target_poses)
        
        encoder_output_mu = encoder_output[0]
        encoder_output_std = encoder_output[1]
        mu = torch.tanh(encoder_output_mu)
        std = torch.abs(torch.tanh(encoder_output_std)) + 0.00001
        decoder_input = reparameterize(mu, std)
        
        pred_poses = decoder(decoder_input)
        
        _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_kld_loss = ae_loss(target_poses, pred_poses, mu, std)  
    
    return _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_kld_loss

def train(train_dataloader, test_dataloader, epochs):
    
    global ae_kld_loss_scale
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae test"] = []
    loss_history["ae norm"] = []
    loss_history["ae pos"] = []
    loss_history["ae quat"] = []
    loss_history["ae kld"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_kld_loss_scale = kld_scales[epoch]
        
        #print("ae_kld_loss_scale ", ae_kld_loss_scale)
        
        ae_train_loss_per_epoch = []
        ae_norm_loss_per_epoch = []
        ae_pos_loss_per_epoch = []
        ae_quat_loss_per_epoch = []
        ae_kld_loss_per_epoch = []
        
        for train_batch in train_dataloader:
            train_batch = train_batch.to(device)
            
            _ae_loss, _ae_norm_loss, _ae_pos_loss, _ae_quat_loss, _ae_kld_loss = ae_train_step(train_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_norm_loss = _ae_norm_loss.detach().cpu().numpy()
            _ae_pos_loss = _ae_pos_loss.detach().cpu().numpy()
            _ae_quat_loss = _ae_quat_loss.detach().cpu().numpy()
            _ae_kld_loss = _ae_kld_loss.detach().cpu().numpy()
            
            #print("_ae_prior_loss ", _ae_prior_loss)
            
            ae_train_loss_per_epoch.append(_ae_loss)
            ae_norm_loss_per_epoch.append(_ae_norm_loss)
            ae_pos_loss_per_epoch.append(_ae_pos_loss)
            ae_quat_loss_per_epoch.append(_ae_quat_loss)
            ae_kld_loss_per_epoch.append(_ae_kld_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_norm_loss_per_epoch = np.mean(np.array(ae_norm_loss_per_epoch))
        ae_pos_loss_per_epoch = np.mean(np.array(ae_pos_loss_per_epoch))
        ae_quat_loss_per_epoch = np.mean(np.array(ae_quat_loss_per_epoch))
        ae_kld_loss_per_epoch = np.mean(np.array(ae_kld_loss_per_epoch))

        ae_test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            test_batch = test_batch.to(device)
            
            _ae_loss, _, _, _, _ = ae_test_step(test_batch)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            ae_test_loss_per_epoch.append(_ae_loss)
        
        ae_test_loss_per_epoch = np.mean(np.array(ae_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epoch))
            torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epoch))
        
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae test"].append(ae_test_loss_per_epoch)
        loss_history["ae norm"].append(ae_norm_loss_per_epoch)
        loss_history["ae pos"].append(ae_pos_loss_per_epoch)
        loss_history["ae quat"].append(ae_quat_loss_per_epoch)
        loss_history["ae kld"].append(ae_kld_loss_per_epoch)
        
        print ('epoch {} : ae train: {:01.4f} ae test: {:01.4f} norm {:01.4f} pos {:01.4f} quat {:01.4f} kld {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_test_loss_per_epoch, ae_norm_loss_per_epoch, ae_pos_loss_per_epoch, ae_quat_loss_per_epoch, ae_kld_loss_per_epoch, time.time()-start))
    
        ae_scheduler.step()
        
    return loss_history

# fit model
loss_history = train(train_dataloader, test_dataloader, epochs)

# save history
utils.save_loss_as_csv(loss_history, "results/histories/history_{}.csv".format(epochs))
utils.save_loss_as_image(loss_history, "results/histories/history_{}.png".format(epochs))

# save model weights
torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epochs))
torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epochs))


# inference and rendering 

poseRenderer = PoseRenderer(edge_list)

def safe_slerp(q1, q2, t):
    # q1, q2: (..., 4)
    # t: interpolation parameter
    dot = (q1 * q2).sum(axis=-1, keepdims=True)
    q2_fixed = np.where(dot < 0, -q2, q2)
    return slerp(q1, q2_fixed, t)

def consistent_quaternion_seq(seq1, seq2):
    """
    Enforces sign consistency for two quaternion sequences for blending.

    Both seq1 and seq2 should be np.ndarray of shape (L, J, 4).

    The function makes sure:
    - Each joint's quaternions are consistent over frames in seq1.
    - Each joint's quaternions are consistent over frames in seq2.
    - The last frame of seq1 for each joint is sign-consistent with the first frame of seq2.

    Returns:
        seq1_out, seq2_out : np.ndarray, with sign-consistent quaternions.
    """
    seq1_out = np.copy(seq1)
    seq2_out = np.copy(seq2)
    L1, J, _ = seq1_out.shape
    L2 = seq2_out.shape[0]

    # Enforce consistency within seq1
    for j in range(J):
        for i in range(1, L1):
            if np.dot(seq1_out[i, j], seq1_out[i-1, j]) < 0:
                seq1_out[i, j] *= -1

    # Enforce consistency within seq2
    for j in range(J):
        for i in range(1, L2):
            if np.dot(seq2_out[i, j], seq2_out[i-1, j]) < 0:
                seq2_out[i, j] *= -1

    # Enforce consistency between seq1's last frame and seq2's first frame (for each joint)
    for j in range(J):
        if np.dot(seq1_out[-1, j], seq2_out[0, j]) < 0:
            seq2_out[:, j, :] *= -1

    return seq1_out, seq2_out

def export_sequence_anim(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    pose_sequence = np.reshape(pose_sequence, (pose_count, joint_count, joint_dim))
    
    pose_sequence = torch.tensor(np.expand_dims(pose_sequence, axis=0)).to(device)
    zero_trajectory = torch.tensor(np.zeros((1, pose_count, 3), dtype=np.float32)).to(device)
    
    skel_sequence = forward_kinematics(pose_sequence, zero_trajectory)
    
    skel_sequence = skel_sequence.detach().cpu().numpy()
    skel_sequence = np.squeeze(skel_sequence)    
    
    view_min, view_max = utils.get_equal_mix_max_positions(skel_sequence)
    skel_images = poseRenderer.create_pose_images(skel_sequence, view_min, view_max, view_ele, view_azi, view_line_width, view_size, view_size)
    skel_images[0].save(file_name, save_all=True, append_images=skel_images[1:], optimize=False, duration=33.0, loop=0)

def export_sequence_bvh(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]

    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_6d"] = pose_sequence
    pred_dataset["motion"]["rot_local"] = repr6d_torch.repr6d2quat(torch.from_numpy(mocap_data["motion"]["rot_6d"]) ).numpy()
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler_bvh(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])

    pred_bvh = mocap_tools.mocap_to_bvh(pred_dataset)
    
    bvh_tools.write(pred_bvh, file_name)

def export_sequence_fbx(pose_sequence, file_name):
    
    pose_count = pose_sequence.shape[0]
    
    pred_dataset = {}
    pred_dataset["frame_rate"] = mocap_data["frame_rate"]
    pred_dataset["rot_sequence"] = mocap_data["rot_sequence"]
    pred_dataset["skeleton"] = mocap_data["skeleton"]
    pred_dataset["motion"] = {}
    pred_dataset["motion"]["pos_local"] = np.repeat(np.expand_dims(pred_dataset["skeleton"]["offsets"], axis=0), pose_count, axis=0)
    pred_dataset["motion"]["rot_6d"] = pose_sequence
    pred_dataset["motion"]["rot_local"] = repr6d_torch.repr6d2quat(torch.from_numpy(mocap_data["motion"]["rot_6d"]) ).numpy()
    pred_dataset["motion"]["rot_local_euler"] = mocap_tools.quat_to_euler(pred_dataset["motion"]["rot_local"], pred_dataset["rot_sequence"])
    
    pred_fbx = mocap_tools.mocap_to_fbx([pred_dataset])
    
    fbx_tools.write(pred_fbx, file_name)

def encode_sequences(orig_sequence, frame_indices):
    
    encoder.eval()
    
    latent_vectors = []
    
    seq_excerpt_count = len(frame_indices)

    for excerpt_index in range(seq_excerpt_count):
        excerpt_start_frame = frame_indices[excerpt_index]
        excerpt_end_frame = excerpt_start_frame + sequence_length

        excerpt = orig_sequence[excerpt_start_frame:excerpt_end_frame]
        excerpt = np.expand_dims(excerpt, axis=0)
        excerpt = torch.from_numpy(excerpt).reshape(1, sequence_length, pose_dim).to(device)

        with torch.no_grad():

            encoder_output = encoder(excerpt)

            encoder_output_mu = encoder_output[0]
            encoder_output_std = encoder_output[1]
            mu = torch.tanh(encoder_output_mu)
            std = torch.abs(torch.tanh(encoder_output_std)) + 0.00001
            
            latent_vector = reparameterize(mu, std)
            
        latent_vector = torch.squeeze(latent_vector)
        latent_vector = latent_vector.detach().cpu().numpy()

        latent_vectors.append(latent_vector)
        
    encoder.train()
        
    return latent_vectors

def decode_sequence_encodings(sequence_encodings, seq_overlap, base_pose):
    
    decoder.eval()
    
    seq_env = np.hanning(sequence_length)
    seq_excerpt_count = len(sequence_encodings)
    gen_seq_length = (seq_excerpt_count - 1) * seq_overlap + sequence_length
    gen_sequence = np.full(shape=(gen_seq_length, joint_count, joint_dim), fill_value=base_pose)
    

    for excerpt_index in range(len(sequence_encodings)):
        latent_vector = sequence_encodings[excerpt_index]
        latent_vector = np.expand_dims(latent_vector, axis=0)
        latent_vector = torch.from_numpy(latent_vector).to(device)
        
        with torch.no_grad():
            excerpt_dec = decoder(latent_vector)
        
        excerpt_dec = torch.squeeze(excerpt_dec)
        excerpt_dec = excerpt_dec.detach().cpu().numpy()
        excerpt_dec = excerpt_dec.reshape((-1, joint_count, joint_dim))
        
        gen_frame = excerpt_index * seq_overlap
        
        # convert sequences from r6d to quat
        current_seq_r6d = gen_sequence[gen_frame:gen_frame + sequence_length, ...]
        target_seq_r6d = excerpt_dec
        
        #print("current_seq_r6d s ", current_seq_r6d.shape)
        #print("target_seq_r6d s ", target_seq_r6d.shape)
        
        current_seq_quat = repr6d_torch.repr6d2quat( torch.from_numpy(current_seq_r6d) )
        target_seq_quat = repr6d_torch.repr6d2quat( torch.from_numpy(target_seq_r6d) )
        
        #print("current_seq_quat s ", current_seq_quat.shape)
        #print("target_seq_quat s ", target_seq_quat.shape)

        # enforce consistent quaternions in each sequence and across the two sequences
        current_seq_quat, target_seq_quat = consistent_quaternion_seq(current_seq_quat, target_seq_quat)
        
        for si in range(sequence_length):
            for ji in range(joint_count): 

                current_quat = current_seq_quat[si, ji, :]
                target_quat = target_seq_quat[si, ji, :]

                quat_mix = seq_env[si]
                
                #print("current_quat s ", current_quat.shape)
                #print("target_quat s ", target_quat.shape)
                
                mix_quat = safe_slerp(current_quat, target_quat, quat_mix )
                #mix_quat = slerp(current_quat, target_quat, quat_mix )
                
                #print("mix_quat s ", mix_quat.shape)
                
                mix_r6d = repr6d_torch.quat2repr6d(torch.from_numpy(mix_quat.reshape(-1, 4))).numpy()[0]
                
                gen_sequence[gen_frame + si, ji, :] = mix_r6d
        
    """
    gen_sequence = gen_sequence.reshape((-1, 4))
    gen_sequence = gen_sequence / np.linalg.norm(gen_sequence, ord=2, axis=1, keepdims=True)
    gen_sequence = gen_sequence.reshape((gen_seq_length, joint_count, 4))
    gen_sequence = qfix(gen_sequence)
    """
    
    decoder.train()
    
    return gen_sequence
    
def create_2d_latent_space_representation(sequence_excerpts):

    encodings = []
    
    excerpt_count = sequence_excerpts.shape[0]
    
    for eI in range(0, excerpt_count, batch_size):
        
        excerpt_batch = sequence_excerpts[eI:eI+batch_size]
        
        #print("excerpt_batch s ", excerpt_batch.shape)
        
        excerpt_batch = torch.from_numpy(excerpt_batch).to(device)
        
        encoder_output = encoder(excerpt_batch)

        encoder_output_mu = encoder_output[0]
        encoder_output_std = encoder_output[1]
        mu = torch.tanh(encoder_output_mu)
        std = torch.abs(torch.tanh(encoder_output_std)) + 0.00001
        
        encoding_batch = reparameterize(mu, std)
        
        #print("encoding_batch s ", encoding_batch.shape)
        
        encoding_batch = encoding_batch.detach().cpu()

        encodings.append(encoding_batch)
        
    encodings = torch.cat(encodings, dim=0)
    
    #print("encodings s ", encodings.shape)
    
    encodings = encodings.numpy()

    # use TSNE for dimensionality reduction
    tsne = TSNE(n_components=2, n_iter=5000, verbose=1)    
    Z_tsne = tsne.fit_transform(encodings)
    
    return Z_tsne

def create_2d_latent_space_image(Z_tsne, highlight_excerpt_ranges, file_name):
    
    Z_tsne_x = Z_tsne[:,0]
    Z_tsne_y = Z_tsne[:,1]

    plot_colors = ["green", "red", "blue", "magenta", "orange"]
    plt.figure()
    fig, ax = plt.subplots()
    ax.plot(Z_tsne_x, Z_tsne_y, '-', c="grey",linewidth=0.2)
    ax.scatter(Z_tsne_x, Z_tsne_y, s=0.1, c="grey", alpha=0.5)
    
    for hI, hR in enumerate(highlight_excerpt_ranges):
        ax.plot(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], '-', c=plot_colors[hI],linewidth=0.6)
        ax.scatter(Z_tsne_x[hR[0]:hR[1]], Z_tsne_y[hR[0]:hR[1]], s=0.8, c=plot_colors[hI], alpha=0.5)
        
        ax.set_xlabel('$c_1$')
        ax.set_ylabel('$c_2$')

    fig.savefig(file_name, dpi=300)
    plt.close()
    
# create latent space plot

Z_tsne = create_2d_latent_space_representation(pose_sequence_excerpts)
create_2d_latent_space_image(Z_tsne, [], "latent_space_plot_epoch_{}.png".format(epochs))

# create original sequence

orig_sequence = all_mocap_data[0]["motion"]["rot_6d"].astype(np.float32)

seq_start = 1000
seq_length = 2000

export_sequence_anim(orig_sequence[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.gif".format(seq_start, seq_length))
export_sequence_fbx(orig_sequence[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.fbx".format(seq_start, seq_length))

# recontruct original sequence

seq_start = 1000
seq_length = 1000
seq_overlap = 16 # 2 for 8, 32 for 128
base_pose = np.reshape(orig_sequence[0], (joint_count, joint_dim))

seq_indices = [ frame_index for frame_index in range(seq_start, seq_start + seq_length, seq_overlap)]

seq_encodings = encode_sequences(orig_sequence, seq_indices)
gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, "results/anims/rec_sequences_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_fbx(gen_sequence, "results/anims/rec_sequences_epoch_{}_seq_start_{}_length_{}.fbx".format(epochs, seq_start, seq_length))



# random walk in latent space
seq_start = 1000
seq_length = 1000

seq_indices = [seq_start]

seq_encodings = encode_sequences(orig_sequence, seq_indices)

for index in range(0, seq_length // seq_overlap):
    random_step = np.random.random((latent_dim)).astype(np.float32) * 2.0
    seq_encodings.append(seq_encodings[index] + random_step)
    
gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, "results/anims/seq_randwalk_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_fbx(gen_sequence, "results/anims/seq_randwalk_epoch_{}_seq_start_{}_length_{}.fbx".format(epochs, seq_start, seq_length))

# sequence offset following

seq_start = 1000
seq_length = 1000
    
seq_indices = [ seq_index for seq_index in range(seq_start, seq_start + seq_length, seq_overlap)]

seq_encodings = encode_sequences(orig_sequence, seq_indices)

offset_seq_encodings = []

for index in range(len(seq_encodings)):
    sin_value = np.sin(index / (len(seq_encodings) - 1) * np.pi * 4.0)
    offset = np.ones(shape=(latent_dim), dtype=np.float32) * sin_value * 4.0
    offset_seq_encoding = seq_encodings[index] + offset
    offset_seq_encodings.append(offset_seq_encoding)
    
gen_sequence = decode_sequence_encodings(offset_seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, "results/anims/seq_offset_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_fbx(gen_sequence, "results/anims/seq_offset_epoch_{}_seq_start_{}_length_{}.fbx".format(epochs, seq_start, seq_length))



# interpolate two original sequences

seq1_start = 1000
seq2_start = 2000
seq_length = 1000

seq1_indices = [ seq_index for seq_index in range(seq1_start, seq1_start + seq_length, seq_overlap)]
seq2_indices = [ seq_index for seq_index in range(seq2_start, seq2_start + seq_length, seq_overlap)]

seq1_encodings = encode_sequences(orig_sequence, seq1_indices)
seq2_encodings = encode_sequences(orig_sequence, seq2_indices)

mix_encodings = []

for index in range(len(seq1_encodings)):
    mix_factor = index / (len(seq1_indices) - 1)
    mix_encoding = seq1_encodings[index] * (1.0 - mix_factor) + seq2_encodings[index] * mix_factor
    mix_encodings.append(mix_encoding)

gen_sequence = decode_sequence_encodings(mix_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, "results/anims/seq_mix_epoch_{}_seq1_start_{}_seq2_start_{}_length_{}.gif".format(epochs, seq1_start, seq2_start, seq_length))
export_sequence_fbx(gen_sequence, "results/anims/seq_mix_epoch_{}_seq1_start_{}_seq2_start_{}_length_{}.fbx".format(epochs, seq1_start, seq2_start, seq_length))

