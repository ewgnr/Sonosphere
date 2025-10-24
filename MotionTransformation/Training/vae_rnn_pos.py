"""
A variational autoencoder for motion capture data of a solo dancer
This version is meant for motion capture data that consists of joint positions only and typically stems from a pose estimation system

"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from collections import OrderedDict

import os, sys, time, subprocess
import numpy as np
import json
import pickle

from common import utils
from common import mocap_tools as mocap
from common.quaternion import qmul, qrot, qnormalize_np, slerp, qfix
from common.pose_renderer import PoseRenderer

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

# Example: MMPose 2D-Pose Estimation Recording
mocap_config_file = "data/configs/COCO_config.json" 
mocap_file_path = "data/mocap"
mocap_files = ["Stocos_Pose2D_BlumenBaile.pkl"]
mocap_sensor_ids = ["/mocap/0/joint/pos_world", "/mocap/0/joint/visibility"]
mocap_root_joint_name = "Left_Hip"
mocap_fps = 30
mocap_joint_dim = 2

"""
# Example: MMPose 2D-Pose Estimation Recording
mocap_config_file = "data/configs/COCO_config.json" 
mocap_file_path = "../../../Data/Mocap/Pose2D/Stocos/Solos"
mocap_files = ["Stocos_Pose2D_BlumenBaile.pkl"]
mocap_sensor_ids = ["/mocap/0/joint/pos_world", "/mocap/0/joint/visibility"]
mocap_root_joint_name = "Left_Hip"
mocap_fps = 30
mocap_joint_dim = 2
"""

"""
# Example: MMPose 3D-Pose Estimation Recording
mocap_config_file = "data/configs/Human36M_config.json" 
mocap_file_path = "../../../Data/Mocap/Pose3D/Stocos/Solos"
mocap_files = ["Stocos_Pose3D_BlumenBaile.pkl"]
mocap_sensor_ids = ["/mocap/0/joint/pos_world", "/mocap/0/joint/scores"]
mocap_root_joint_name = "Bottom_Torso"
mocap_fps = 30
mocap_joint_dim = 3
"""

"""
Model Settings
"""

latent_dim = 32
sequence_length = 64
ae_rnn_layer_count = 2
ae_rnn_layer_size = 512
ae_dense_layer_sizes = [ 512 ]

save_weights = True

# load model weights
load_weights = False
encoder_weights_file = "results_xsens_64/weights/encoder_weights_epoch_600"
decoder_weights_file = "results_xsens_64/weights/decoder_weights_epoch_600"

"""
Training Settings
"""

sequence_offset = 2 # when creating sequence excerpts, each excerpt is offset from the previous one by this value
batch_size = 16
test_percentage  = 0.2
ae_learning_rate = 1e-4
ae_pos_loss_scale = 1.0
ae_kld_loss_scale = 0.0 # will be calculated
kld_scale_cycle_duration = 100
kld_scale_min_const_duration = 20
kld_scale_max_const_duration = 20
min_kld_scale = 0.0
max_kld_scale = 0.1

epochs = 200
model_save_interval = 50
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
Load Mocap Data
"""

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
    
    joint_pos = motion_data["/mocap/0/joint/pos_world"]
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

pose_sequence_excerpts = []
vis_sequence_excerpts = []

for motion_data in all_motion_data:
    
    pose_sequence = motion_data["/mocap/0/joint/pos_root_zero"]
    pose_sequence = np.reshape(pose_sequence, (-1, pose_dim))
    
    #print("pose_sequence s ", pose_sequence.shape)
    
    if mocap_joint_dim == 2:
        vis_sequence = motion_data["/mocap/0/joint/visibility"]
    else:
        vis_sequence = motion_data["/mocap/0/joint/scores"]
    
    vis_sequence = np.reshape(vis_sequence, (-1, joint_count))
    
    #print("vis_sequence s ", vis_sequence.shape)

    frame_range_start = 0
    frame_range_end = pose_sequence.shape[0]
    
    for seq_excerpt_start in np.arange(frame_range_start, frame_range_end - sequence_length, sequence_offset):
        #print("valid: start ", frame_range_start, " end ", frame_range_end, " exc: start ", seq_excerpt_start, " end ", (seq_excerpt_start + sequence_length) )
        pose_sequence_excerpt =  pose_sequence[seq_excerpt_start:seq_excerpt_start + sequence_length]
        pose_sequence_excerpts.append(pose_sequence_excerpt)
        
        #print("pose_sequence_excerpt s ", pose_sequence_excerpt.shape)
        
        vis_sequence_excerpt = vis_sequence[seq_excerpt_start:seq_excerpt_start + sequence_length]
        
        #print("vis_sequence_excerpt s ", vis_sequence_excerpt.shape)
        
        vis_sequence_excerpts.append(vis_sequence_excerpt)

    
pose_sequence_excerpts = np.array(pose_sequence_excerpts, dtype=np.float32)
vis_sequence_excerpts = np.array(vis_sequence_excerpts, dtype=np.float32)

# create dataset

sequence_excerpts_count = pose_sequence_excerpts.shape[0]

class SequenceDataset(Dataset):
    def __init__(self, pose_sequence_excerpts, vis_sequence_excerpts):
        self.pose_sequence_excerpts = pose_sequence_excerpts
        self.vis_sequence_excerpts = vis_sequence_excerpts
    
    def __len__(self):
        return self.pose_sequence_excerpts.shape[0]
    
    def __getitem__(self, idx):
        return self.pose_sequence_excerpts[idx, ...], self.vis_sequence_excerpts[idx, ...]
        

full_dataset = SequenceDataset(pose_sequence_excerpts, vis_sequence_excerpts)
dataset_size = len(full_dataset)

pose_seq_item, vis_seq_item = full_dataset[0]
print("pose_seq_item s ", pose_seq_item.shape)
print("vis_seq_item s ", vis_seq_item.shape)

test_size = int(test_percentage * dataset_size)
train_size = dataset_size - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

pose_seq_batch, vis_seq_batch = next(iter(train_dataloader))

print("pose_seq_batch s ", pose_seq_batch.shape)
print("vis_seq_batch s ", vis_seq_batch.shape)

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
    
# Training

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
ae_scheduler = torch.optim.lr_scheduler.StepLR(ae_optimizer, step_size=50, gamma=0.33) # reduce the learning every 100 epochs by a factor of 10

mse_loss = nn.MSELoss()
cross_entropy = nn.BCELoss()

# pose mean and std

pose_mean = torch.tensor(pose_mean).reshape(1, 1, -1).to(device)
pose_std = torch.tensor(pose_std).reshape(1, 1, -1).to(device)

# joint loss weights

if len(joint_loss_weights) == 1:
    joint_loss_weights *= joint_count

joint_loss_weights = torch.tensor(joint_loss_weights, dtype=torch.float32)
joint_loss_weights = joint_loss_weights.reshape(1, 1, -1).to(device)

# KL Divergence

def variational_loss(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #see also: see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    #https://arxiv.org/abs/1312.6114
    vl=-0.5*torch.mean(1+ 2*torch.log(std)-mu.pow(2) -(std.pow(2)))
    return vl
   
def variational_loss2(mu, std):
    #returns the varialtional loss from arguments mean and standard deviation std
    #alternative: mean squared distance from ideal mu=0 and std=1:
    vl=torch.mean(mu.pow(2)+(1-std).pow(2))
    return vl

def reparameterize(mu, std):
    z = mu + std*torch.randn_like(std)
    return z

def ae_pos_loss(y, yhat, vis):
    # y and yhat shapes: batch_size, seq_length, pose_dim
    
    #print("ae_pos_loss")
    #print("y s ", y.shape)
    #print("yhat s ", yhat.shape)
    #print("vis s ", vis.shape)
    
    # reshape into batch_size x sequence_length x joint_count x joint_dim
    _y = y.reshape(y.shape[0], y.shape[1], joint_count, joint_dim )
    _yhat = yhat.reshape(y.shape[0], y.shape[1], joint_count, joint_dim )
    
    #print("_y s ", _y.shape)
    #print("_yhat s ", _yhat.shape)

    _pos_diff = torch.norm((_y - _yhat), dim=3)
    
    #print("_pos_diff s ", _pos_diff.shape)
    
    #print("_pos_diff s ", _pos_diff.shape)
    #print("joint_loss_weights s ", joint_loss_weights.shape)
    #print("_vis s ", _vis.shape)
    
    _pos_diff_weighted = _pos_diff * joint_loss_weights * vis

    #_pos_diff_weighted = _pos_diff * joint_loss_weights.reshape(1, joint_count) * _vis
    
    _loss = torch.mean(_pos_diff_weighted)

    return _loss

# autoencoder loss function
def ae_loss(y, yhat, vis, mu, std):
    # function parameters
    # y: encoder input
    # yhat: decoder output (i.e. reconstructed encoder input)
    # disc_fake_output: discriminator output for encoder generated prior
    
    _pos_loss = ae_pos_loss(y, yhat, vis)

    # kld loss
    _ae_kld_loss = variational_loss(mu, std)
    
    _total_loss = 0.0
    _total_loss += _pos_loss * ae_pos_loss_scale
    _total_loss += _ae_kld_loss * ae_kld_loss_scale
    
    return _total_loss, _pos_loss, _ae_kld_loss

def ae_train_step(target_poses, joint_vis):
    
    # normalise target poses
    target_poses_norm = (target_poses - pose_mean) / pose_std
    target_poses_norm = torch.nan_to_num(target_poses_norm)
 
    # let autoencoder preproduce target_poses (decoder output) and also return encoder output
    encoder_output = encoder(target_poses_norm)

    encoder_output_mu = encoder_output[0]
    encoder_output_std = encoder_output[1]
    mu = torch.tanh(encoder_output_mu)
    std = torch.abs(torch.tanh(encoder_output_std)) + 0.00001
    decoder_input = reparameterize(mu, std)
    
    pred_poses_norm = decoder(decoder_input)
    
    _ae_loss, _ae_pos_loss, _ae_kld_loss = ae_loss(target_poses_norm, pred_poses_norm, joint_vis, mu, std) 

    #print("_ae_pos_loss ", _ae_pos_loss)
    
    # Backpropagation
    ae_optimizer.zero_grad()
    _ae_loss.backward()
    
    #torch.nn.utils.clip_grad_norm(encoder.parameters(), 0.01)
    #torch.nn.utils.clip_grad_norm(decoder.parameters(), 0.01)

    ae_optimizer.step()
    
    return _ae_loss, _ae_pos_loss, _ae_kld_loss

def ae_test_step(target_poses, joint_vis):
    
    with torch.no_grad():
        # normalise target poses
        target_poses_norm = (target_poses - pose_mean) / pose_std
        target_poses_norm = torch.nan_to_num(target_poses_norm)
     
        # let autoencoder preproduce target_poses (decoder output) and also return encoder output
        encoder_output = encoder(target_poses_norm)
    
        encoder_output_mu = encoder_output[0]
        encoder_output_std = encoder_output[1]
        mu = torch.tanh(encoder_output_mu)
        std = torch.abs(torch.tanh(encoder_output_std)) + 0.00001
        decoder_input = reparameterize(mu, std)
        
        pred_poses_norm = decoder(decoder_input)
        
        _ae_loss, _ae_pos_loss, _ae_kld_loss = ae_loss(target_poses_norm, pred_poses_norm, joint_vis, mu, std) 
    
        #print("_ae_pos_loss ", _ae_pos_loss)
    
    return _ae_loss, _ae_pos_loss, _ae_kld_loss


def train(train_dataloader, test_dataloader, epochs):
    
    global ae_kld_loss_scale
    
    loss_history = {}
    loss_history["ae train"] = []
    loss_history["ae test"] = []
    loss_history["ae pos"] = []
    loss_history["ae kld"] = []
    
    for epoch in range(epochs):

        start = time.time()
        
        ae_kld_loss_scale = kld_scales[epoch]
        
        #print("ae_kld_loss_scale ", ae_kld_loss_scale)
        
        ae_train_loss_per_epoch = []
        ae_pos_loss_per_epoch = []
        ae_prior_loss_per_epoch = []
        ae_kld_loss_per_epoch = []
        
        for train_batch_pose, train_batch_vis in train_dataloader:
            train_batch_pose = train_batch_pose.to(device)
            train_batch_vis = train_batch_vis.to(device)
            
            _ae_loss, _ae_pos_loss, _ae_kld_loss = ae_train_step(train_batch_pose, train_batch_vis)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            _ae_pos_loss = _ae_pos_loss.detach().cpu().numpy()
            _ae_kld_loss = _ae_kld_loss.detach().cpu().numpy()
            
            #print("_ae_prior_loss ", _ae_prior_loss)
            
            ae_train_loss_per_epoch.append(_ae_loss)
            ae_pos_loss_per_epoch.append(_ae_pos_loss)
            ae_kld_loss_per_epoch.append(_ae_kld_loss)

        ae_train_loss_per_epoch = np.mean(np.array(ae_train_loss_per_epoch))
        ae_pos_loss_per_epoch = np.mean(np.array(ae_pos_loss_per_epoch))
        ae_kld_loss_per_epoch = np.mean(np.array(ae_kld_loss_per_epoch))

        ae_test_loss_per_epoch = []
        
        for test_batch_pose, test_batch_vis in test_dataloader:
            test_batch_pose = test_batch_pose.to(device)
            test_batch_vis = test_batch_vis.to(device)
            
            _ae_loss, _, _ = ae_test_step(test_batch_pose, test_batch_vis)
            
            _ae_loss = _ae_loss.detach().cpu().numpy()
            ae_test_loss_per_epoch.append(_ae_loss)
        
        ae_test_loss_per_epoch = np.mean(np.array(ae_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(encoder.state_dict(), "results/weights/encoder_weights_epoch_{}".format(epoch))
            torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epoch))
        
        loss_history["ae train"].append(ae_train_loss_per_epoch)
        loss_history["ae test"].append(ae_test_loss_per_epoch)
        loss_history["ae pos"].append(ae_pos_loss_per_epoch)
        loss_history["ae kld"].append(ae_kld_loss_per_epoch)
        
        print ('epoch {} : ae train: {:01.4f} ae test: {:01.4f} pos {:01.4f} kld {:01.4f} time {:01.2f}'.format(epoch + 1, ae_train_loss_per_epoch, ae_test_loss_per_epoch, ae_pos_loss_per_epoch, ae_kld_loss_per_epoch, time.time()-start))
    
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

def encode_sequences(pose_sequence, frame_indices):
    
    pose_count = pose_sequence.shape[0]
    pose_sequence = torch.from_numpy(pose_sequence).to(torch.float32).to(device)
    pose_sequence = pose_sequence.reshape(1, pose_count, pose_dim)
    pose_sequence_norm = (pose_sequence - pose_mean) / pose_std
    pose_sequence_norm = torch.nan_to_num(pose_sequence_norm)
    
    encoder.eval()
    
    latent_vectors = []
    
    seq_excerpt_count = len(frame_indices)

    for excerpt_index in range(seq_excerpt_count):
        excerpt_start_frame = frame_indices[excerpt_index]
        excerpt_end_frame = excerpt_start_frame + sequence_length
        excerpt_norm = pose_sequence_norm[:, excerpt_start_frame:excerpt_end_frame, :]

        with torch.no_grad():

            encoder_output = encoder(excerpt_norm)

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
    #seq_env = np.concatenate([np.linspace(0.0, 1.0, seq_overlap), np.ones(sequence_length - seq_overlap)])

    seq_excerpt_count = len(sequence_encodings)
    gen_seq_length = (seq_excerpt_count - 1) * seq_overlap + sequence_length

    gen_sequence = np.full(shape=(gen_seq_length, joint_count, joint_dim), fill_value=base_pose)
    
    for excerpt_index in range(len(sequence_encodings)):
        
        #print("excerpt_index ", excerpt_index)
        
        latent_vector = sequence_encodings[excerpt_index]
        latent_vector = np.expand_dims(latent_vector, axis=0)
        latent_vector = torch.from_numpy(latent_vector).to(device)
        
        #print("latent_vector s ", latent_vector.shape)
        
        with torch.no_grad():
            excerpt_dec_norm = decoder(latent_vector)
        
        excerpt_dec = excerpt_dec_norm * pose_std + pose_mean
        excerpt_dec = excerpt_dec.detach().cpu()
        excerpt_dec = np.reshape(excerpt_dec, (sequence_length, joint_count, joint_dim))
        
        #print("excerpt_dec s ", excerpt_dec.shape)

        gen_frame = excerpt_index * seq_overlap
        
        for si in range(sequence_length):
            
            mix = seq_env[si]
            
            current_pose = gen_sequence[gen_frame + si]
            target_pose = excerpt_dec[si]
            
            #print("si ", si, " mix ", mix, "current_pose s ", current_pose.shape, " target_pose s ", target_pose.shape)
            
            mix_pose = np.add(current_pose * (1.0 - mix) , target_pose * mix)

            gen_sequence[gen_frame + si] = mix_pose
        
    gen_sequence = np.reshape(gen_sequence, (gen_seq_length, joint_count, joint_dim))

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

orig_sequence = all_motion_data[0]["/mocap/0/joint/pos_root_zero"].astype(np.float32)

seq_start = 1000
seq_length = 1000

export_sequence_anim(orig_sequence[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.gif".format(seq_start, seq_length))
export_sequence_pkl(orig_sequence[seq_start:seq_start+seq_length], "results/anims/orig_sequence_seq_start_{}_length_{}.pkl".format(seq_start, seq_length))

# recontruct original sequence

seq_start = 1000
seq_length = 1000
seq_overlap = 16 # 2 for 8, 32 for 128
base_pose = np.reshape(orig_sequence[0], (joint_count, joint_dim))

seq_indices = [ frame_index for frame_index in range(seq_start, seq_start + seq_length, seq_overlap)]

seq_encodings = encode_sequences(orig_sequence, seq_indices)
gen_sequence = decode_sequence_encodings(seq_encodings, seq_overlap, base_pose)
export_sequence_anim(gen_sequence, "results/anims/rec_sequences_epoch_{}_seq_start_{}_length_{}.gif".format(epochs, seq_start, seq_length))
export_sequence_pkl(gen_sequence, "results/anims/rec_sequences_epoch_{}_seq_start_{}_length_{}.pkl".format(epochs, seq_start, seq_length))


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
export_sequence_pkl(gen_sequence, "results/anims/seq_randwalk_epoch_{}_seq_start_{}_length_{}.pkl".format(epochs, seq_start, seq_length))

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
export_sequence_pkl(gen_sequence, "results/anims/seq_offset_epoch_{}_seq_start_{}_length_{}.pkl".format(epochs, seq_start, seq_length))


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
export_sequence_pkl(gen_sequence, "results/anims/seq_mix_epoch_{}_seq1_start_{}_seq2_start_{}_length_{}.pkl".format(epochs, seq1_start, seq2_start, seq_length))

