"""
same as lstm_ntf_mel_v2.py
but for reading multiple audio files
"""

import numpy as np
import time
import csv
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchaudio
import torchaudio.transforms as transforms
from collections import OrderedDict

from vocos import Vocos

"""
Compute Device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

"""
Audio Settings
"""

audio_files = []
audio_files += ["E:/Data/audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"]
audio_sample_rate = 48000

audio_file_excerpts = []
audio_file_excerpts += [[14 * audio_sample_rate, 314 * audio_sample_rate]]


"""
Model Settings
"""

rnn_layer_dim = 512
rnn_layer_count = 2

save_weights = True
load_weights = False
rnn_weights_file = "results/weights/rnn_weights_epoch_200"

"""
Training settings
"""

batch_size = 32
test_percentage = 0.1

seq_input_length = 64
seq_output_length = 10 # this is only used for non-teacher forcing scenarios

learning_rate = 1e-4
teacher_forcing_prob = 0.0
model_save_interval = 10

epochs = 200
save_history = True

"""
Load Audio Data and Calulate Mel Spectra
"""

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1")

all_audio_features = []

for audio_file_index in range(len(audio_files)):
    
    print("audio file ", audio_files[audio_file_index])
    
    audio_file = audio_files[audio_file_index]
    audio_file_excerpt = audio_file_excerpts[audio_file_index]
    
    waveform_range_start = audio_file_excerpt[0]
    waveform_range_end = audio_file_excerpt[1]
    
    # load audio file
    waveform_data, _ = torchaudio.load(audio_file_path + audio_file)
    
    # audio excerpt
    waveform_data = waveform_data[:, waveform_range_start:waveform_range_end]
    
    print("waveform_data s ", waveform_data.shape)
    
    # audio features
    audio_features = vocos.feature_extractor(waveform_data)
    
    print("audio_features s ", audio_features.shape)
    
    audio_features = audio_features.squeeze(0)
    audio_features = torch.permute(audio_features, (1, 0))
    
    print("audio_features 2 s ", audio_features.shape)
    
    all_audio_features.append(audio_features)
    
"""
Create Dataset
"""

X = []
y = []

audio_features_dim = all_audio_features[0].shape[-1]

audio_features_dim

for audio_features in all_audio_features:
    
    total_sequence_length = audio_features.shape[0]
    
    for pI in range(total_sequence_length - seq_input_length - seq_output_length - 1):
        X_sample = audio_features[pI:pI+seq_input_length]
        X.append(X_sample)
        
        y_sample = audio_features[pI+seq_input_length:pI+seq_input_length+seq_output_length]
        y.append(y_sample)

X = np.array(X)
y = np.array(y)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, ...], self.y[idx, ...]


full_dataset = SequenceDataset(X, y)

test_percentage = 0.1
test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
Recurrent Model
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

rnn = Reccurent(audio_features_dim, rnn_layer_dim, audio_features_dim, rnn_layer_count).to(device)
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

rec_loss = nn.MSELoss()

def loss(y, yhat):
    _rec_loss = rec_loss(yhat, y)

    return _rec_loss

def train_step(input_features, target_features, teacher_forcing):
    
    rnn.train()

    #print("ar_train_step")    
    #print("teacher_forcing ", teacher_forcing)
    #print("pose_sequences s ", pose_sequences.shape)
    #print("target_poses s ", target_poses.shape)

    #_input_poses = pose_sequences.detach().clone()    
    _input_features = input_features  
    output_features_length = target_features.shape[1]
    
    #print("output_features_length ", output_features_length)
    
    _pred_features_for_loss = []
    _target_features_for_loss = []
    
    for o_i in range(1, output_features_length):
        
        #print("_input_features s ", _input_features.shape)
        
        _pred_features = rnn(_input_features)
        _pred_features = torch.unsqueeze(_pred_features, axis=1)
        
        #print("_pred_features s ", _pred_features.shape)
        
        _target_features = target_features[:,o_i,:].detach().clone()
        _target_features = torch.unsqueeze(_target_features, axis=1)

        #print("_target_features s ", _target_features.shape)
        
        _pred_features_for_loss.append(_pred_features)
        _target_features_for_loss.append(_target_features)
        
        # shift input feature seqeunce one feature to the right
        # remove feature from beginning input feature sequence
        # detach necessary to avoid error concerning running backprob a second time
        _input_features = _input_features[:, 1:, :].detach().clone()
        _target_features = _target_features.detach().clone()
        _pred_features = _pred_features.detach().clone()
        
        # add predicted or target feature to end of input feature sequence
        if teacher_forcing == True:
            _input_features = torch.concat((_input_features, _target_features), axis=1)
        else:
            _input_features = torch.cat((_input_features, _pred_features), axis=1)
            
        #print("_input_features s ", _input_features.shape)

        
    _pred_features_for_loss = torch.cat(_pred_features_for_loss, dim=1)
    _target_features_for_loss = torch.cat(_target_features_for_loss, dim=1)
    
    #print("_pred_features_for_loss 2 s ", _pred_features_for_loss.shape)
    #print("_target_features_for_loss 2 s ", _target_features_for_loss.shape)
    
    _loss = loss(_target_features_for_loss, _pred_features_for_loss) 
    
    # Backpropagation
    optimizer.zero_grad()
    _loss.backward()
    optimizer.step()
    
    #print("_ar_loss_total mean s ", _ar_loss_total.shape)
    
    #return _ar_loss, _ar_norm_loss, _ar_quat_loss
    
    return _loss

def test_step(input_features, target_features, teacher_forcing):
    
    rnn.eval()

    #print("ar_train_step")    
    #print("teacher_forcing ", teacher_forcing)
    #print("pose_sequences s ", pose_sequences.shape)
    #print("target_poses s ", target_poses.shape)

    #_input_poses = pose_sequences.detach().clone()    
    _input_features = input_features  
    output_features_length = target_features.shape[1]
    
    #print("output_features_length ", output_features_length)
    
    _pred_features_for_loss = []
    _target_features_for_loss = []
    
    with torch.no_grad():
    
        for o_i in range(1, output_features_length):
            
            #print("_input_features s ", _input_features.shape)
            
            _pred_features = rnn(_input_features)
            _pred_features = torch.unsqueeze(_pred_features, axis=1)
            
            #print("_pred_features s ", _pred_features.shape)
            
            _target_features = target_features[:,o_i,:].detach().clone()
            _target_features = torch.unsqueeze(_target_features, axis=1)
    
            #print("_target_features s ", _target_features.shape)
            
            _pred_features_for_loss.append(_pred_features)
            _target_features_for_loss.append(_target_features)
            
            # shift input feature seqeunce one feature to the right
            # remove feature from beginning input feature sequence
            # detach necessary to avoid error concerning running backprob a second time
            _input_features = _input_features[:, 1:, :].detach().clone()
            _target_features = _target_features.detach().clone()
            _pred_features = _pred_features.detach().clone()
            
            # add predicted or target feature to end of input feature sequence
            if teacher_forcing == True:
                _input_features = torch.concat((_input_features, _target_features), axis=1)
            else:
                _input_features = torch.cat((_input_features, _pred_features), axis=1)
                
            #print("_input_features s ", _input_features.shape)

        
    _pred_features_for_loss = torch.cat(_pred_features_for_loss, dim=1)
    _target_features_for_loss = torch.cat(_target_features_for_loss, dim=1)
    
    #print("_pred_features_for_loss 2 s ", _pred_features_for_loss.shape)
    #print("_target_features_for_loss 2 s ", _target_features_for_loss.shape)
    
    _loss = loss(_target_features_for_loss, _pred_features_for_loss) 
    
    #print("_ar_loss_total mean s ", _ar_loss_total.shape)
    
    #return _ar_loss, _ar_norm_loss, _ar_quat_loss
    
    rnn.eval()
    
    return _loss

def train(train_dataloader, test_dataloader, epochs):
    
    loss_history = {}
    loss_history["train"] = []
    loss_history["test"] = []

    for epoch in range(epochs):
        start = time.time()
        
        _train_loss_per_epoch = []

        for train_batch in train_dataloader:
            input_feature_sequences = train_batch[0].to(device)
            target_features = train_batch[1].to(device)
            
            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            
            _loss = train_step(input_feature_sequences, target_features, use_teacher_forcing)
            
            _loss = _loss.detach().cpu().numpy()
            
            _train_loss_per_epoch.append(_loss)

        _train_loss_per_epoch = np.mean(np.array(_train_loss_per_epoch))

        _test_loss_per_epoch = []
        
        for test_batch in test_dataloader:
            input_feature_sequences = train_batch[0].to(device)
            target_features = train_batch[1].to(device)
            
            use_teacher_forcing = np.random.uniform() < teacher_forcing_prob
            
            _loss = test_step(input_feature_sequences, target_features, use_teacher_forcing)
            
            _loss = _loss.detach().cpu().numpy()
            
            _test_loss_per_epoch.append(_loss)
        
        _test_loss_per_epoch = np.mean(np.array(_test_loss_per_epoch))
        
        if epoch % model_save_interval == 0 and save_weights == True:
            torch.save(rnn.state_dict(), "results/weights/rnn_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, time.time()-start))
    
    return loss_history

# fit model
loss_history = train(train_loader, test_loader, epochs)

# save history

def save_loss_as_image(loss_history, image_file_name):
    keys = list(loss_history.keys())
    epochs = len(loss_history[keys[0]])
    
    for key in keys:
        plt.plot(range(epochs), loss_history[key], label=key)
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_file_name)
    plt.show()

def save_loss_as_csv(loss_history, csv_file_name):
    with open(csv_file_name, 'w') as csv_file:
        csv_columns = list(loss_history.keys())
        csv_row_count = len(loss_history[csv_columns[0]])
        
        
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns, delimiter=',', lineterminator='\n')
        csv_writer.writeheader()
    
        for row in range(csv_row_count):
        
            csv_row = {}
        
            for key in loss_history.keys():
                csv_row[key] = loss_history[key][row]

            csv_writer.writerow(csv_row)


save_loss_as_csv(loss_history, "results/histories/rnn_history_{}.csv".format(epochs))
save_loss_as_image(loss_history, "results/histories/rnn_history_{}.png".format(epochs))

# save model weights
torch.save(rnn.state_dict(), "results/weights/rnn_weights_epoch_{}".format(epochs))

"""
Inference
"""

rnn.eval()

def export_orig_audio(waveform_data, start_time, end_time, file_name):
    
    start_time_samples = int(start_time * audio_sample_rate)
    end_time_samples = int(end_time * audio_sample_rate)
    
    torchaudio.save(file_name, waveform_data[:, start_time_samples:end_time_samples], audio_sample_rate)

def export_ref_audio(waveform_data, start_time, end_time, file_name):
    
    start_time_samples = int(start_time * audio_sample_rate)
    end_time_samples = int(end_time * audio_sample_rate)
    
    # audio features
    audio_features = vocos.feature_extractor(waveform_data[:, start_time_samples:end_time_samples])
    
    ref_audio = vocos.decode(audio_features)
    
    torchaudio.save(file_name, ref_audio.detach().cpu(), audio_sample_rate)
    

def export_pred_audio(waveform_data, start_time, end_time, file_name):
    
    start_time_samples = int(start_time * audio_sample_rate)
    end_time_samples = int(end_time * audio_sample_rate)
    
    # audio features
    audio_features = vocos.feature_extractor(waveform_data[:, start_time_samples:end_time_samples])
    
    #print("audio_features s ", audio_features.shape)
    
    audio_features = audio_features.squeeze(0)
    audio_features = torch.permute(audio_features, (1, 0))
    audio_feature_count = audio_features.shape[0]
    
    #print("audio_feature_count ", audio_feature_count)
    
    input_features = audio_features[:seq_input_length]
    input_features = input_features.unsqueeze(0)
    
    output_features_length = audio_feature_count - seq_input_length
    
    #print("output_features_length ", output_features_length)
    
    _input_features = input_features  
    pred_features = []
    
    with torch.no_grad():
    
        for o_i in range(1, output_features_length):
            
            _input_features = _input_features.to(device)
            
            #print("_input_features s ", _input_features.shape)
            
            _pred_features = rnn(_input_features)
            _pred_features = torch.unsqueeze(_pred_features, axis=1)

            _input_features = _input_features[:, 1:, :].detach().clone()
            _pred_features = _pred_features.detach().clone()
            
            pred_features.append(_pred_features.cpu())
            
            _input_features = torch.cat((_input_features, _pred_features), axis=1)
                
            #print("_input_features s ", _input_features.shape)
            
    pred_features = torch.cat(pred_features, axis=1)
    pred_features = torch.permute(pred_features, (0, 2, 1))
    pred_audio = vocos.decode(pred_features)
    
    torchaudio.save(file_name, pred_audio.detach().cpu(), audio_sample_rate)


audio_file = "E:/Data/audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
audio_start_time_sec = 10.0
audio_end_time_sec = 20.0

waveform_data, _ = torchaudio.load(audio_file)

export_orig_audio(waveform_data, audio_start_time_sec, audio_end_time_sec, "results/audio/orig_{}-{}.wav".format(audio_start_time_sec, audio_end_time_sec))

export_ref_audio(waveform_data, audio_start_time_sec, audio_end_time_sec, "results/audio/ref_{}-{}.wav".format(audio_start_time_sec, audio_end_time_sec))

export_pred_audio(waveform_data, audio_start_time_sec, audio_end_time_sec, "results/audio/pred_{}-{}_epoch_{}.wav".format(audio_start_time_sec, audio_end_time_sec, epochs))




