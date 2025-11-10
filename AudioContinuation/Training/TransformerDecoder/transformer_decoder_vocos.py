"""
Audio Continuation (Transformer Decoder Version)
"""

"""
Imports
"""

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

import simpleaudio as sa

import numpy as np
import math
import time
import csv
import os
import matplotlib.pyplot as plt

"""
Settings
"""

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

decoder_layer_count = 6
decoder_head_count = 8
decoder_embed_dim = 512
decoder_ff_dim = 2048
decoder_dropout = 0.1


save_weights = True
load_weights = True
decoder_weights_file = "results_transformer_decoder_vocos_Gutenberg_v2/weights/decoder_weights_epoch_200"

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
Load Audio Data and Calculate Audio Features
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
    waveform_data, _ = torchaudio.load(audio_file)
    
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

test_size = int(test_percentage * len(full_dataset))
train_size = len(full_dataset) - test_size

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

"""
Create Models
"""

"""
PositionalEncoding
"""

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

"""
Create TransformerDecoder
"""

class TransformerDecoder(nn.Module):

    # Constructor
    def __init__(
        self,
        audio_dim,
        embed_dim,
        num_heads,
        num_decoder_layers,
        ff_dim,
        dropout_p,
        pos_encoding_max_length
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.audio2embed = nn.Linear(audio_dim, embed_dim) # map audio data to embedding

        self.positional_encoder = PositionalEncoding(
            dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length
        )
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout_p, batch_first=True)
        #self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_decoder_layers)

        # build a decoder directly from TransformerDecoderLayer
        # rather than using the nn.TransformerDecoder module which requires also a Transformer Encoder
        self.decoder = self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout_p,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_decoder_layers)
        ])

        self.embed2audio = nn.Linear(embed_dim, audio_dim) # map embedding to audio data

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask
        
       
    def forward(self, audio_data):
        
        #print("forward")
        
        #print("audio_data s ", audio_data.shape)
        
        # dummy "memory" as zero (only self-attention is used)
        memory = torch.zeros(audio_data.size(0), audio_data.size(1), self.embed_dim, device=audio_data.device)

        #print("memory s ", memory.shape)

        # Lower triangular matrix for autoregressive masking
        tgt_mask = self.get_tgt_mask(audio_data.shape[1]).to(audio_data.device)

        #print("tgt_mask s ", tgt_mask.shape)

        audio_embedded = self.audio2embed(audio_data) * math.sqrt(self.embed_dim)
        
        #print("audio_embedded 1 s ", audio_embedded.shape)
        
        audio_embedded = self.positional_encoder(audio_embedded)
        
        #print("audio_embedded 2 s ", audio_embedded.shape)
        
        x = audio_embedded
        
        #print("x s ", x.shape)
        
        for layer in self.layers:
            
            #print("x in s ", x.shape)
            
            x = layer(x, memory, tgt_mask=tgt_mask)
            
            #print("x out s ", x.shape)

        decoder_out = x

        out = self.embed2audio(decoder_out)
        
        out = out[:, -1, :] # only last time step 
        
        return out

decoder = TransformerDecoder(audio_dim=audio_features_dim,
                          embed_dim=decoder_embed_dim, 
                          num_heads=decoder_head_count, 
                          num_decoder_layers=decoder_layer_count, 
                          ff_dim = decoder_ff_dim,
                          dropout_p=decoder_dropout,
                          pos_encoding_max_length=seq_input_length).to(device)

print(decoder)

if load_weights == True:
    decoder.load_state_dict(torch.load(decoder_weights_file))

# test transformer decoder
x_batch, _ = next(iter(train_loader))

decoder_input = x_batch.to(device)
decoder_output = decoder(decoder_input)

print("decoder_input s ", decoder_input.shape)
print("decoder_output s ", decoder_output.shape)

"""
Training
"""

optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.336) # reduce the learning every 20 epochs by a factor of 10

rec_loss = nn.MSELoss()

def loss(y, yhat):
    _rec_loss = rec_loss(yhat, y)

    return _rec_loss

def train_step(input_features, target_features, teacher_forcing):
    
    decoder.train()

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
        
        _pred_features = decoder(_input_features)
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
    
    decoder.eval()

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
            
            _pred_features = decoder(_input_features)
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
    
    decoder.eval()
    
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
            torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epoch))
        
        loss_history["train"].append(_train_loss_per_epoch)
        loss_history["test"].append(_test_loss_per_epoch)
        
        scheduler.step()
        
        print ('epoch {} : train: {:01.4f} test: {:01.4f} time {:01.2f}'.format(epoch + 1, _train_loss_per_epoch, _test_loss_per_epoch, time.time()-start))
    
    return loss_history

"""
Run Training
"""

loss_history = train(train_loader, test_loader, epochs)


"""
Save Training Results
"""

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
torch.save(decoder.state_dict(), "results/weights/decoder_weights_epoch_{}".format(epochs))

"""
Inference
"""

decoder.eval()

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
            
            _pred_features = decoder(_input_features)
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
audio_start_time_sec = 60.0
audio_end_time_sec = 100.0

waveform_data, _ = torchaudio.load(audio_file)

export_orig_audio(waveform_data, audio_start_time_sec, audio_end_time_sec, "results/audio/orig_{}-{}.wav".format(audio_start_time_sec, audio_end_time_sec))

export_ref_audio(waveform_data, audio_start_time_sec, audio_end_time_sec, "results/audio/ref_{}-{}.wav".format(audio_start_time_sec, audio_end_time_sec))

export_pred_audio(waveform_data, audio_start_time_sec, audio_end_time_sec, "results/audio/pred_{}-{}_epoch_{}.wav".format(audio_start_time_sec, audio_end_time_sec, epochs))




