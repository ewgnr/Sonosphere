"""
Audio Autoencoder (CNN Version)
"""

"""
Imports
"""

import os
import sys
import math
import numpy as np
import threading
import queue

import torch
from torch import nn
from collections import OrderedDict
import torchaudio
from torchaudio.functional import highpass_biquad

from vocos import Vocos

import sounddevice as sd

from pythonosc import dispatcher
from pythonosc import osc_server

"""
Settings
"""

"""
Device Settings
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device.upper()} device")

"""
Audio Settings
"""

audio_file_1 = "data/audio/Take3_RO_37-4-1_HQ_audio_crop_48khz.wav"
audio_file_2 = "data/audio/Take2_Hibr_II_HQ_audio_crop_48khz.wav"
audio_sample_rate = 48000
audio_channel_count = 1
audio_buffer_size = 2048
audio_output_device = 7 # windows: 7, macOs: 2
max_audio_queue_length = 32

# automatically calculated settings
gen_buffer_size = audio_buffer_size
window_size = gen_buffer_size
window_offset = window_size // 2
play_buffer_size = window_size * 16
playback_latency = 1.0

"""
Autoencoder Settings
"""

latent_dim = 32
ae_conv_channel_counts = [ 16, 32, 64, 128 ]
ae_conv_kernel_size = (5, 3)
ae_dense_layer_sizes = [ 512 ]
ae_encoder_weights_file = "data/results/weights/audio_vocos_cnn_encoder_weights_epoch_400"
ae_decoder_weights_file = "data/results/weights/audio_vocos_cnn_decoder_weights_epoch_400"

"""
OSC Control Settings
"""

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9005

"""
Load Audio
"""

assert os.path.exists(audio_file_1), f"Audio file 1 not found: {audio_file_1}"
assert os.path.exists(audio_file_2), f"Audio file 2 not found: {audio_file_2}"

audio_waveform_1, _ = torchaudio.load(audio_file_1)
audio_source_samples_1 = audio_waveform_1[0].to(device)
audio_source_frame_index_1 = 0

audio_waveform_2, _ = torchaudio.load(audio_file_2)
audio_source_samples_2 = audio_waveform_2[0].to(device)
audio_source_frame_index_2 = 0

hann_window = torch.from_numpy(np.hanning(window_size)).float().to(device) # Move to device once

"""
Create Models
"""

"""
Load Vocos Model
"""

vocos = Vocos.from_pretrained("kittn/vocos-mel-48khz-alpha1").to(device)
vocos.eval()
with torch.no_grad():
    dummy_features = vocos.feature_extractor(torch.rand(size=(1, gen_buffer_size), device=device))
    mel_count = dummy_features.shape[-1]
    mel_filters = dummy_features.shape[1]

"""
Create VAE Encoder
"""

class Encoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filter_count, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filter_count = mel_filter_count
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)

        padding = stride
        
        self.conv_layers.append(nn.Conv2d(1, conv_channel_counts[0], self.conv_kernel_size, stride=stride, padding=padding))
        self.conv_layers.append(nn.LeakyReLU(0.2))
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[0]))
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.Conv2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index]))

        self.flatten = nn.Flatten()
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        last_conv_layer_size_x = int(mel_filter_count // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))

        preflattened_size = [conv_channel_counts[-1], last_conv_layer_size_x, last_conv_layer_size_y]
        
        dense_layer_input_size = conv_channel_counts[-1] * last_conv_layer_size_x * last_conv_layer_size_y

        self.dense_layers.append(nn.Linear(dense_layer_input_size, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        # create final dense layers
        self.fc_mu = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)
        self.fc_std = nn.Linear(self.dense_layer_sizes[-1], self.latent_dim)


    def forward(self, x):
        for lI, layer in enumerate(self.conv_layers):
            x = layer(x)
        x = self.flatten(x)
        for lI, layer in enumerate(self.dense_layers):
            x = layer(x)
        mu = self.fc_mu(x)
        std = self.fc_std(x)
        
        return mu, std
    
    def reparameterize(self, mu, std):
        z = mu + std*torch.randn_like(std)
        return z

encoder = Encoder(latent_dim, mel_count, mel_filters, ae_conv_channel_counts, ae_conv_kernel_size, ae_dense_layer_sizes).to(device)
encoder.load_state_dict(torch.load(ae_encoder_weights_file, map_location=device))
encoder.eval()

"""
Create VAE Decoder
"""

class Decoder(nn.Module):
    
    def __init__(self, latent_dim, mel_count, mel_filters, conv_channel_counts, conv_kernel_size, dense_layer_sizes):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mel_count = mel_count
        self.mel_filters = mel_filters
        self.conv_channel_counts = conv_channel_counts
        self.conv_kernel_size = conv_kernel_size
        self.dense_layer_sizes = dense_layer_sizes
        
        # create dense layers
        self.dense_layers = nn.ModuleList()
        
        stride = ((self.conv_kernel_size[0] - 1) // 2, (self.conv_kernel_size[1] - 1) // 2)

        self.dense_layers.append(nn.Linear(latent_dim, self.dense_layer_sizes[0]))
        self.dense_layers.append(nn.ReLU())
        
        dense_layer_count = len(dense_layer_sizes)
        for layer_index in range(1, dense_layer_count):
            self.dense_layers.append(nn.Linear(self.dense_layer_sizes[layer_index-1], self.dense_layer_sizes[layer_index]))
            self.dense_layers.append(nn.ReLU())
            
        last_conv_layer_size_x = int(mel_filters // np.power(stride[0], len(conv_channel_counts)))
        last_conv_layer_size_y = int(mel_count // np.power(stride[1], len(conv_channel_counts)))

        preflattened_size = [conv_channel_counts[0], last_conv_layer_size_x, last_conv_layer_size_y]

        dense_layer_output_size = conv_channel_counts[0] * last_conv_layer_size_x * last_conv_layer_size_y
 
        self.dense_layers.append(nn.Linear(self.dense_layer_sizes[-1], dense_layer_output_size))
        self.dense_layers.append(nn.ReLU())

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=preflattened_size)
        
        # create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        padding = stride
        output_padding = (padding[0] - 1, padding[1] - 1) # does this universally work?
        
        conv_layer_count = len(conv_channel_counts)
        for layer_index in range(1, conv_layer_count):
            self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[layer_index-1]))
            self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[layer_index-1], conv_channel_counts[layer_index], self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))
            self.conv_layers.append(nn.LeakyReLU(0.2))
            
        self.conv_layers.append(nn.BatchNorm2d(conv_channel_counts[-1]))
        self.conv_layers.append(nn.ConvTranspose2d(conv_channel_counts[-1], 1, self.conv_kernel_size, stride=stride, padding=padding, output_padding=output_padding))

    def forward(self, x):

        for lI, layer in enumerate(self.dense_layers):
            x = layer(x)
        x = self.unflatten(x)
        for lI, layer in enumerate(self.conv_layers):
            x = layer(x)

        return x

decoder = Decoder(latent_dim, mel_count, mel_filters, list(reversed(ae_conv_channel_counts)), ae_conv_kernel_size, list(reversed(ae_dense_layer_sizes))).to(device)
decoder.load_state_dict(torch.load(ae_decoder_weights_file, map_location=device))
decoder.eval()

"""
Audio Callbacks
"""

audio_source_start_frame_index_1 = 0
audio_source_start_frame_index_2 = 0
audio_source_end_frame_index_1 = audio_source_samples_1.shape[0]
audio_source_end_frame_index_2 = audio_source_samples_2.shape[0]

audio_encoding_mix_factors = torch.zeros((latent_dim), dtype=torch.float32).to(device)
audio_encoding_offset_factors = torch.zeros((latent_dim), dtype=torch.float32).to(device)

@torch.no_grad()
def encode_audio(waveform):
    waveform = waveform.unsqueeze(0).to(device)
    mels = vocos.feature_extractor(waveform)
    audio_encoder_in = mels.unsqueeze(1)
    mu, std = encoder(audio_encoder_in)
    std = torch.nn.functional.softplus(std) + 1e-6
    return Encoder.reparameterize(mu, std)

@torch.no_grad()
def decode_audio(latent):
    mel_pred = decoder(latent).squeeze(1)
    mel_pred = mel_pred.squeeze(1)
    waveform = vocos.decode(mel_pred).reshape(1, -1)
    return waveform

@torch.no_grad()
def synthesize_audio():
    
    global audio_source_frame_index_1, audio_source_frame_index_2

    waveform_excerpt_1 = audio_source_samples_1[audio_source_frame_index_1:audio_source_frame_index_1 + window_size]
    waveform_excerpt_2 = audio_source_samples_2[audio_source_frame_index_2:audio_source_frame_index_2 + window_size]
    
    waveform_excerpt_1 = waveform_excerpt_1.unsqueeze(0).to(device)
    waveform_excerpt_2 = waveform_excerpt_2.unsqueeze(0).to(device)
    
    #print("waveform_excerpt_1 s ", waveform_excerpt_1.shape)
    
    mels_excerpt_1 = vocos.feature_extractor(waveform_excerpt_1)
    mels_excerpt_2 = vocos.feature_extractor(waveform_excerpt_2)
    
    #print("mels_excerpt_1 s ", mels_excerpt_1.shape)
    
    encoder_in_1 = mels_excerpt_1.unsqueeze(1)
    encoder_in_2 = mels_excerpt_2.unsqueeze(1)
    
    #print("encoder_in_1 s ", encoder_in_1.shape)
    
    mu_1, std_1 = encoder(encoder_in_1)
    mu_2, std_2 = encoder(encoder_in_2)
    
    #print("mu_1 s ", mu_1.shape)
    
    std_1 = torch.nn.functional.softplus(std_1) + 1e-6
    std_2 = torch.nn.functional.softplus(std_2) + 1e-6
    
    encoding_1 = encoder.reparameterize(mu_1, std_1)
    encoding_2 = encoder.reparameterize(mu_2, std_2)
    
    #print("encoding_1 s ", encoding_1.shape)
    
    decoder_in = encoding_1 * (1.0 - audio_encoding_mix_factors) + encoding_2 * audio_encoding_mix_factors
    decoder_in = decoder_in + audio_encoding_offset_factors
    
    #print("decoder_in s ", decoder_in.shape)
    
    gen_mels_excerpt = decoder(decoder_in)
    
    #print("gen_mels_excerpt s ", gen_mels_excerpt.shape)
    
    gen_mels_excerpt = gen_mels_excerpt.squeeze(1)
    gen_waveform_excerpt = vocos.decode(gen_mels_excerpt.detach()).reshape(-1)
    
    audio_source_frame_index_1 = audio_source_frame_index_1 + window_offset
    if audio_source_frame_index_1 >= audio_source_samples_1.shape[0] - window_size:
        audio_source_frame_index_1 = 0
    
    audio_source_frame_index_2 = audio_source_frame_index_2 + window_offset
    if audio_source_frame_index_2 >= audio_source_samples_2.shape[0] - window_size:
        audio_source_frame_index_2 = 0

    return gen_waveform_excerpt

"""
Audio Threading
"""

audio_queue = queue.Queue(maxsize=max_audio_queue_length)
export_audio_buffer = []
last_chunk = np.zeros(window_size, dtype=np.float32)

def producer_thread():
    """Continuously generates audio and fills the output buffer queue."""
    while True:
        if not audio_queue.full():
            gen_waveform = synthesize_audio()
            audio_queue.put(gen_waveform.cpu().numpy())
        else:
            sd.sleep(10)

def audio_callback(out_data, frames, time_info, status):
    """sounddevice stream callback function."""
    global last_chunk
    output = np.zeros((frames, audio_channel_count), dtype=np.float32)
    cursor = 0
    overlap_len = window_size // 2
    output[cursor:cursor+overlap_len, 0] += last_chunk[overlap_len:]
    samples_needed = frames
    while samples_needed > 0:
        try:
            chunk = audio_queue.get_nowait()
            chunk = (chunk * hann_window.cpu().numpy())
        except queue.Empty:
            chunk = np.zeros(window_size, dtype=np.float32)
        chunk_copy_size = output[cursor:cursor+window_size, 0].shape[0]
        output[cursor:cursor+chunk_copy_size, 0] += chunk[:chunk_copy_size]
        cursor += window_size // 2
        samples_needed = frames - cursor
        last_chunk[:] = chunk[:]
    out_data[:] = output

def run_audio_stream():
    """Main thread for launching audio producer and output stream."""
    threading.Thread(target=producer_thread, daemon=True).start()
    sd.sleep(2000)
    with sd.OutputStream(
        samplerate=audio_sample_rate,
        device=audio_output_device,
        channels=audio_channel_count,
        callback=audio_callback,
        blocksize=play_buffer_size,
        latency=playback_latency
    ):
        print("Streaming audio with FIFO queue... press Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped.")


"""
OSC Control
"""

def osc_setIndex1(address, *args):
    global audio_source_start_frame_index_1
    
    audio_source_start_frame_index_1 = args[0]
    
def osc_setIndex2(address, *args):
    global audio_source_start_frame_index_2
    
    audio_source_start_frame_index_2 = args[0]
     
def osc_setRange1(address, *args):
    global audio_source_start_frame_index_1, audio_source_end_frame_index_1
    
    audio_source_start_frame_index_1 = args[0]
    audio_source_end_frame_index_1 = args[1]

def osc_setRange2(address, *args):
    global audio_source_start_frame_index_2, audio_source_end_frame_index_2
    4
    audio_source_start_frame_index_2 = args[0]
    audio_source_end_frame_index_2 = args[1]

def osc_setEncodingMix(address, *args):
    
    global audio_encoding_mix_factors
    
    arg_count = len(args)
    
    for i in range(min(arg_count, latent_dim)):
        audio_encoding_mix_factors[i] = args[i]

def osc_setEncodingOffset(address, *args):
    
    global audio_encoding_offset_factors

    arg_count = len(args)
        
    for i in range(min(arg_count, latent_dim)):
        audio_encoding_offset_factors[i] = args[i]

osc_handler = dispatcher.Dispatcher()
osc_handler.map("/audio/sampleindex1", osc_setIndex1)
osc_handler.map("/audio/sampleindex2", osc_setIndex2)
osc_handler.map("/audio/samplerange1", osc_setRange1)
osc_handler.map("/audio/samplerange2", osc_setRange2)
osc_handler.map("/synth/encodingmix", osc_setEncodingMix)
osc_handler.map("/synth/encodingoffset", osc_setEncodingOffset)

"""
Start OSC
"""

osc_server = osc_server.ThreadingOSCUDPServer((osc_receive_ip, osc_receive_port), osc_handler)

def osc_start_receive():
    osc_server.serve_forever()

osc_thread = threading.Thread(target=osc_start_receive)
osc_thread.start()

"""
Start Audio
"""

audio_thread = threading.Thread(target=run_audio_stream, daemon=True)
audio_thread.start()


