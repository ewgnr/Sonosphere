"""
Imports
"""

import os
import numpy as np
import librosa
import torch
import torchaudio
import soundfile as sf
import audio_analysis
from matplotlib import pyplot as plt
import wave
import time

from sklearn.cluster import KMeans

"""
Audio Settings
"""

audio_file_path = "/Users/dbisig/Data/audio/DanielBisig/ostrawa_recordings_selection"
audio_file_extensions = ["wav", "aiff", "aif"] 
audio_sample_rate = 48000
audio_sample_count = audio_sample_rate // 1
audio_sample_offset = audio_sample_count // 2

audio_feature_names = ["root mean square", "mfcc"]

"""
Load Audio
"""

audio_waveforms_full = []

for root, _, fnames in sorted(os.walk(audio_file_path, followlinks=True)):
    for fname in sorted(fnames):
        
        path = root + "/" + fname
        
        print("path ", path)
        
        audio_waveform, _ = librosa.load(path, sr=audio_sample_rate)

        print("waveform s ", audio_waveform.shape)
        
        audio_waveforms_full.append(audio_waveform)

"""
Create Audio Excerpts
"""

audio_waveform_excerpts = []

for waveform_full in audio_waveforms_full:
    waveform_sample_count = waveform_full.shape[0]
    for sI in range(0, waveform_sample_count - audio_sample_count, audio_sample_offset):
        waveform_except = waveform_full[sI:sI + audio_sample_count]
        audio_waveform_excerpts.append(waveform_except)
        
audio_waveforms = np.stack(audio_waveform_excerpts, axis=0)
        
"""
Calculate Audio Features
"""

audio_features = {}

audio_features["waveform"] = audio_waveforms
if "root mean square" in audio_feature_names:
    audio_features["root mean square"] = audio_analysis.rms(audio_waveforms)
if "chroma stft" in audio_feature_names:
    audio_features["chroma stft"] = audio_analysis.chroma_stft(audio_waveforms, audio_sample_rate)
if "chroma cqt" in audio_feature_names:
    audio_features["chroma cqt"] = audio_analysis.chroma_cqt(audio_waveforms, audio_sample_rate)
if "chroma cens" in audio_feature_names:
    audio_features["chroma cens"] = audio_analysis.chroma_cens(audio_waveforms, audio_sample_rate)
if "chroma vqt" in audio_feature_names:
    audio_features["chroma vqt"] = audio_analysis.chroma_vqt(audio_waveforms, audio_sample_rate)
if "mel spectrogram" in audio_feature_names:
    audio_features["mel spectrogram"] = audio_analysis.mel_spectrogram(audio_waveforms, audio_sample_rate)
if "mfcc" in audio_feature_names:
    audio_features["mfcc"] = audio_analysis.mfcc(audio_waveforms, audio_sample_rate)
if "spectral centroid" in audio_feature_names:
    audio_features["spectral centroid"] = audio_analysis.spectral_centroid(audio_waveforms, audio_sample_rate)
if "spectral bandwidth" in audio_feature_names:
    audio_features["spectral bandwidth"] = audio_analysis.spectral_bandwidth(audio_waveforms, audio_sample_rate)
if "spectral contrast" in audio_feature_names:
    audio_features["spectral contrast"] = audio_analysis.spectral_contrast(audio_waveforms, audio_sample_rate)
if "spectral flatness" in audio_feature_names:
    audio_features["spectral flatness"] = audio_analysis.spectral_flatness(audio_waveforms)
if "spectral rolloff" in audio_feature_names:
    audio_features["spectral rolloff"] = audio_analysis.spectral_rolloff(audio_waveforms, audio_sample_rate)
if "tempo" in audio_feature_names:
    audio_features["tempo"] = audio_analysis.tempo(audio_waveforms, audio_sample_rate)
if "tempogram" in audio_feature_names:
    audio_features["tempogram"] = audio_analysis.tempogram(audio_waveforms, audio_sample_rate)
if "tempogram ratio" in audio_feature_names:
    audio_features["tempogram ratio"] = audio_analysis.tempogram_ratio(audio_waveforms, audio_sample_rate)

"""
Normalise Audio Features
"""

for audio_feature_name in list(audio_features.keys()):
    
    #print(audio_feature_name)
    
    audio_feature = audio_features[audio_feature_name]
    
    audio_feature_mean = np.mean(audio_feature)
    audio_feature_std = np.std(audio_feature)
    
    audio_feature_norm = (audio_feature - audio_feature_mean) / audio_feature_std
    
    #print("audio_feature_norm s ", audio_feature_norm.shape)

    audio_features[audio_feature_name + " norm"] = audio_feature_norm

"""
Combine Audio Features
"""

audio_features_proc = []
for audio_feature_name in audio_feature_names:
    audio_norm_feature_name = audio_feature_name + " norm"
    audio_feature = audio_features[audio_norm_feature_name]
    #print("name ", audio_norm_feature_name, " shape ", audio_feature.shape)
    audio_features_proc.append(audio_feature)
    
audio_features_proc = np.concatenate(audio_features_proc, axis=1)

"""
KMeans Clustering
"""

cluster_count = 10
random_state = 170

km = KMeans(n_clusters=cluster_count, n_init= "auto", random_state = random_state)
labels =  km.fit_predict(audio_features_proc)

"""
Export Audio Files Per Cluster
"""

audio_waveform_sc = audio_waveforms[0].shape[0]
audio_amp_envelope = np.hanning(audio_waveform_sc)

fileIndex = 0

for excerpt_index, label in enumerate(labels):
    cluster_index = label
    audio_waveform = audio_waveforms[excerpt_index]

    audio_file_name = "data_proc/audio_cluster_{:05d}_excerpt_{:05d}_fcount_{:05d}.wav".format(cluster_index, excerpt_index, fileIndex)
    sf.write(audio_file_name, audio_waveform * audio_amp_envelope, audio_sample_rate)
    
    fileIndex += 1
