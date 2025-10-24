"""
Audio Clustering
"""

"""
Imports
"""

import audio_analysis
import audio_model
import audio_synthesis
import audio_gui
import audio_control

import librosa
import numpy as np
import os, sys, time, subprocess, threading

import sounddevice as sd

"""
Audio Settings
"""

audio_file_path = "../../Data/Audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
audio_sample_rate = 48000
audio_channel_count = 1
audio_range_sec = [ 10.0, 70.0 ]
audio_excerpt_length = 100 # in milisecs
audio_excerpt_offset = 90 # in milisecs

audio_output_device = 7
audio_buffer_size = 512

"""
Cluster Settings
"""

cluster_count = 20
cluster_random_state = 170

"""
OSC Control Settings
"""

osc_receive_ip = "0.0.0.0"
osc_receive_port = 9002

"""
Load Audio
"""

audio_waveform, _ = librosa.load(audio_file_path, sr=audio_sample_rate)
audio_waveform = audio_waveform[int(audio_range_sec[0] * audio_sample_rate):int(audio_range_sec[1] * audio_sample_rate)]
audio_waveform_sc = audio_waveform.shape[0]
"""
Create Audio Excerpts
"""

audio_excerpts = []

audio_excerpt_length_sc = int(audio_excerpt_length / 1000 * audio_sample_rate)
audio_excerpt_offset_sc = int(audio_excerpt_offset / 1000 * audio_sample_rate)

for sI in range(0, audio_waveform_sc - audio_excerpt_length_sc, audio_excerpt_offset_sc):
    
    audio_excerpt = audio_waveform[sI:sI + audio_excerpt_length_sc]
    audio_excerpts.append(audio_excerpt)
    
audio_excerpts = np.stack(audio_excerpts, axis=0)

"""
Calculate Audio Features
"""

audio_features = {}
audio_features["waveform"] = audio_excerpts
audio_features["root mean square"] = audio_analysis.rms(audio_excerpts)
audio_features["chroma stft"] = audio_analysis.chroma_stft(audio_excerpts, audio_sample_rate)
#audio_features["chroma cqt"] = audio_analysis.chroma_cqt(audio_excerpts, audio_sample_rate)
#audio_features["chroma cens"] = audio_analysis.chroma_cens(audio_excerpts, audio_sample_rate)
#audio_features["chroma vqt"] = audio_analysis.chroma_vqt(audio_excerpts, audio_sample_rate)
audio_features["mel spectrogram"] = audio_analysis.mel_spectrogram(audio_excerpts, audio_sample_rate)
audio_features["mfcc"] = audio_analysis.mfcc(audio_excerpts, audio_sample_rate)
audio_features["spectral centroid"] = audio_analysis.spectral_centroid(audio_excerpts, audio_sample_rate)
audio_features["spectral bandwidth"] = audio_analysis.spectral_bandwidth(audio_excerpts, audio_sample_rate)
audio_features["spectral contrast"] = audio_analysis.spectral_contrast(audio_excerpts, audio_sample_rate)
audio_features["spectral flatness"] = audio_analysis.spectral_flatness(audio_excerpts)
audio_features["spectral rolloff"] = audio_analysis.spectral_rolloff(audio_excerpts, audio_sample_rate)
#audio_features["tempo"] = audio_analysis.tempo(audio_excerpts, audio_sample_rate)
audio_features["tempogram"] = audio_analysis.tempogram(audio_excerpts, audio_sample_rate)
#audio_features["tempogram ratio"] = audio_analysis.tempogram_ratio(audio_excerpts, audio_sample_rate)


"""
Create Clustering Model
"""

audio_model.config = {
    "audio_excerpts": audio_excerpts,
    "audio_features": audio_features,
    "cluster_method": "kmeans",
    "cluster_count": cluster_count,
    "cluster_random_state": cluster_random_state
    }

clustering = audio_model.createModel(audio_model.config)


"""
Create Cluster Player
"""

audio_synthesis.config = {
    "model": clustering,
    "audio_excerpts": audio_excerpts,
    "audio_sample_rate": audio_sample_rate,
    "audio_excerpt_length": audio_excerpt_length,
    "audio_excerpt_offset": audio_excerpt_offset
    }

synthesis = audio_synthesis.AudioSynthesis(audio_synthesis.config)
synthesis.setClusterLabel(1)
synthesis.selectAudioFeature(list(audio_features.keys())[0])


"""
Create GUI
"""

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pathlib import Path

app = QtWidgets.QApplication(sys.argv)
gui = audio_gui.AudioGui()

# set close event
def closeEvent():
    QtWidgets.QApplication.quit()
app.lastWindowClosed.connect(closeEvent) # myExitHandler is a callable

"""
Create OSC Control
"""

audio_control.config["synthesis"] = synthesis
audio_control.config["model"] = clustering
audio_control.config["ip"] = "0.0.0.0"
audio_control.config["port"] = 9002

osc_control = audio_control.AudioControl(audio_control.config)

"""
Real-Time Audio
"""

event = threading.Event()

def audio_callback(out_data, frame_count, time_info, status):
    
    synthesis.update(out_data.reshape(-1))


audio_stream = sd.OutputStream(
    samplerate=audio_sample_rate, device=audio_output_device, channels=audio_channel_count,
    callback=audio_callback)


osc_control.start()
audio_stream.start()
gui.show()
app.exec_()

"""
audio_stream.stop()
osc_control.stop()
"""


