"""
Audio Analysis
"""

"""
Imports
"""

import os
import numpy as np
import scipy, sklearn
import librosa, librosa.display
import simpleaudio as sa
from matplotlib import pyplot as plt
from sklearn import preprocessing

"""
Audio File Settings
"""

audio_file = "../../Data/Audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
audio_sample_rate = 48000
audio_excerpt_sec = [ 10.0, 14.0 ]

"""
Load Audio
"""

audio_waveform, _ = librosa.load(audio_file, sr=audio_sample_rate)
audio_waveform = audio_waveform[ int(audio_excerpt_sec[0] * audio_sample_rate) : int(audio_excerpt_sec[1] * audio_sample_rate) ]

"""
Play Audio
"""

play_obj = sa.play_buffer(audio_waveform, 1, 4, audio_sample_rate)
play_obj.wait_done()

"""
Plot Audio Waveform
"""

plt.title('Waveform')
plt.plot(audio_waveform)
plt.show()

"""
Plot Audio Spectrogram
"""

plt.title('Spectrogram')
plt.specgram(audio_waveform, Fs=audio_sample_rate)
plt.show()

"""
Calculate Audio Features
"""

audio_features = {}

audio_features["waveform"] = audio_waveform

audio_features["root mean square"] = librosa.feature.rms(y=audio_waveform)
audio_features["mel spectrogram"] = librosa.feature.melspectrogram(y=audio_waveform, sr=audio_sample_rate)
audio_features["mfcc"] = librosa.feature.mfcc(y=audio_waveform, sr=audio_sample_rate)
audio_features["spectral centroid"] = librosa.feature.spectral_centroid(y=audio_waveform, sr=audio_sample_rate)
audio_features["spectral bandwidth"] = librosa.feature.spectral_bandwidth(y=audio_waveform, sr=audio_sample_rate)
audio_features["spectral contrast"] = librosa.feature.spectral_contrast(y=audio_waveform, sr=audio_sample_rate)
audio_features["spectral flatness"] = librosa.feature.spectral_flatness(y=audio_waveform)
audio_features["spectral rolloff"] = librosa.feature.spectral_rolloff(y=audio_waveform, sr=audio_sample_rate)
audio_features["tempogram ratio"] = librosa.feature.tempogram(y=audio_waveform, sr=audio_sample_rate)


"""
Plot Audio Features
"""

plt.title('root mean square')
plt.plot(audio_features["root mean square"][0])
plt.show()

plt.title('mel spectrogram')
plt.imshow(librosa.amplitude_to_db(audio_features["mel spectrogram"]), aspect=1, origin='lower')
plt.show()

plt.title('mfcc')
plt.imshow(audio_features["mfcc"], aspect=10, origin='lower')
plt.show()

plt.title('spectral centroid')
plt.plot(audio_features["spectral centroid"][0]) 
plt.show()

plt.title('spectral bandwidth')
plt.plot(audio_features["spectral bandwidth"][0]) 
plt.show()

plt.title('spectral contrast')
for i in range(audio_features["spectral contrast"].shape[0]):
    plt.plot(audio_features["spectral contrast"][i])
plt.show()

plt.title('spectral flatness')
plt.plot(audio_features["spectral flatness"][0])
plt.show()

plt.title('spectral rolloff')
plt.plot(audio_features["spectral rolloff"][0])
plt.show()


"""
Plot Audio Features Superimposed Over Waveform
"""

# helper function: normalise feature
def normalize_feature(feature):
    
    min_value = np.min(feature, axis=1, keepdims=True)
    max_value = np.max(feature, axis=1, keepdims=True)
    
    return (feature - min_value) / (max_value - min_value)

# helper function: 
def resample_feature(feature):
    return scipy.signal.resample(feature, audio_waveform.shape[0])

plt.title('root mean square')
plt.plot(audio_waveform)
plt.plot(resample_feature(audio_features["root mean square"][0]), color='r')
plt.show()

plt.title('spectral centroid')
plt.plot(audio_waveform)
plt.plot(resample_feature(normalize_feature(audio_features["spectral centroid"])[0]), color='r')
plt.show()

plt.title('spectral bandwidth')
plt.plot(audio_waveform)
plt.plot(resample_feature(normalize_feature(audio_features["spectral bandwidth"])[0]), color='r')
plt.show()

plt.title('spectral contrast')
plt.plot(audio_waveform)
for i in range(audio_features["spectral contrast"].shape[0]):
    plt.plot(resample_feature(normalize_feature(audio_features["spectral contrast"])[i]))
plt.show()

plt.title('spectral flatness')
plt.plot(audio_waveform)
plt.plot(resample_feature(normalize_feature(audio_features["spectral flatness"])[0]), color='r')
plt.show()

plt.title('spectral rolloff')
plt.plot(audio_waveform)
plt.plot(resample_feature(normalize_feature(audio_features["spectral rolloff"])[0]), color='r')
plt.show()
