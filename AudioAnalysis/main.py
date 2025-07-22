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
Load Audio
"""

audio_file = "/Users/dbisig/Data/audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
audio_sample_rate = 48000
audio_excerpt_sec = [ 10.0, 14.0 ]

audio_waveform, _ = librosa.load(audio_file, sr=audio_sample_rate)
audio_waveform = audio_waveform[ int(audio_excerpt_sec[0] * audio_sample_rate) : int(audio_excerpt_sec[1] * audio_sample_rate) ]

"""
Play Audio
"""

play_obj = sa.play_buffer(audio_waveform, 1, 4, audio_sample_rate)
play_obj.wait_done()

"""
Plot Waveform
"""

plt.title('Waveform')
plt.plot(audio_waveform)

"""
Plot Spectrogram
"""

plt.title('Spectrogram')
plt.specgram(audio_waveform, Fs=audio_sample_rate)

"""
Calculate Audio Features
"""

audio_features = {}

audio_features["waveform"] = audio_waveform

# root mean square : energy over time
audio_features["root mean square"] = librosa.feature.rms(y=audio_waveform)

audio_features["chroma stft"] = librosa.feature.chroma_stft(y=audio_waveform, sr=audio_sample_rate)
audio_features["chroma cqt"] = librosa.feature.chroma_cqt(y=audio_waveform, sr=audio_sample_rate)
audio_features["chroma cens"] = librosa.feature.chroma_cens(y=audio_waveform, sr=audio_sample_rate)
audio_features["chroma vqt"] = librosa.feature.chroma_vqt(y=audio_waveform, sr=audio_sample_rate, intervals="ji5")
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

def normalize(x, axis=0):
    return preprocessing.minmax_scale(x, axis=axis)

plt.title('root mean square')
plt.plot(audio_features["root mean square"][0])
plt.show()

plt.title('chroma stft')
plt.imshow(audio_features["chroma stft"], aspect=10, origin='lower')
plt.show()

plt.title('chroma cqt')
plt.imshow(audio_features["chroma cqt"], aspect=10, origin='lower')
plt.show()

plt.title('chroma cens')
plt.imshow(audio_features["chroma cens"], aspect=10, origin='lower')
plt.show()

plt.title('chroma vqt')
plt.imshow(audio_features["chroma vqt"], aspect=10, origin='lower')
plt.show()

plt.title('mel spectrogram')
plt.imshow(audio_features["mel spectrogram"], aspect=1, origin='lower')
plt.show()

plt.title('mfcc')
plt.imshow(audio_features["mfcc"], aspect=10, origin='lower')
plt.show()

plt.title('spectral centroid')
plt.plot(audio_waveform)
plt.plot(scipy.signal.resample(normalize(audio_features["spectral centroid"][0]), audio_waveform.shape[0]), color='r') # normalize for visualization purposes
plt.show()

plt.title('spectral bandwidth')
plt.plot(audio_waveform)
plt.plot(scipy.signal.resample(normalize(audio_features["spectral bandwidth"][0]), audio_waveform.shape[0]), color='r') # normalize for visualization purposes
plt.show()

plt.title('spectral contrast')
plt.plot(audio_waveform)
plt.plot(scipy.signal.resample(normalize(audio_features["spectral contrast"][0]), audio_waveform.shape[0]), color='r') # normalize for visualization purposes
plt.show()

plt.title('spectral flatness')
plt.plot(audio_waveform)
plt.plot(scipy.signal.resample(normalize(audio_features["spectral flatness"][0]), audio_waveform.shape[0]), color='r') # normalize for visualization purposes
plt.show()

plt.title('spectral rolloff')
plt.plot(audio_waveform)
plt.plot(scipy.signal.resample(normalize(audio_features["spectral rolloff"][0]), audio_waveform.shape[0]), color='r') # normalize for visualization purposes
plt.show()

plt.title('tempogram ratio')
plt.imshow(audio_features["tempogram ratio"], aspect=1, origin='lower')
plt.show()



