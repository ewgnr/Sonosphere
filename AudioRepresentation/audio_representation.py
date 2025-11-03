"""
Audio Representation
"""

"""
Imports
"""

import numpy as np
import torch
import torchaudio as ta
import simpleaudio as sa
import librosa
from matplotlib import pyplot as plt

"""
Audio File Settings
"""

audio_file_path = "../../Data/Audio/Gutenberg/Night_and_Day_by_Virginia_Woolf_48khz.wav"
audio_sample_rate = 48000
audio_range_sec = [ 10.0, 14.0 ]

"""
Load Audio File
"""

audio_waveform, _ = librosa.load(audio_file_path, sr=audio_sample_rate)

if len(audio_waveform.shape) == 1:
    audio_waveform = np.expand_dims(audio_waveform, 0)

audio_waveform = audio_waveform[:, int(audio_range_sec[0] * audio_sample_rate) : int(audio_range_sec[1] * audio_sample_rate) ]

"""
Play Audio
"""

play_obj = sa.play_buffer(audio_waveform, 1, 4, audio_sample_rate)
play_obj.wait_done()

"""
Plot Audio Waveform
"""

plt.title('Waveform')
plt.plot(audio_waveform[0])

"""
Audio Buffer
"""

audio_buffer_size = 1024

audio_buffer = audio_waveform[0, :audio_buffer_size]

plt.title('Audio Buffer')
plt.plot(audio_buffer)
plt.show()

"""
Amplitude Envelope
"""

audio_window = np.hanning(audio_buffer_size)

plt.title('Hanning Window')
plt.plot(audio_window)
plt.show()

"""
Windowed Audio Buffer
"""

audio_buffer_windowed = audio_buffer * audio_window

plt.title('Audio Buffer Windowed')
plt.plot(audio_buffer_windowed)
plt.show()

"""
Calculate Audio Spectrum
"""

audio_buffer_windowed = torch.from_numpy(audio_buffer_windowed)

audio_spectrum = torch.fft.fft(audio_buffer_windowed)

"""
Plot Audio Spectrum
"""

# magnitude of spectrum
plt.title('Audio Spectrum Magnitude')
plt.plot(audio_spectrum[:audio_buffer_size//2].abs().numpy())
plt.show()

# phase of spectrum
plt.title('Audio Spectrum Phase')
plt.plot(audio_spectrum[:audio_buffer_size//2].angle().numpy())
plt.show()

"""
Reconstruct Audio Buffer from Spectrum
"""

audio_buffer_rec = torch.fft.ifft(audio_spectrum).real

plt.title('Reconstructed Audio Buffer')
plt.plot(audio_buffer_rec.numpy())

"""
Calculate Audio Spectrogram

see: https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html
"""

nFFT = 1024

audio_spectrogram = ta.transforms.Spectrogram(n_fft=nFFT)(torch.from_numpy(audio_waveform))

"""
Plot Audio Spectrogram
"""

plt.figure(figsize=(10, 4))
plt.imshow(audio_spectrogram.squeeze().log2().numpy(), aspect='auto', origin='lower')
plt.title('Audio Spectrogram (log2)')
plt.xlabel('Frame')
plt.ylabel('FFT bins')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

"""
Calculate MEL Spectrogram
"""

nFFT = 1024
nMels = 128

# Create MelSpectrogram transform
mel_transform = ta.transforms.MelSpectrogram(
    sample_rate=audio_sample_rate,
    n_fft=nFFT,
    hop_length=nFFT // 2,
    n_mels=nMels
)

# Compute Mel spectrogram (shape: [channels, n_mels, time])
mel_spec = mel_transform(torch.from_numpy(audio_waveform))

"""
Plot Mel Spectrogram
"""

plt.figure(figsize=(10, 4))
plt.imshow(mel_spec.squeeze().log2().numpy(), aspect='auto', origin='lower')
plt.title('Mel Spectrogram (log2)')
plt.xlabel('Frame')
plt.ylabel('Mel bins')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()