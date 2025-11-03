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
Load Audio
"""

audioFile = "../../Data/sounds/Night_and_Day_by_Virginia_Woolf_48khz.wav"
audioExcerptSec = [ 10.0, 14.0 ]

audioWaveform, audioSampleRate = ta.load(audioFile)
audioWaveform = audioWaveform[:, int(audioExcerptSec[0] * audioSampleRate) : int(audioExcerptSec[1] * audioSampleRate) ]

"""
Play Audio
"""

play_obj = sa.play_buffer(audioWaveform.numpy(), 1, 4, audioSampleRate)
play_obj.wait_done()

"""
Plot Waveform
"""

plt.title('Waveform')
plt.plot(audioWaveform[0].numpy())

"""
FFT
"""

# Audio Buffer
audioBufferSize = 1024 # 4096
audioBuffer = audioWaveform[:, :audioBufferSize]

plt.title('Audio Buffer')
plt.plot(audioBuffer[0].numpy())
plt.show()

# Audio Window
audioWindow = torch.from_numpy(np.hanning(audioBufferSize)).unsqueeze(0)

plt.title('Hanning Window')
plt.plot(audioWindow[0].numpy())
plt.show()

# Windowed Audio

audioBufferWindowed = audioBuffer * audioWindow

plt.title('Audio Buffer Windowed')
plt.plot(audioBufferWindowed[0].numpy())
plt.show()

# FFT

spectrum = torch.fft.fft(audioBufferWindowed)

# entire spectrum real part
plt.title('Audio Spectrum Real')
plt.plot(torch.abs(spectrum[0].real).numpy())
plt.show()

# first half of spectrum absolute of real part
plt.title('Audio Spectrum Real')
plt.plot(torch.abs(spectrum[0, :audioBufferSize//2].real).numpy())
plt.show()

# first half of spectrum absolute of imaginary part
plt.title('Audio Spectrum Imaginary')
plt.plot(torch.abs(spectrum[0, :audioBufferSize//2].imag).numpy())
plt.show()

# magnitude of spectrum
plt.title('Audio Spectrum Magnitude')
plt.plot(spectrum[0, :audioBufferSize//2].abs().numpy())
plt.show()

# phase of spectrum
plt.title('Audio Spectrum Phase')
plt.plot(spectrum[0, :audioBufferSize//2].angle().numpy())
plt.show()

"""
IFFT
"""

audioBufferRec = torch.fft.ifft(spectrum).real

plt.title('Reconstructed Audio Buffer')
plt.plot(audioBufferRec[0].numpy())


"""
Spectrogram

see: https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html
"""

nFFT = 1024
spectrogram = ta.transforms.Spectrogram(n_fft=nFFT)(audioWaveform)


plt.title('Waveform')
plt.plot(audioWaveform[0].numpy())

plt.imshow(librosa.power_to_db(spectrogram[0].numpy()), origin="lower", aspect="auto", interpolation="nearest")


