import numpy as np
import librosa

def rms(audio_excerpts):
    
    """
    Compute root-mean-square (RMS) value for each frame from the audio samples
    """
    
    root_mean_square = []
    
    for audio_excerpt in audio_excerpts:
    
        rms = librosa.feature.rms(y=audio_excerpt)
        rms = rms.flatten()
        
        root_mean_square.append(rms)

    root_mean_square = np.stack(root_mean_square, axis=0)
    
    return root_mean_square

def chroma_stft(audio_excerpts, audio_sample_rate):
    
    """
    Compute a chromagram from a waveform
    """
    
    chroma_stft = []
    
    for audio_excerpt in audio_excerpts:
    
        stft = librosa.feature.chroma_stft(y=audio_excerpt, sr=audio_sample_rate)
        stft = stft.flatten()
        
        chroma_stft.append(stft)
    
    chroma_stft = np.stack(chroma_stft, axis=0)
    
    return chroma_stft


def chroma_cqt(audio_excerpts, audio_sample_rate):

    # Constant-Q chromagram (super slow to calculate)
    
    chroma_cqt = []
    
    for audio_excerpt in audio_excerpts:
        
        cqt = librosa.feature.chroma_cqt(y=audio_excerpt, sr=audio_sample_rate)
        cqt = cqt.flatten()
        
        chroma_cqt.append(cqt)
    
    chroma_cqt = np.stack(chroma_cqt, axis=0)
    
    return chroma_cqt


def chroma_cens(audio_excerpts, audio_sample_rate):

    # Compute the chroma variant "Chroma Energy Normalized" (CENS)(super slow to calculate)
    
    chroma_cens = []
    
    for audio_excerpt in audio_excerpts:

        cens = librosa.feature.chroma_cens(y=audio_excerpt, sr=audio_sample_rate)
        cens = cens.flatten()
        
        chroma_cens.append(cens)
    
    chroma_cens = np.stack(chroma_cens, axis=0)
    
    return chroma_cens


def chroma_vqt(audio_excerpts, audio_sample_rate):
    
    # Variable-Q chromagram (slow to calculate)

    chroma_vqt = []
    
    for audio_excerpt in audio_excerpts:

        vqt = librosa.feature.chroma_vqt(y=audio_excerpt, sr=audio_sample_rate, intervals="ji5")
        vqt = vqt.flatten()
        
        chroma_vqt.append(vqt)
    
    chroma_vqt = np.stack(chroma_vqt, axis=0)
    
    return chroma_vqt

def mel_spectrogram(audio_excerpts, audio_sample_rate):
    
    # Compute a mel-scaled spectrogram.

    mel_spectrogram = []
    
    for audio_excerpt in audio_excerpts:
    
        mel = librosa.feature.melspectrogram(y=audio_excerpt, sr=audio_sample_rate)
        mel = mel.flatten()
        
        mel_spectrogram.append(mel)
    
    mel_spectrogram = np.stack(mel_spectrogram, axis=0)
    
    return mel_spectrogram

def mfcc(audio_excerpts, audio_sample_rate):
    
    # Mel-frequency cepstral coefficients (MFCCs)

    mfcc = []
    
    for audio_excerpt in audio_excerpts:

        mfcc_ = librosa.feature.mfcc(y=audio_excerpt, sr=audio_sample_rate)
        mfcc_ = mfcc_.flatten()
        
        mfcc.append(mfcc_)
    
    mfcc = np.stack(mfcc, axis=0)
    
    return mfcc

def spectral_centroid(audio_excerpts, audio_sample_rate):

    # Compute the spectral centroid.

    spectral_centroid = []
    
    for audio_excerpt in audio_excerpts:
    
        cent = librosa.feature.spectral_centroid(y=audio_excerpt, sr=audio_sample_rate)
        cent = cent.flatten()
        
        spectral_centroid.append(cent)
    
    spectral_centroid = np.stack(spectral_centroid, axis=0)
    
    return spectral_centroid

def spectral_bandwidth(audio_excerpts, audio_sample_rate):

    # Compute p'th-order spectral bandwidth.

    spectral_bandwidth = []
    
    for audio_excerpt in audio_excerpts:
    
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_excerpt, sr=audio_sample_rate)
        bandwidth = bandwidth.flatten()
        
        spectral_bandwidth.append(bandwidth)
    
    spectral_bandwidth = np.stack(spectral_bandwidth, axis=0)
    
    return spectral_bandwidth

def spectral_contrast(audio_excerpts, audio_sample_rate):

    # Compute spectral contrast
    # High contrast values generally correspond to clear, narrow-band signals, while low contrast values correspond to broad-band noise. 

    spectral_contrast = []
    
    for audio_excerpt in audio_excerpts:
    
        contrast = librosa.feature.spectral_contrast(y=audio_excerpt, sr=audio_sample_rate)
        contrast = contrast.flatten()
        
        spectral_contrast.append(contrast)
    
    spectral_contrast = np.stack(spectral_contrast, axis=0)
    
    return spectral_contrast

def spectral_flatness(audio_excerpts):

    # Compute spectral flatness
    # Spectral flatness (or tonality coefficient) is a measure to quantify how much noise-like a sound is, as opposed to being tone-like 
    
    spectral_flatness = []
    
    for audio_excerpt in audio_excerpts:

        flatness = librosa.feature.spectral_flatness(y=audio_excerpt)
        flatness = flatness.flatten()
        
        spectral_flatness.append(flatness)
    
    spectral_flatness = np.stack(spectral_flatness, axis=0)

    return spectral_flatness

def spectral_rolloff(audio_excerpts, audio_sample_rate):

    # Compute roll-off frequency.
    
    spectral_rolloff = []
    
    for audio_excerpt in audio_excerpts:
    
        rolloff = librosa.feature.spectral_rolloff(y=audio_excerpt, sr=audio_sample_rate)
        rolloff = rolloff.flatten()
        
        spectral_rolloff.append(rolloff)
    
    spectral_rolloff = np.stack(spectral_rolloff, axis=0)
    
    return spectral_rolloff


def tempo(audio_excerpts, audio_sample_rate):

    # Estimate the tempo (beats per minute)
    
    tempo = []
    
    for audio_excerpt in audio_excerpts:

        tempo_ = librosa.feature.tempo(y=audio_excerpt, sr=audio_sample_rate)
        tempo_ = tempo_.flatten()
        
        tempo.append(tempo_)
    
    tempo = np.stack(tempo, axis=0)
        
    return tempo

def tempogram(audio_excerpts, audio_sample_rate):
    
    # Compute the tempogram: local autocorrelation of the onset strength envelope.
    
    tempogram = []
    
    for audio_excerpt in audio_excerpts:

        tempogram_ = librosa.feature.tempogram(y=audio_excerpt, sr=audio_sample_rate)
        tempogram_ = tempogram_.flatten()
        
        tempogram.append(tempogram_)
    
    tempogram = np.stack(tempogram, axis=0)
    
    return tempogram

def fourier_tempogram(audio_excerpts, audio_sample_rate):

    # Compute the Fourier tempogram: the short-time Fourier transform of the onset strength envelope.
    
    fourier_tempogram = []
    
    for audio_excerpt in audio_excerpts:

        tempogram = librosa.feature.fourier_tempogram(y=audio_excerpt, sr=audio_sample_rate)
        tempogram = tempogram.flatten()
        
        fourier_tempogram.append(tempogram)
    
    fourier_tempogram = np.stack(fourier_tempogram, axis=0)
    fourier_tempogram = np.stack([np.real(fourier_tempogram), np.imag(fourier_tempogram)], axis=-1)
    fourier_tempogram = fourier_tempogram.reshape(audio_excerpts.shape[0], -1)
    
    return fourier_tempogram

def tempogram_ratio(audio_excerpts, audio_sample_rate):

    # Tempogram ratio features, also known as spectral rhythm patterns.
    
    tempogram_ratio = []
    
    for audio_excerpt in audio_excerpts:

        ratio = librosa.feature.tempogram_ratio(y=audio_excerpt, sr=audio_sample_rate)
        ratio = ratio.flatten()
        
        tempogram_ratio.append(ratio)
    
    tempogram_ratio = np.stack(tempogram_ratio, axis=0)
    
    return tempogram_ratio
