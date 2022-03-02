import librosa as lr
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from deepOstinato.preprocessing.constants import FRAME_SIZE, HOP_SIZE

class STFT(BaseEstimator, TransformerMixin):
    """short-time Fourier transform"""

    def __init__(self):
        pass

    def stft(audio, n_fft = FRAME_SIZE-1, hop_length = HOP_SIZE, transform="decibel"):
        """Returns the log short-time Fourier transform version of an audio file"""
        transform_type = {"decibel": lambda x: lr.power_to_db(x**2)} #power spectrogram (amplitude squared) to decibel (dB) units
        audio = np.array(audio)
        stft = lr.stft(audio, n_fft = FRAME_SIZE-1, hop_length = HOP_SIZE)
        stft = np.abs(stft) ** 2
        #Convert to selected unit (decibel or else)
        return transform_type[transform](stft)


class ISTFT:
    """inverse short_time Fourier transform"""

    def __init__(self):
        pass

    def istft(audio):

        inversed_audio = lr.istft(audio)
        return inversed_audio
