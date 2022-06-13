import librosa
import numpy as np


class FeatureExtractor:
    def __init__(self) -> None:
        super().__init__()

    def get_power(self, signal):
        energy, power = 0, 0
        M = len(signal)
        for instance in signal:
            instance = int(instance)
            energy += (instance ** 2)
        power = energy / len(signal)
        return power

    def get_zcr(self, signal):
        zcr = 0
        for i in range(1, len(signal)):
            zcr += abs(self.Sign(signal[i]) - self.Sign(signal[i - 1]))
        return zcr

    def Sign(self, instance):
        if instance >= 1:
            return 1
        else:
            return -1

    def get_mfcc(self, signal):
        mfcc = librosa.feature.mfcc(signal, n_mfcc=12, n_mels=24, win_length=20)
        mfcc_delta = librosa.feature.delta(mfcc, order=1)
        mfcc_delta_delta = librosa.feature.delta(mfcc, order=2)
        final = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta))
        return final

    def get_lpc(self, signal):
        return librosa.lpc(signal, order=14)
