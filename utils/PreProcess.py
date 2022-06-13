import numpy as np


class PreProcess:
    def __init__(self) -> None:
        super().__init__()

    def align_signal(self, signal, max_len):
        aligned_signal = np.zeros(max_len)
        diff = max_len - len(signal)
        aligned_signal[int(diff / 2): int(diff / 2) + len(signal)] = signal
        return aligned_signal