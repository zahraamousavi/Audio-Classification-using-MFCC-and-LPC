import numpy as np


class Distributions:
    def __init__(self) -> None:
        super().__init__()

    def gaussian_distribution(self, signal, mean, variance):
        gaussian = []
        # for instance in signal:
        #     instance = int(instance)

        gaussian = (1 / (variance * np.sqrt(2 * np.pi))) * (np.exp((((signal - mean) / variance) ** 2) * (-0.5)))
        return gaussian
