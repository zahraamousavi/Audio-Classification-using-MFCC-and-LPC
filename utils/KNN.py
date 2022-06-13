import numpy as np


class KNN:
    def __init__(self) -> None:
        super().__init__()

    def fit(self, dist_matt, k):
        final_res = []
        min_dist = np.argsort(dist_matt)[:, :k]
        result = (min_dist / 50).astype(int)
        for item in result:
            final_res.append(np.bincount(item).argmax())
        return final_res
