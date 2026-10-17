import math

import faiss
import numpy as np


NEIGHBORS = 8
MAX_NPROBE = 16
REFERENCE_QUANTILE = 0.75
REFERENCE_FRAMES = 500.0


class FeatureIndex:
    def __init__(self, index):
        try:
            ivf = faiss.extract_index_ivf(index)
        except RuntimeError:
            ivf = None
        if ivf is not None:
            ivf.nprobe = max(1, min(int(ivf.nlist), MAX_NPROBE))
            ivf.make_direct_map()
        index.reconstruct(0)
        self.index = index
        self.dim = int(index.d)
        self.neighbors = max(1, min(NEIGHBORS, int(index.ntotal)))
        self.reference = None

    def reset(self):
        self.reference = None

    def _update_reference(self, nearest):
        level = math.log(float(np.quantile(nearest, REFERENCE_QUANTILE)))
        if self.reference is None:
            self.reference = level
        else:
            step = 1.0 - math.exp(-nearest.shape[0] / REFERENCE_FRAMES)
            self.reference += step * (level - self.reference)
        return math.exp(self.reference)

    def blend(self, query, rate):
        query = np.ascontiguousarray(query, dtype=np.float32)
        score, ids = self.index.search(query, k=self.neighbors)
        valid = ids >= 0
        if not valid.any(axis=1).all():
            raise RuntimeError("The selected feature index returned no valid neighbors.")
        score = np.maximum(score, 1e-6)
        weight = np.where(valid, np.square(1.0 / score), 0.0)
        weight /= weight.sum(axis=1, keepdims=True)
        neighbors = self.index.reconstruct_batch(np.where(valid, ids, 0).ravel())
        neighbors = neighbors.reshape(*ids.shape, -1)
        retrieved = np.sum(neighbors * np.expand_dims(weight, axis=2), axis=1)
        nearest = score[:, 0]
        reference = self._update_reference(nearest)
        amount = float(rate) * np.minimum(1.0, reference / nearest)
        blended = query + np.expand_dims(amount, axis=1) * (retrieved - query)
        return blended.astype(np.float32, copy=False)
