import math

import faiss
import numpy as np
import torch


NEIGHBORS = 8
MAX_NPROBE = 16
REFERENCE_QUANTILE = 0.75
REFERENCE_FRAMES = 500.0
SEARCH_BYTES = 64 << 20
GPU_HEADROOM = 512 << 20


class TorchFeatureSearch:
    def __init__(self, index, ivf, device):
        self.device = device
        self.dim = int(index.d)
        self.count = int(index.ntotal)
        self.centers = None
        if index.metric_type != faiss.METRIC_L2:
            raise ValueError("GPU feature retrieval requires an L2 feature index.")
        if not isinstance(index, (faiss.IndexIVFFlat, faiss.IndexFlat)):
            raise ValueError("GPU feature retrieval supports Flat and IVF Flat feature indexes.")
        center_count = int(ivf.nlist) if ivf is not None else 0
        required = (self.count + center_count + 1) * self.dim * 4 + self.count * 12
        free, _ = torch.cuda.mem_get_info(device)
        available = free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if required + SEARCH_BYTES + GPU_HEADROOM > available:
            raise torch.OutOfMemoryError(
                f"The feature index needs {required / (1 << 20):.0f} MiB of GPU memory plus working space. "
                "Use a smaller index or select a device with more free memory."
            )
        if ivf is None:
            vectors = index.reconstruct_n(0, self.count)
        else:
            sizes = np.array([ivf.invlists.list_size(i) for i in range(ivf.nlist)], dtype=np.int64)
            offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(sizes)))
            if offsets[-1] != self.count or ivf.code_size != self.dim * 4:
                raise ValueError("The selected IVF feature index has an invalid vector layout.")
            vectors = np.empty((self.count, self.dim), dtype=np.float32)
            for number, size in enumerate(sizes):
                if size:
                    codes = ivf.invlists.get_codes(number)
                    try:
                        values = faiss.rev_swig_ptr(codes, int(size) * ivf.code_size)
                        vectors[offsets[number]:offsets[number + 1]] = values.view(np.float32).reshape(-1, self.dim)
                    finally:
                        ivf.invlists.release_codes(number, codes)
            self.centers = torch.as_tensor(ivf.quantizer.reconstruct_n(0, ivf.nlist), device=device)
            self.list_ids = torch.repeat_interleave(
                torch.arange(ivf.nlist, device=device), torch.as_tensor(sizes, device=device), output_size=self.count
            )
            self.nprobe = int(ivf.nprobe)
        self.vectors = torch.as_tensor(vectors, device=device)
        self.origin = self.vectors.mean(dim=0)
        self.vectors.sub_(self.origin)
        self.norms = torch.linalg.vector_norm(self.vectors, dim=1).square_()

    def search(self, query, neighbors):
        scores = torch.full((query.shape[0], neighbors), float("inf"), device=self.device)
        positions = torch.zeros(query.shape[0], neighbors, device=self.device, dtype=torch.long)
        candidates = min(max(64, neighbors * 4), self.count)
        per_query = self.count * 5 + candidates * self.dim * 4
        batch = max(1, min(64, SEARCH_BYTES // max(1, per_query)))
        for start in range(0, query.shape[0], batch):
            source = query[start:start + batch]
            centered = source - self.origin
            distances = torch.addmm(self.norms[None], centered, self.vectors.T, beta=1, alpha=-2)
            if self.centers is not None:
                coarse = torch.cdist(source, self.centers, compute_mode="donot_use_mm_for_euclid_dist")
                lists = coarse.topk(self.nprobe, dim=1, largest=False).indices
                allowed = torch.zeros(source.shape[0], self.centers.shape[0], dtype=torch.bool, device=self.device)
                allowed.scatter_(1, lists, True)
                distances.masked_fill_(~allowed[:, self.list_ids], float("inf"))
            approximate, selected = distances.topk(candidates, dim=1, largest=False)
            values = self.vectors[selected]
            exact = torch.cdist(centered[:, None], values, compute_mode="donot_use_mm_for_euclid_dist").squeeze(1).square_()
            exact.masked_fill_(~torch.isfinite(approximate), float("inf"))
            best, order = exact.topk(neighbors, dim=1, largest=False)
            scores[start:start + batch] = best
            positions[start:start + batch] = selected.gather(1, order)
        return scores, positions


class FeatureIndex:
    def __init__(self, index, device="cpu"):
        try:
            ivf = faiss.extract_index_ivf(index)
        except RuntimeError:
            ivf = None
        if ivf is not None:
            ivf.nprobe = max(1, min(int(ivf.nlist), MAX_NPROBE))
            ivf.make_direct_map()
        index.reconstruct(0)
        self.dim = int(index.d)
        self.neighbors = max(1, min(NEIGHBORS, int(index.ntotal)))
        self.device = torch.device(device)
        self.gpu = TorchFeatureSearch(index, ivf, self.device) if self.device.type == "cuda" else None
        self.index = index if self.gpu is None else None
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

    def _blend_cpu(self, query, rate):
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

    def _blend_gpu(self, query, rate):
        score, positions = self.gpu.search(query, self.neighbors)
        valid = torch.isfinite(score)
        if not bool(valid.any(dim=1).all()):
            raise RuntimeError("The selected feature index returned no valid neighbors.")
        score = score.clamp_min_(1e-6)
        weight = score.reciprocal().square_()
        weight /= weight.sum(dim=1, keepdim=True)
        retrieved = (self.gpu.vectors[positions] * weight[..., None]).sum(dim=1) + self.gpu.origin
        nearest = score[:, 0]
        level = torch.quantile(nearest.double(), REFERENCE_QUANTILE).log_()
        if self.reference is None:
            self.reference = level
        else:
            step = 1.0 - math.exp(-nearest.shape[0] / REFERENCE_FRAMES)
            self.reference += step * (level - self.reference)
        amount = float(rate) * (self.reference.exp().float() / nearest).clamp_max_(1.0)
        return query + amount[:, None] * (retrieved - query)

    @torch.inference_mode()
    def blend(self, query, rate):
        tensor_input = torch.is_tensor(query)
        if query.ndim != 2 or query.shape[1] != self.dim:
            raise ValueError(f"The feature index expects a query with {self.dim} channels.")
        if query.shape[0] == 0:
            return query.clone() if tensor_input else np.asarray(query, dtype=np.float32).copy()
        if self.gpu is not None:
            with torch.autocast("cuda", enabled=False):
                source = torch.as_tensor(query, device=self.device, dtype=torch.float32)
                result = self._blend_gpu(source, rate)
            return result.to(device=query.device, dtype=query.dtype) if tensor_input else result.cpu().numpy()
        source = query.detach().cpu().numpy() if tensor_input else query
        result = self._blend_cpu(source, rate)
        return torch.from_numpy(result).to(device=query.device, dtype=query.dtype) if tensor_input else result
