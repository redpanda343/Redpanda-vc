from pathlib import Path

import h5py
import torch


class IndexedDataset:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f'IndexedDataset not found: {self.path}')
        self.dset = None

    def __del__(self):
        if self.dset:
            self.dset.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['dset'] = None
        return state

    def __getitem__(self, i):
        if self.dset is None:
            self.dset = h5py.File(self.path, 'r')
        return {k: v[()].item() if v.shape == () else torch.from_numpy(v[()]) for k, v in self.dset[str(i)].items()}


class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dset = h5py.File(self.path, 'w')
        self.counter = 0

    def add_item(self, item):
        item_no = self.counter
        self.counter += 1
        for k, v in item.items():
            if v is not None:
                self.dset.create_dataset(f'{item_no}/{k}', data=v)
        return item_no

    def finalize(self):
        self.dset.close()
