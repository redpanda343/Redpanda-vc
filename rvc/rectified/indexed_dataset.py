import os
from pathlib import Path

import h5py
import numpy as np


class IndexedDataset:
    def __init__(self, path, recipe_hash=None):
        self.path = Path(path)
        self.recipe_hash = recipe_hash
        self.dset = None
        self.lookup = None

    def _open(self):
        if self.dset is not None:
            return
        self.dset = h5py.File(self.path, "r")
        if self.recipe_hash is not None and self.dset.attrs.get("recipe_hash", "") != self.recipe_hash:
            self.close()
            raise ValueError(f"Stale indexed training feature cache: {self.path}")
        keys = self.dset["keys"][()]
        self.lookup = {
            value.decode("ascii") if isinstance(value, (bytes, np.bytes_)) else str(value): index
            for index, value in enumerate(keys)
        }

    def close(self):
        if self.dset is not None:
            self.dset.close()
        self.dset = None
        self.lookup = None

    def __del__(self):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["dset"] = None
        state["lookup"] = None
        return state

    def __len__(self):
        self._open()
        return len(self.lookup)

    def __contains__(self, key):
        self._open()
        return key in self.lookup

    def __getitem__(self, key):
        self._open()
        try:
            index = self.lookup[key]
        except KeyError as error:
            raise KeyError(f"Training feature cache entry not found: {key}") from error
        group = self.dset["items"][str(index)]
        return {name: value[()] for name, value in group.items()}

    @staticmethod
    def matches(path, recipe_hash, keys):
        path = Path(path)
        if not path.is_file():
            return False
        try:
            with h5py.File(path, "r") as dset:
                if dset.attrs.get("recipe_hash", "") != recipe_hash:
                    return False
                stored = dset["keys"][()]
                if len(stored) != len(keys):
                    return False
                return all(
                    (value.decode("ascii") if isinstance(value, (bytes, np.bytes_)) else str(value)) == key
                    for value, key in zip(stored, keys)
                )
        except Exception:
            return False


class IndexedDatasetBuilder:
    def __init__(self, path, recipe_hash, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dset = h5py.File(self.path, "w")
        self.dset.attrs["recipe_hash"] = recipe_hash
        self.keys = self.dset.create_dataset("keys", shape=(size,), dtype="S64")
        self.items = self.dset.create_group("items")

    def add_item(self, index, key, item):
        self.keys[index] = np.bytes_(key)
        group = self.items.create_group(str(index))
        for name, value in item.items():
            if value is not None:
                group.create_dataset(name, data=value)

    def finalize(self):
        if self.dset is not None:
            self.dset.flush()
            self.dset.close()
            self.dset = None

    def abort(self):
        self.finalize()
        if self.path.exists():
            self.path.unlink()

    def __del__(self):
        if getattr(self, "dset", None) is not None:
            self.abort()
