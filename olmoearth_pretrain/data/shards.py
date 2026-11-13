"""Byte-packed shard datasets: ``<dir>/shard_index.npz`` plus ``shards/shard_*.bin``.

A pretraining sample file is a self-contained HDF5 file; a packed dataset concatenates those bytes verbatim into a
few large files and stores each sample's ``(shard, offset, nbytes)`` in ``shard_index.npz`` (writer:
``geochameleon``'s ``scripts/pretraining_data/python/pack_shards.py``, PLAN #205 §2.5). Reading a sample is then one
contiguous range read served to ``h5py.File`` through :class:`ShardSlice`, a bounded read-only file-like — no
re-encoding, byte-identical to the original file.

The dataset keeps its directory contract (count-named directory, ``sample_metadata.csv``,
``latlon_distribution.npy``); only the ``sample_<i>.h5`` files move into shards. ``OlmoEarthDataset`` detects the
index next to the sidecars and reads through it; unpacked datasets are untouched.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

INDEX_FNAME = "shard_index.npz"
SHARD_DIR = "shards"
SHARD_NAME = "shard_{shard:04d}.bin"


class ShardSlice(io.RawIOBase):
    """A read-only view of ``[offset, offset + nbytes)`` inside ``path``, as a file-like for ``h5py.File``."""

    def __init__(self, path: Path, offset: int, nbytes: int) -> None:
        super().__init__()
        self._handle = open(path, "rb")
        self._handle.seek(offset)
        self._start, self._nbytes, self._pos = offset, nbytes, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, cookie: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = cookie
        elif whence == io.SEEK_CUR:
            position = self._pos + cookie
        else:
            position = self._nbytes + cookie
        self._pos = max(0, position)
        self._handle.seek(self._start + self._pos)
        return self._pos

    def readinto(self, buffer) -> int:  # type: ignore[no-untyped-def]
        remaining = self._nbytes - self._pos
        if remaining <= 0:
            return 0
        data = self._handle.read(min(remaining, len(buffer)))
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def read(self, size: int = -1) -> bytes:  # io.RawIOBase's read is small-chunked; read whole asks here instead
        remaining = self._nbytes - self._pos
        if size is None or size < 0:
            size = remaining
        data = self._handle.read(max(0, min(size, remaining)))
        self._pos += len(data)
        return data

    def close(self) -> None:
        if not self.closed:
            self._handle.close()
        super().close()


class ShardIndex:
    """``<root>/shard_index.npz``: sample ``i`` lives at ``(shard, offset, nbytes)`` of the shard directory."""

    def __init__(self, root: Path | str) -> None:
        root = Path(str(root))
        with np.load(root / INDEX_FNAME) as arrays:
            self._shard = np.asarray(arrays["shard"])
            self._offset = np.asarray(arrays["offset"])
            self._nbytes = np.asarray(arrays["nbytes"])
        self.root = root

    def __len__(self) -> int:
        return int(self._shard.size)

    def open(self, sample_index: int) -> ShardSlice:
        return ShardSlice(
            self.root / SHARD_DIR / SHARD_NAME.format(shard=int(self._shard[sample_index])),
            int(self._offset[sample_index]),
            int(self._nbytes[sample_index]),
        )
