"""Held-out classes and held-out locations, from one split spec shared with pretraining.

The spec (JSON) lists ``held_names`` (norm_key class names: each held-out class and its
aliases) and the location rule: a ``region_deg`` region is held when
md5("i,j") % 1000 < held_fraction * 1000, for i, j its floored lat/lon indices; a point
outside held regions is ``buffer`` when its ``buffer_deg`` block touches a held region
(8-neighbourhood). Held-class samples are held out wherever they are; buffer samples are
neither trained on nor scored.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

import numpy as np

HELD_CLASS = "held_class"
HELD_LOCATION = "held_location"
BUFFER = "buffer"
SPLITS = (HELD_CLASS, HELD_LOCATION, BUFFER)


def norm_key(text: str) -> str:
    """Lowercase alphanumerics separated by single spaces (as train.norm_key)."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


class SplitSpec:
    """Which classes and places are held out."""

    def __init__(
        self,
        held_names: Iterable[str],
        region_deg: float,
        buffer_deg: float,
        held_fraction: float,
    ):
        """Hold out ``held_names`` and the ``held_fraction`` of ``region_deg`` regions."""
        self.held_names = frozenset(norm_key(n) for n in held_names)
        self.region_deg = region_deg
        self.buffer_deg = buffer_deg
        self.held_fraction = held_fraction
        self._per = int(round(region_deg / buffer_deg))
        self._regions: dict[tuple[int, int], bool] = {}
        words = sorted(self.held_names, key=len, reverse=True)
        self._mention = (
            re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")s?\b")
            if words
            else None
        )

    @classmethod
    def load(cls, path: str) -> SplitSpec:
        """Read a spec JSON."""
        spec = json.load(open(path))
        return cls(
            spec["held_names"],
            spec["region_deg"],
            spec["buffer_deg"],
            spec["held_fraction"],
        )

    def class_held(self, name: str) -> bool:
        """Whether a class name is held out."""
        return norm_key(name) in self.held_names

    def mentions_held(self, text: str) -> bool:
        """Whether free text names a held-out class (whole words, plural-tolerant)."""
        return bool(self._mention and self._mention.search(norm_key(text)))

    def _region_held(self, i: int, j: int) -> bool:
        if (i, j) not in self._regions:
            digest = hashlib.md5(f"{i},{j}".encode(), usedforsecurity=False).hexdigest()
            self._regions[(i, j)] = int(digest, 16) % 1000 < self.held_fraction * 1000
        return self._regions[(i, j)]

    def _block_split(self, a: int, b: int) -> str:
        if self._region_held(a // self._per, b // self._per):
            return HELD_LOCATION
        if any(
            self._region_held((a + da) // self._per, (b + db) // self._per)
            for da in (-1, 0, 1)
            for db in (-1, 0, 1)
        ):
            return BUFFER
        return "train"

    def location(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """HELD_LOCATION, BUFFER or "train" per point."""
        blocks = np.stack(
            [
                np.floor(np.asarray(lat, np.float64) / self.buffer_deg),
                np.floor(np.asarray(lon, np.float64) / self.buffer_deg),
            ],
            1,
        ).astype(int)
        unique, inverse = np.unique(blocks, axis=0, return_inverse=True)
        per_block = np.array(
            [self._block_split(a, b) for a, b in unique.tolist()], dtype="<U16"
        )
        return per_block[inverse.reshape(-1)]

    def held_mask(self, names: np.ndarray) -> np.ndarray:
        """Whether each name is a held-out class."""
        unique, inverse = np.unique(np.asarray(names, str), return_inverse=True)
        return np.array([self.class_held(n) for n in unique], bool)[inverse.reshape(-1)]

    def assign(
        self, held: np.ndarray, lat: np.ndarray, lon: np.ndarray, base: np.ndarray
    ) -> np.ndarray:
        """Each sample's split: HELD_CLASS where ``held``, else its location split, else ``base``."""
        loc = self.location(lat, lon)
        return np.where(
            held, HELD_CLASS, np.where(loc == "train", np.asarray(base, str), loc)
        )
