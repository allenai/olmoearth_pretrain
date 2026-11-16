"""Held-out classes and locations for text targets, from a ClipEarth split spec.

The spec (JSON) names the held-out classes (``held_names``, already normalized with
norm_key) and the location rule: the world is cut into ``region_deg`` regions, a region
is held out when ``md5("i,j") % 1000 < held_fraction * 1000`` for its indices, and a point
outside held regions is in the buffer when its ``buffer_deg`` block's 8-neighbourhood
touches a held region. Training text targets drop held classes everywhere and every text
target in held-out and buffer locations; imagery is untouched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from olmoearth_pretrain.data.text_labels import LabelVocabulary, norm_key


def region_held(i: int, j: int, held_fraction: float) -> bool:
    """Whether region (i, j) is held out."""
    digest = hashlib.md5(f"{i},{j}".encode(), usedforsecurity=False).hexdigest()
    return int(digest, 16) % 1000 < held_fraction * 1000


@dataclass
class TextHoldout:
    """A split spec: held-out class names and the location rule."""

    held_names: frozenset[str]
    region_deg: float
    buffer_deg: float
    held_fraction: float

    @classmethod
    def load(cls, path: str) -> TextHoldout:
        """Read a split spec."""
        spec = json.load(open(path))
        return cls(
            held_names=frozenset(norm_key(n) for n in spec["held_names"]),
            region_deg=spec["region_deg"],
            buffer_deg=spec["buffer_deg"],
            held_fraction=spec["held_fraction"],
        )

    def _neighbours_held(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """[9, N] whether each point's buffer-block neighbours lie in a held region; row 4 is its own."""
        per = int(round(self.region_deg / self.buffer_deg))
        bi = np.floor(np.asarray(lat, dtype=np.float64) / self.buffer_deg).astype(
            np.int64
        )
        bj = np.floor(np.asarray(lon, dtype=np.float64) / self.buffer_deg).astype(
            np.int64
        )
        # One int64 key per region (j offset to stay non-negative) for a flat np.unique.
        keys = np.concatenate(
            [
                ((bi + da) // per << 32) + ((bj + db) // per + (1 << 31))
                for da in (-1, 0, 1)
                for db in (-1, 0, 1)
            ]
        )
        unique, inverse = np.unique(keys, return_inverse=True)
        held = np.array(
            [
                region_held(
                    int(k) >> 32, (int(k) & 0xFFFFFFFF) - (1 << 31), self.held_fraction
                )
                for k in unique
            ],
            bool,
        )
        return held[inverse].reshape(9, len(bi))

    def location_split(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """'held', 'buffer' or 'train' per point."""
        held = self._neighbours_held(lat, lon)
        return np.where(held[4], "held", np.where(held.any(0), "buffer", "train"))

    def masked_locations(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Whether each point is held out or in the buffer: no text targets there."""
        return self._neighbours_held(lat, lon).any(0)

    def held_label_ids(self, vocab: LabelVocabulary) -> np.ndarray:
        """Sorted ids of the vocabulary's held-out labels."""
        ids = {vocab.id(n) for n in self.held_names}
        return np.array(sorted(i for i in ids if i is not None), dtype=np.int64)
