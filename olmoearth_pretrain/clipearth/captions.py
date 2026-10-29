"""Captioned archive chips as training samples for ``clipearth.train``.

Inputs come from the clipearth v2 pipeline
(``/weka/dfive-default/agent/piperbots/2026-10-22-piperw-clipearth-v2``):

* a chips ``.npz`` (``sample_embed.py`` / ``select_diverse.py``) with per chip ``id``,
  ``lat``, ``lon``, ``chip`` (128) and ``quads`` (4, 128, row-major): means of unit pixel
  vectors in the space the Studio lab scores, stored as unit directions plus, in newer
  files, their lengths ``chip_norm`` and ``quads_norm``. Centring needs the lengths;
  without them the means are taken as unit length;
* a captions ``.jsonl`` (``caption.py``), one GPT-5.5 caption per chip id.

Each chip gives one sample against its chip embedding and one per quadrant against that
quadrant's embedding. Texts are grouped into the levels ``train`` samples from.
"""

from __future__ import annotations

import json
import re

import numpy as np

LEVELS = ("name", "short", "detailed")
QUADRANTS = ("top_left", "top_right", "bottom_left", "bottom_right")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def first_sentences(text: str, n: int = 2) -> str:
    """The first ``n`` sentences of ``text``."""
    return " ".join(_SENTENCE_END.split(text.strip())[:n])


def caption_levels(caption: dict) -> dict[str, list[str]]:
    """A caption's texts by level; detailed texts are at most two sentences."""
    levels = {
        "name": [caption.get("label", "")],
        "short": [caption.get("short", "")],
        "detailed": [
            caption.get("sentence", ""),
            first_sentences(caption.get("detailed", "")),
        ],
    }
    return {k: [t.strip() for t in v if t and t.strip()] for k, v in levels.items()}


def load_caption_samples(
    chips_path: str, captions_path: str, min_visibility: float = 0.5
) -> list[dict]:
    """Chip and quadrant samples for every usable caption.

    Returns dicts with ``group`` (unique per chip or quadrant), ``chip`` (the chip id),
    ``source`` ("chip" or "quad"), ``lat``, ``lon``, ``embedding``, ``context`` (the
    chip embedding, for quadrants too) and ``levels``.
    """
    # Read every array once: indexing the lazy NpzFile re-reads the whole array each time.
    with np.load(chips_path) as stored:
        chips = {key: stored[key] for key in stored.files}
    row = {str(c): i for i, c in enumerate(chips["id"])}
    chip_norm = chips["chip_norm"] if "chip_norm" in chips else np.ones(len(row))
    quads_norm = (
        chips["quads_norm"] if "quads_norm" in chips else np.ones((len(row), 4))
    )
    samples = []
    with open(captions_path) as f:
        for line in f:
            record = json.loads(line)
            caption = record.get("caption")
            if record.get("status") != "ok" or not caption or record["id"] not in row:
                continue
            if float(caption.get("visibility", 0)) < min_visibility:
                continue
            i = row[record["id"]]
            chip_embedding = chips["chip"][i].astype(np.float32) * chip_norm[i]
            base = {
                "chip": record["id"],
                "lat": float(chips["lat"][i]),
                "lon": float(chips["lon"][i]),
                "context": chip_embedding,
            }
            samples.append(
                {
                    **base,
                    "group": record["id"],
                    "source": "chip",
                    "embedding": chip_embedding,
                    "levels": caption_levels(caption),
                }
            )
            for q, name in enumerate(QUADRANTS):
                text = (caption.get("quadrants") or {}).get(name, "").strip()
                if text:
                    samples.append(
                        {
                            **base,
                            "group": f"{record['id']}/{name}",
                            "source": "quad",
                            "embedding": chips["quads"][i, q].astype(np.float32)
                            * quads_norm[i, q],
                            "levels": {"short": [text]},
                        }
                    )
    return samples


def geo_holdout(lat: float, lon: float, fraction: float, cell_deg: float = 5.0) -> bool:
    """Deterministically hold out ``fraction`` of ``cell_deg`` x ``cell_deg`` cells."""
    cell = (int(np.floor(lat / cell_deg)), int(np.floor(lon / cell_deg)))
    h = (cell[0] * 73856093) ^ (cell[1] * 19349663)
    return (h % 1000) < fraction * 1000
