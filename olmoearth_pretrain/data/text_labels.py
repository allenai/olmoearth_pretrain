"""Per-pixel text labels: class names from the label layers and the open-set bank.

The ``text_label`` raster holds up to ``CHANNELS`` label ids per pixel, each an index into
one label vocabulary (``names.json`` in a vocabulary dir, embedded alongside it as
``embeddings.npy``). MISSING_VALUE means no label in that channel:

  0  land cover: the WorldCover class, or the open-set class for open-set windows
  1  crop: the CDL crop
  2  cereal: the most specific WorldCereal product above 50%
  3  OSM: one feature, preferring rare objects over buildings and roads

Pretraining windows get the raster at load time from their raw label layers; converted
open-set windows store it. Generic names ("background", "other") and absence classes
("non-water", "not forest") are never labels.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from olmoearth_pretrain.data.constants import MISSING_VALUE

CHANNELS = 4
LAND_COVER, CROP, CEREAL, OSM_CHANNEL = range(CHANNELS)

WORLDCOVER_NAMES = {
    10: "tree cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "built-up",
    60: "bare or sparse vegetation",
    70: "snow and ice",
    80: "open water",
    90: "herbaceous wetland",
    95: "mangroves",
    100: "moss and lichen",
}
# Band order of the openstreetmap_raster modality.
OSM_NAMES = [
    "aerialway pylon",
    "aerodrome",
    "airstrip",
    "fuel station",
    "building",
    "chimney",
    "communications tower",
    "crane",
    "flagpole",
    "fountain",
    "wind turbine",
    "helipad",
    "road",
    "leisure area",
    "lighthouse",
    "obelisk",
    "observatory",
    "parking",
    "petroleum well",
    "power plant",
    "power substation",
    "power tower",
    "river",
    "runway",
    "satellite dish",
    "silo",
    "storage tank",
    "taxiway",
    "water tower",
    "industrial works",
]
# Where several OSM features share a pixel, the earliest in this list wins.
OSM_LAST = ["leisure area", "building", "road"]
# Band order of the worldcereal modality; values are 0-100 confidences.
WORLDCEREAL_NAMES = [
    "temporary crops",
    "irrigated maize",
    "maize",
    "irrigated second-season maize",
    "second-season maize",
    "spring cereals",
    "irrigated winter cereals",
    "winter cereals",
]
# USDA Cropland Data Layer crop codes; its land-cover codes are left to WorldCover.
CDL_NAMES = {
    1: "corn",
    2: "cotton",
    3: "rice",
    4: "sorghum",
    5: "soybeans",
    6: "sunflower",
    10: "peanuts",
    11: "tobacco",
    12: "sweet corn",
    13: "popcorn",
    14: "mint",
    21: "barley",
    22: "durum wheat",
    23: "spring wheat",
    24: "winter wheat",
    25: "other small grains",
    26: "winter wheat double-cropped with soybeans",
    27: "rye",
    28: "oats",
    29: "millet",
    30: "speltz",
    31: "canola",
    32: "flaxseed",
    33: "safflower",
    34: "rapeseed",
    35: "mustard",
    36: "alfalfa",
    37: "hay",
    38: "camelina",
    39: "buckwheat",
    41: "sugar beets",
    42: "dry beans",
    43: "potatoes",
    44: "other crops",
    45: "sugarcane",
    46: "sweet potatoes",
    47: "vegetables and fruit",
    48: "watermelons",
    49: "onions",
    50: "cucumbers",
    51: "chickpeas",
    52: "lentils",
    53: "peas",
    54: "tomatoes",
    55: "caneberries",
    56: "hops",
    57: "herbs",
    58: "clover and wildflowers",
    59: "sod and grass seed",
    60: "switchgrass",
    61: "fallow cropland",
    66: "cherries",
    67: "peaches",
    68: "apples",
    69: "grapes",
    70: "Christmas trees",
    71: "tree crops",
    72: "citrus",
    74: "pecans",
    75: "almonds",
    76: "walnuts",
    77: "pears",
    92: "aquaculture",
    176: "grassland or pasture",
    204: "pistachios",
    205: "triticale",
    206: "carrots",
    207: "asparagus",
    208: "garlic",
    209: "cantaloupes",
    210: "prunes",
    211: "olives",
    212: "oranges",
    213: "honeydew melons",
    214: "broccoli",
    215: "avocados",
    216: "peppers",
    217: "pomegranates",
    218: "nectarines",
    219: "greens",
    220: "plums",
    221: "strawberries",
    222: "squash",
    223: "apricots",
    224: "vetch",
    225: "winter wheat double-cropped with corn",
    226: "oats double-cropped with corn",
    227: "lettuce",
    228: "triticale double-cropped with corn",
    229: "pumpkins",
    236: "winter wheat double-cropped with sorghum",
    237: "barley double-cropped with corn",
    238: "winter wheat double-cropped with cotton",
    241: "corn double-cropped with soybeans",
    242: "blueberries",
    243: "cabbage",
    244: "cauliflower",
    245: "celery",
    246: "radishes",
    247: "turnips",
    248: "eggplants",
    249: "gourds",
    250: "cranberries",
    254: "barley double-cropped with soybeans",
}
GENERIC_LABELS = {
    "background",
    "other",
    "others",
    "unknown",
    "unlabeled",
    "unclassified",
    "no data",
    "nodata",
    "none",
    "non field",
    "not field",
    "void",
    "ignore",
}


def norm_key(text: str) -> str:
    """Lowercase alphanumerics separated by single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def is_generic(key: str) -> bool:
    """Whether a normalized name says nothing about what is there."""
    return key in GENERIC_LABELS or key.startswith(("non ", "not "))


def layer_label_names() -> list[str]:
    """Every name the pretraining label layers can produce."""
    return [
        *WORLDCOVER_NAMES.values(),
        *CDL_NAMES.values(),
        *WORLDCEREAL_NAMES,
        *OSM_NAMES,
    ]


class LabelVocabulary:
    """Label names and their ids; names that normalize alike share an id."""

    def __init__(self, names: list[str]):
        """Build from names in id order; generic and repeated names are dropped.

        Names are kept as text to embed, with underscores read as spaces.
        """
        self.names: list[str] = []
        self._ids: dict[str, int] = {}
        for name in names:
            key = norm_key(name)
            if key and not is_generic(key) and key not in self._ids:
                self._ids[key] = len(self.names)
                self.names.append(" ".join(name.replace("_", " ").split()))
        self._layer_ids = _layer_ids(self)

    @classmethod
    def load(cls, path: str) -> LabelVocabulary:
        """Read ``names.json`` from a vocabulary dir."""
        return cls(json.load(open(os.path.join(path, "names.json"))))

    def save(self, path: str) -> None:
        """Write ``names.json`` to a vocabulary dir."""
        os.makedirs(path, exist_ok=True)
        json.dump(self.names, open(os.path.join(path, "names.json"), "w"), indent=0)

    def __len__(self) -> int:
        """The number of labels."""
        return len(self.names)

    def id(self, name: str) -> int | None:
        """The id of ``name``, or None for a generic or unknown name."""
        return self._ids.get(norm_key(name))

    def layer_raster(self, layers: dict[str, np.ndarray]) -> np.ndarray | None:
        """[H, W, 1, CHANNELS] float32 label ids from raw label layers, or None if none.

        ``layers`` holds whichever of the raw ``worldcover`` ([H, W, 1, 1] codes),
        ``cdl`` (codes), ``worldcereal`` ([H, W, 1, 8] 0-100) and
        ``openstreetmap_raster`` ([H, W, 1, 30] presence) the window has.
        """
        shape = next(iter(layers.values())).shape[:2] if layers else None
        if shape is None:
            return None
        out = np.full((*shape, 1, CHANNELS), MISSING_VALUE, dtype=np.float32)
        ids = self._layer_ids
        if "worldcover" in layers:
            out[..., 0, LAND_COVER] = ids["worldcover"][_codes(layers["worldcover"])]
        if "cdl" in layers:
            out[..., 0, CROP] = ids["cdl"][_codes(layers["cdl"])]
        if "worldcereal" in layers:
            on = layers["worldcereal"][..., 0, :] > 50
            # Specific products first, then "temporary crops".
            order = [*range(1, len(WORLDCEREAL_NAMES)), 0]
            out[..., 0, CEREAL] = _first_on(on, order, ids["worldcereal"])
        if "openstreetmap_raster" in layers:
            on = layers["openstreetmap_raster"][..., 0, :] > 0
            out[..., 0, OSM_CHANNEL] = _first_on(on, _OSM_ORDER, ids["osm"])
        return out if (out != MISSING_VALUE).any() else None


_OSM_ORDER = [i for i, n in enumerate(OSM_NAMES) if n not in OSM_LAST] + [
    OSM_NAMES.index(n) for n in OSM_LAST
]


def _codes(layer: np.ndarray) -> np.ndarray:
    """Raw [H, W, 1, 1] class codes as table indices; out-of-range codes map to 0."""
    codes = layer[..., 0, 0].astype(np.int64)
    return np.where((codes >= 0) & (codes < 256), codes, 0)


def _first_on(on: np.ndarray, order: list[int], ids: np.ndarray) -> np.ndarray:
    """Per pixel, the id of the first band in ``order`` that is on."""
    out = np.full(on.shape[:-1], MISSING_VALUE, dtype=np.float32)
    for band in reversed(order):
        out = np.where(on[..., band], ids[band], out)
    return out


def _layer_ids(vocab: LabelVocabulary) -> dict[str, np.ndarray]:
    """Lookup arrays from raw layer values (or band index) to label ids."""

    def code_table(names: dict[int, str]) -> np.ndarray:
        table = np.full(256, MISSING_VALUE, dtype=np.float32)
        for code, name in names.items():
            table[code] = _id_or_missing(vocab, name)
        return table

    return {
        "worldcover": code_table(WORLDCOVER_NAMES),
        "cdl": code_table(CDL_NAMES),
        "worldcereal": np.array(
            [_id_or_missing(vocab, n) for n in WORLDCEREAL_NAMES], dtype=np.float32
        ),
        "osm": np.array(
            [_id_or_missing(vocab, n) for n in OSM_NAMES], dtype=np.float32
        ),
    }


def _id_or_missing(vocab: LabelVocabulary, name: str) -> float:
    label = vocab._ids.get(norm_key(name))
    return float(MISSING_VALUE if label is None else label)


def embed_vocabulary(
    path: str, model_name: str, prompt_name: str | None, batch_size: int = 512
) -> None:
    """Write ``embeddings.npy`` ([labels, dim] float32, unit) and ``meta.json``."""
    import torch
    from sentence_transformers import SentenceTransformer

    vocab = LabelVocabulary.load(path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={
            "torch_dtype": torch.bfloat16 if device == "cuda" else torch.float32
        },
    )
    emb = model.encode(
        vocab.names,
        prompt_name=prompt_name,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    np.save(os.path.join(path, "embeddings.npy.tmp.npy"), emb)
    os.replace(
        os.path.join(path, "embeddings.npy.tmp.npy"),
        os.path.join(path, "embeddings.npy"),
    )
    meta = {"model": model_name, "prompt_name": prompt_name, "dim": emb.shape[1]}
    json.dump(meta, open(os.path.join(path, "meta.json"), "w"), indent=1)


def main() -> None:
    """Embed a label vocabulary dir: ``python -m olmoearth_pretrain.data.text_labels``."""
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("vocab_dir")
    p.add_argument("--model", default="google/embeddinggemma-300m")
    p.add_argument("--prompt_name", default="document")
    a = p.parse_args()
    embed_vocabulary(a.vocab_dir, a.model, a.prompt_name or None)


if __name__ == "__main__":
    main()
