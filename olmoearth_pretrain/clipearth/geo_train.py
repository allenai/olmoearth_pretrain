"""Location-conditioned text head ("v5-geo"): (text, lat, lon, year, scale) -> v1.3 space.

The text side of a native aligner (``clipearth.train --native``) maps a frozen text
embedding into OlmoEarth's 128-d archive space, where it is scored by cosine against
the raw unit pixel vectors. Here that map also sees where and when it is used:

  a     = W1 t + b1                                   (t: 768-d text embedding)
  gamma, beta = split(F2 gelu(F1 c + f1) + f2)        (c: condition vector, below)
  y     = W2 gelu(a * (1 + gamma) + beta) + b2        (FiLM on the hidden layer)

``c`` = [sin(B p), cos(B p), unknown location, one-hot year (2017-2025, unknown),
one-hot scale (pixel, area)], with ``p`` the unit 3D vector of (lat, lon) and ``B`` fixed
random directions at a few wavelengths (``WAVELENGTHS_KM``). The FiLM output layer starts
at zero and the text layers start from a trained native aligner, so training begins
exactly at that aligner.

Location shortcut guard. With plain in-batch CLIP each text carries its own pair's
location, so a model can match pairs by location alone. Here every image is scored
only against texts conditioned on *its own* condition (the batch's texts, re-conditioned),
so location cannot tell the candidates apart. Pixel samples are drawn in cells of
nearby samples (``--cell_deg``) with different labels; each one's cell-mates' texts are
hard negatives at the same place, and text-to-image is computed only within a cell,
where every candidate image shares the location.

Samples: pixel samples (``v5/FORMAT.md``) at scale "pixel"; GPT-5.5 captions against
their chip's 2.56 km mean at scale "area"; optionally OSM region means at scale "area".

Run:
  python -m olmoearth_pretrain.clipearth.geo_train train --init <v4>/aligner.pt
      --pixels a.npz,b.npz --pixel_texts a.jsonl,b.jsonl --out_dir <dir>
  python -m olmoearth_pretrain.clipearth.geo_train export --head <dir>/geo_head.pt
      --browser_model <hf repo> --out textHeadGeo.json --parity parity.json
"""

from __future__ import annotations

import argparse
import base64
import copy as copy_module
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from olmoearth_pretrain.clipearth.train import (
    LEVEL_FALLBACK,
    encode_texts,
    load_pixel_samples,
)

logger = logging.getLogger(__name__)

YEARS = tuple(range(2017, 2026))
SCALES = ("pixel", "area")
LEVELS = ("name", "short", "detailed")
EARTH_RADIUS_KM = 6371.0
# Continental to ~10 km. Below ~10 km the head would vary within one 2.56 km view and
# could memorize single labelled windows.
WAVELENGTHS_KM = (4000.0, 1500.0, 500.0, 150.0, 50.0, 12.0)


def unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Unit 3D vectors of (lat, lon) in degrees, float64."""
    lat, lon = np.radians(lat), np.radians(lon)
    return np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1
    )


def make_frequencies(
    wavelengths_km: tuple[float, ...] = WAVELENGTHS_KM,
    per_scale: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """``[len(wavelengths) * per_scale, 3]`` random directions times angular frequency.

    ``sin(w d . p)`` on the unit sphere has wavelength ``2 pi R / w`` along a great
    circle in direction ``d``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for wavelength in wavelengths_km:
        d = rng.standard_normal((per_scale, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        rows.append(d * (2 * math.pi * EARTH_RADIUS_KM / wavelength))
    return np.concatenate(rows)


def condition_dim(frequencies: np.ndarray) -> int:
    """Width of the condition vector."""
    return 2 * len(frequencies) + 1 + len(YEARS) + 1 + len(SCALES)


def condition(
    lat: np.ndarray,
    lon: np.ndarray,
    year: np.ndarray,
    scale: np.ndarray,
    frequencies: np.ndarray,
) -> np.ndarray:
    """Condition vectors ``[N, condition_dim]``, float32.

    A NaN latitude or longitude means the location is unknown (its features are zero
    and the flag is 1); a year outside ``YEARS`` is the unknown year. ``scale`` indexes
    ``SCALES``.
    """
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    known = np.isfinite(lat) & np.isfinite(lon)
    p = unit_xyz(np.where(known, lat, 0.0), np.where(known, lon, 0.0))
    phase = p @ frequencies.T
    loc = np.concatenate([np.sin(phase), np.cos(phase)], -1) * known[:, None]
    year = np.asarray(year)
    year_index = np.where(np.isin(year, YEARS), year - YEARS[0], len(YEARS)).astype(int)
    years = np.eye(len(YEARS) + 1)[year_index]
    scales = np.eye(len(SCALES))[np.asarray(scale, int)]
    return np.concatenate(
        [loc, (~known)[:, None].astype(np.float64), years, scales], -1
    ).astype(np.float32)


class GeoTextHead(nn.Module):
    """The native text head with its hidden layer modulated by the condition."""

    def __init__(
        self,
        text_dim: int,
        hidden: int,
        out_dim: int,
        frequencies: np.ndarray,
        cond_hidden: int = 256,
    ) -> None:
        """Build the layers; the FiLM output starts at zero."""
        super().__init__()
        self.register_buffer(
            "frequencies", torch.from_numpy(np.asarray(frequencies, np.float64))
        )
        # Same layout and state-dict keys as ``train.Aligner.text`` in native mode.
        self.text = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )
        self.film = nn.Sequential(
            nn.Linear(condition_dim(frequencies), cond_hidden),
            nn.GELU(),
            nn.Linear(cond_hidden, 2 * hidden),
        )
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)
        self.logit_scale = nn.Parameter(
            torch.tensor(np.log(1 / 0.07), dtype=torch.float32)
        )

    def hidden(self, text: Tensor) -> Tensor:
        """First layer, before modulation: ``[..., hidden]``."""
        return self.text[0](text)

    def modulation(self, cond: Tensor) -> tuple[Tensor, Tensor]:
        """FiLM scale and shift ``gamma, beta`` per condition."""
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return gamma, beta

    def output(self, hidden: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        """Unnormalized output of modulated hidden activations."""
        return self.text[2](F.gelu(hidden * (1 + gamma) + beta))

    def forward(self, text: Tensor, cond: Tensor, dims: int | None = None) -> Tensor:
        """Normalized prediction, one text per condition row."""
        gamma, beta = self.modulation(cond)
        y = self.output(self.hidden(text), gamma, beta)
        return F.normalize(y[..., :dims], dim=-1)


def load_init(model: GeoTextHead, path: str) -> dict:
    """Copy the text layers and temperature of a native aligner; returns its args."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not ckpt["args"].get("native"):
        raise ValueError("--init needs a native aligner")
    state = {
        k: v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("text.") or k == "logit_scale"
    }
    missing, _ = model.load_state_dict(state, strict=False)
    assert all(k.startswith("film.") or k == "frequencies" for k in missing), missing
    return ckpt["args"]


def native_unit(x: Tensor, dims: int | None) -> Tensor:
    """Leading ``dims`` of image vectors, renormalized, as the lab reads them."""
    return F.normalize(x[..., :dims].float(), dim=-1)


def pair_logits(
    model: GeoTextHead,
    hidden: Tensor,
    cand: Tensor,
    cond: Tensor,
    image: Tensor,
    prefixes: list[int | None],
) -> list[Tensor]:
    """Cosine logits of each image against its candidate texts under its condition.

    Args:
        model: the geo head.
        hidden: ``[U, H]`` first-layer activations of the unique texts.
        cand: ``[B, K]`` indices into ``hidden`` (pads already replaced by any index).
        cond: ``[B, C]`` condition of each image.
        image: ``[B, D]`` image vectors.
        prefixes: Matryoshka widths to score (None for all).

    Returns one ``[B, K]`` tensor per Matryoshka prefix, unscaled.
    """
    gamma, beta = model.modulation(cond)
    y = model.output(hidden[cand], gamma[:, None], beta[:, None])  # [B, K, D]
    return [
        (F.normalize(y[..., :dims], dim=-1) * native_unit(image, dims)[:, None]).sum(-1)
        for dims in prefixes
    ]


def batch_loss(
    logits: Tensor, slot_text: Tensor, target: Tensor, cell_size: int, n_cells: int
) -> tuple[Tensor, Tensor | None]:
    """Image-to-text and within-cell text-to-image losses.

    Rows are ``n_cells * cell_size`` images in cell order (padding rows included);
    the first ``cell_size`` candidate slots of every row are the texts of its cell's
    members in member order, so ``logits[:, :cell_size]`` reshaped to
    ``[cell, image q, text j]`` scores member ``j``'s text on member ``q``.

    Args:
        logits: ``[B, K]`` scaled logits.
        slot_text: ``[B, K]`` text id per slot, -1 for an unusable slot (padding,
            or a repeat of an earlier slot's text).
        target: ``[B]`` slot of each row's own text, -1 for a padding row.
        cell_size: members per cell.
        n_cells: number of cells.
    """
    valid = slot_text >= 0
    rows = target >= 0
    masked = logits.masked_fill(~valid, float("-inf"))
    image_to_text = F.cross_entropy(masked[rows], target[rows])

    k = cell_size
    scores = masked[:, :k].reshape(n_cells, k, k)  # [cell, q, j]
    own = slot_text.gather(1, target.clamp(min=0)[:, None])[:, 0].reshape(n_cells, k)
    member = rows.reshape(n_cells, k)
    text_j = slot_text[:, :k].reshape(n_cells, k, k)[:, 0]  # cell texts, any row
    # Row 0 of a cell is always a real member, so its first k slots list the cell.
    positive = (own[:, :, None] == text_j[:, None, :]) & member[:, :, None]
    distinct = (text_j >= 0).sum(1) >= 2
    columns = (text_j >= 0) & distinct[:, None]
    if not columns.any():
        return image_to_text, None
    by_column = (
        scores.masked_fill(~member[:, :, None], float("-inf"))
        .transpose(1, 2)
        .log_softmax(-1)
    )  # [cell, j, q]
    pos = positive.transpose(1, 2)
    per_column = -(by_column.masked_fill(~pos, 0).sum(-1)) / pos.sum(-1).clamp(min=1)
    return image_to_text, per_column[columns].mean()


class Samples:
    """All training images with their condition fields and text groups."""

    def __init__(self) -> None:
        """Start empty."""
        self.text_index: dict[str, int] = {}
        self.group_levels: list[dict[str, list[int]]] = []
        self.parts: dict[str, list] = {
            k: []
            for k in (
                "image",
                "lat",
                "lon",
                "year",
                "scale",
                "group",
                "weight",
                "origin",
                "cellable",
            )
        }
        self.origins: list[str] = []
        self._group_key: dict[str, int] = {}

    def group(self, by_level: dict[str, list[str]]) -> int:
        """Group id of a text set; identical sets share one group."""
        key = json.dumps(by_level, sort_keys=True)
        if key not in self._group_key:
            self.group_levels.append(
                {
                    lv: [
                        self.text_index.setdefault(t, len(self.text_index)) for t in ts
                    ]
                    for lv, ts in by_level.items()
                    if ts
                }
            )
            self._group_key[key] = len(self.group_levels) - 1
        return self._group_key[key]

    def add(
        self,
        name: str,
        image: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
        year: np.ndarray,
        scale: str,
        groups: np.ndarray,
        weight: float,
        cellable: bool,
    ) -> None:
        """Append one source's samples."""
        n = len(image)
        p = self.parts
        p["image"].append(
            (image / np.linalg.norm(image, axis=1, keepdims=True).clip(1e-6)).astype(
                np.float16
            )
        )
        p["lat"].append(np.asarray(lat, np.float64))
        p["lon"].append(np.asarray(lon, np.float64))
        p["year"].append(np.asarray(year, np.int64))
        p["scale"].append(np.full(n, SCALES.index(scale)))
        p["group"].append(np.asarray(groups, np.int64))
        p["weight"].append(np.full(n, weight))
        p["origin"].append(np.full(n, len(self.origins)))
        p["cellable"].append(np.full(n, cellable))
        self.origins.append(name)
        logger.info(f"{name}: {n} samples at scale {scale}, weight {weight}")

    def finish(self) -> None:
        """Concatenate the parts and index texts."""
        for k, v in self.parts.items():
            setattr(self, k, np.concatenate(v))
        del self.parts
        self.texts = [""] * len(self.text_index)
        for t, i in self.text_index.items():
            self.texts[i] = t
        self.candidates = [
            {
                lv: next((g[f] for f in LEVEL_FALLBACK[lv] if g.get(f)), [])
                for lv in LEVELS
            }
            for g in self.group_levels
        ]


def add_pixels(samples: Samples, npz_path: str, texts_path: str, weight: float) -> None:
    """Pixel samples of one ``v5/FORMAT.md`` file."""
    pixels = load_pixel_samples(npz_path, texts_path)
    with np.load(npz_path) as stored:
        lat, lon, year = (stored[k] for k in ("lat", "lon", "year"))
    rows = pixels["id"]
    group_of = {tid: samples.group(lv) for tid, lv in pixels["texts"].items()}
    groups = np.array([group_of[int(t)] for t in pixels["text_id"]])
    samples.add(
        os.path.basename(npz_path).removesuffix(".npz"),
        pixels["embedding"],
        lat[rows],
        lon[rows],
        year[rows],
        "pixel",
        groups,
        weight,
        True,
    )


def add_captions(
    samples: Samples, chips_path: str, captions_path: str, weight: float, quads: bool
) -> None:
    """GPT-5.5 captions on their chip means (and optionally quadrant means)."""
    from olmoearth_pretrain.clipearth.captions import load_caption_samples

    with np.load(chips_path) as stored:
        year_of = dict(zip(stored["id"].astype(str), stored["year"]))
    kept = [
        s
        for s in load_caption_samples(chips_path, captions_path)
        if quads or s["source"] == "chip"
    ]
    samples.add(
        os.path.basename(captions_path).removesuffix(".jsonl"),
        np.stack([s["embedding"] for s in kept]),
        np.array([s["lat"] for s in kept]),
        np.array([s["lon"] for s in kept]),
        np.array([year_of[s["chip"]] for s in kept]),
        "area",
        np.array([samples.group(s["levels"]) for s in kept]),
        weight,
        False,
    )


def add_osm_regions(
    samples: Samples, regions_path: str, texts_path: str, weight: float
) -> None:
    """OSM region means (polygon means) at scale "area"."""
    with np.load(regions_path) as stored:
        regions = {k: stored[k] for k in stored.files}
    levels = {}
    with open(texts_path) as f:
        for line in f:
            record = json.loads(line)
            levels[int(record["id"])] = record["levels"]
    keep = np.array([int(i) in levels for i in regions["id"]])
    samples.add(
        "osm_regions",
        regions["embedding"][keep] * regions["embedding_norm"][keep, None],
        regions["lat"][keep],
        regions["lon"][keep],
        regions["year"][keep],
        "area",
        np.array([samples.group(levels[int(i)]) for i in regions["id"][keep]]),
        weight,
        False,
    )


class CellIndex:
    """Cellable samples grouped by spatial cell and, within a cell, by text group."""

    def __init__(self, samples: Samples, cell_deg: float) -> None:
        """Sort samples by (cell, group) and index the runs."""
        rows = np.flatnonzero(samples.cellable)
        cy = np.floor(samples.lat[rows] / cell_deg).astype(np.int64)
        cx = np.floor(samples.lon[rows] / cell_deg).astype(np.int64)
        cell = cy * 100_000 + cx
        order = np.lexsort((samples.group[rows], cell))
        self.rows = rows[order]
        cell, group = cell[order], samples.group[self.rows]
        new_run = np.r_[True, (cell[1:] != cell[:-1]) | (group[1:] != group[:-1])]
        self.run_start = np.r_[np.flatnonzero(new_run), len(self.rows)]
        run_cell = cell[new_run]
        new_cell = np.r_[True, run_cell[1:] != run_cell[:-1]]
        self.cell_run = np.r_[np.flatnonzero(new_cell), len(run_cell)]
        cell_of_run = np.cumsum(new_cell) - 1
        run_of_pos = np.cumsum(new_run) - 1
        self.sample_run = np.empty(len(samples.lat), np.int64)
        self.sample_run[self.rows] = run_of_pos
        self.run_cell = cell_of_run
        sizes = np.diff(self.cell_run)
        logger.info(
            f"cells: {len(sizes)} cells over {len(self.rows)} samples, "
            f"{(sizes >= 2).sum()} with 2+ labels, median labels per cell "
            f"{np.median(sizes):.0f}"
        )

    def mates(self, anchor: int, k: int, rng: np.random.Generator) -> list[int]:
        """The anchor and up to ``k - 1`` samples of other groups in its cell."""
        run = self.sample_run[anchor]
        c = self.run_cell[run]
        runs = np.arange(self.cell_run[c], self.cell_run[c + 1])
        runs = runs[runs != run]
        picked = rng.choice(runs, min(k - 1, len(runs)), replace=False)
        lo, hi = self.run_start[picked], self.run_start[picked + 1]
        return [anchor] + list(
            self.rows[lo + (rng.random(len(lo)) * (hi - lo)).astype(int)]
        )


def draw_weighted(cdf: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` indices drawn by weight (inverse CDF, with replacement)."""
    u = rng.random(n) * cdf[-1]
    return np.searchsorted(cdf, u, side="right").clip(max=len(cdf) - 1)


def build_batch(
    samples: Samples,
    cells: CellIndex | None,
    pools: dict[str, np.ndarray],
    cdfs: dict[str, np.ndarray],
    n_cells: int,
    cell_size: int,
    n_area: int,
    n_random: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """One batch: member matrix, the text per member and the candidate slots.

    Pixel cells hold up to ``cell_size`` members; each area sample is a cell of one.
    Each group in the batch gets one text, of a random level.
    """
    members = []
    if cells is not None and n_cells:
        for a in pools["cell"][draw_weighted(cdfs["cell"], n_cells, rng)]:
            m = cells.mates(int(a), cell_size, rng)
            members.append(m + [-1] * (cell_size - len(m)))
    if n_area and "area" in pools:
        for a in pools["area"][draw_weighted(cdfs["area"], n_area, rng)]:
            members.append([int(a)] + [-1] * (cell_size - 1))
    member = np.array(members, np.int64)  # [n, k]
    flat = member.reshape(-1)
    real = flat >= 0
    groups, inverse = np.unique(samples.group[flat[real]], return_inverse=True)
    level = rng.integers(len(LEVELS), size=len(groups))
    chosen = np.array(
        [
            c[rng.integers(len(c))] if (c := samples.candidates[g][LEVELS[lv]]) else -1
            for g, lv in zip(groups, level)
        ]
    )
    text = np.full(len(flat), -1)
    text[real] = chosen[inverse]
    unique = np.unique(text[text >= 0])
    k = cell_size
    cell_texts = np.repeat(text.reshape(-1, k), k, axis=0)  # [B, k]
    random = unique[rng.integers(len(unique), size=(len(flat), n_random))]
    slot_text = np.concatenate([cell_texts, random], 1)
    slot_text[~real] = -1
    # One usable slot per text: later repeats (and repeats of earlier slots) are masked.
    order = np.argsort(slot_text, axis=1, kind="stable")
    sorted_ = np.take_along_axis(slot_text, order, 1)
    repeat = np.zeros_like(slot_text, bool)
    np.put_along_axis(
        repeat,
        order,
        np.c_[np.zeros(len(flat), bool), sorted_[:, 1:] == sorted_[:, :-1]],
        1,
    )
    slot_text[repeat] = -1
    has_text = text >= 0
    target = np.where(has_text, (slot_text == text[:, None]).argmax(1), -1)
    # A member without any text sits out (as a padding row) but keeps its slot.
    return {
        "member": flat,
        "slot_text": slot_text,
        "target": target,
        "text_pos": np.searchsorted(unique, slot_text.clip(min=0)),
        "unique": unique,
        "n_cells": len(member),
    }


def batch_condition(
    samples: Samples,
    rows: np.ndarray,
    drop_loc: float,
    drop_year: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Condition vectors of ``rows`` (padding rows take row 0), with dropout."""
    r = rows.clip(min=0)
    lat, lon = samples.lat[r].copy(), samples.lon[r].copy()
    year = samples.year[r].copy()
    lost = rng.random(len(r)) < drop_loc
    lat[lost] = lon[lost] = np.nan
    year[rng.random(len(r)) < drop_year] = 0
    return condition(lat, lon, year, samples.scale[r], samples.frequencies)


def category_fit(
    model: GeoTextHead,
    image: Tensor,
    cond: Tensor,
    text_emb: Tensor,
    target: np.ndarray,
    chunk: int = 32,
) -> float:
    """Top-1 of each image's softmax over all category texts under its condition."""
    hidden = model.hidden(text_emb)
    hits = []
    cand = torch.arange(len(text_emb))[None]
    for s in range(0, len(image), chunk):
        n = len(image[s : s + chunk])
        logits = pair_logits(
            model,
            hidden,
            cand.expand(n, -1),
            cond[s : s + chunk],
            image[s : s + chunk],
            [None],
        )[0]
        hits.append(logits.argmax(1).numpy() == target[s : s + chunk])
    return float(np.concatenate(hits).mean())


def train(args: argparse.Namespace) -> None:
    """Train the geo head."""
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    samples = Samples()
    pixel_files = [p for p in args.pixels.split(",") if p]
    pixel_texts = [p for p in args.pixel_texts.split(",") if p]
    if len(pixel_files) != len(pixel_texts):
        raise ValueError("--pixels and --pixel_texts need one file each")
    weights = (
        [float(w) for w in args.pixel_weights.split(",")]
        if args.pixel_weights
        else [1.0] * len(pixel_files)
    )
    for path, texts_path, w in zip(pixel_files, pixel_texts, weights):
        add_pixels(samples, path, texts_path, w)
    if args.captions:
        add_captions(samples, args.caption_chips, args.captions, 1.0, args.quads)
    if args.osm_regions:
        add_osm_regions(samples, args.osm_regions, args.osm_texts, args.osm_weight)
    samples.finish()
    samples.frequencies = make_frequencies(
        tuple(float(w) for w in args.wavelengths_km.split(",")),
        args.per_scale,
        args.seed,
    )
    logger.info(
        f"{len(samples.lat)} samples, {len(samples.group_levels)} groups, "
        f"{len(samples.texts)} texts"
    )

    model = GeoTextHead(768, 1024, 128, samples.frequencies, args.cond_hidden)
    init_args = load_init(model, args.init)
    text_args = {
        k: init_args.get(k)
        for k in ("text_model", "text_prefix", "text_dims", "text_onnx_file")
    }
    view_emb = torch.from_numpy(
        encode_texts(
            samples.texts,
            text_args["text_model"],
            "cpu",
            text_args["text_prefix"] or "",
            text_args["text_dims"],
            text_args["text_onnx_file"] or "model_q4.onnx",
            args.text_cache or None,
        )
    )
    image = torch.from_numpy(samples.image)
    prefixes: list[int | None] = [None] + [
        int(d) for d in args.matryoshka.split(",") if d
    ]

    cellable = samples.cellable
    cells = CellIndex(samples, args.cell_deg) if cellable.any() else None
    pools = {"cell": np.flatnonzero(cellable), "area": np.flatnonzero(~cellable)}
    pools = {k: v for k, v in pools.items() if len(v)}
    cdfs = {k: np.cumsum(samples.weight[v]) for k, v in pools.items()}
    n_area = round(args.batch_images * args.area_frac) if "area" in pools else 0
    n_cells = (args.batch_images - n_area) // args.cell_size if cells else 0
    if not cells:
        n_area = args.batch_images
    logger.info(
        f"batch: {n_cells} cells of up to {args.cell_size} pixels, {n_area} area "
        f"samples, {args.random_negatives} random negatives each"
    )

    film_params = list(model.film.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": film_params, "lr": args.lr},
            {"params": list(model.text.parameters()), "lr": args.text_lr},
            {"params": [model.logit_scale], "lr": args.text_lr, "weight_decay": 0.0},
        ],
        weight_decay=args.weight_decay,
    )
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr, args.text_lr, args.text_lr],
        total_steps=args.steps,
        pct_start=0.05,
    )

    # Fit only (no holdout): a fixed subset of each pixel file against its own labels.
    fit_sets = []
    eval_rng = np.random.default_rng(args.seed + 1)
    for o, name in enumerate(samples.origins):
        rows = np.flatnonzero((samples.origin == o) & cellable)
        rows = rows[[bool(samples.candidates[g]["name"]) for g in samples.group[rows]]]
        if not len(rows):
            continue
        rows = np.sort(
            eval_rng.choice(rows, min(len(rows), args.fit_samples), replace=False)
        )
        first = [samples.candidates[g]["name"][0] for g in samples.group[rows]]
        ids, target = np.unique(first, return_inverse=True)
        cond = torch.from_numpy(
            condition(
                samples.lat[rows],
                samples.lon[rows],
                samples.year[rows],
                samples.scale[rows],
                samples.frequencies,
            )
        )
        blind = torch.from_numpy(
            condition(
                np.full(len(rows), np.nan),
                np.full(len(rows), np.nan),
                np.zeros(len(rows)),
                samples.scale[rows],
                samples.frequencies,
            )
        )
        fit_sets.append((name, rows, ids, target, cond, blind))

    def run_fit() -> dict:
        model.eval()
        out = {}
        with torch.no_grad():
            for name, rows, ids, target, cond, blind in fit_sets:
                img, emb = image[rows].float(), view_emb[ids]
                out[f"fit_{name}"] = category_fit(model, img, cond, emb, target)
                out[f"fit_{name}_noloc"] = category_fit(model, img, blind, emb, target)
                out[f"num_categories_{name}"] = len(ids)
        model.train()
        return out

    history = [{"step": 0, **run_fit()}]
    logger.info(json.dumps(history[-1]))
    start = time.time()
    for step in range(1, args.steps + 1):
        batch = build_batch(
            samples,
            cells,
            pools,
            cdfs,
            n_cells,
            args.cell_size,
            n_area,
            args.random_negatives,
            rng,
        )
        rows = batch["member"]
        cond = torch.from_numpy(
            batch_condition(samples, rows, args.drop_loc, args.drop_year, rng)
        )
        hidden = model.hidden(view_emb[batch["unique"]])
        scale = model.logit_scale.exp().clamp(max=100)
        slot_text = torch.from_numpy(batch["slot_text"])
        target = torch.from_numpy(batch["target"])
        losses = []
        for logits in pair_logits(
            model,
            hidden,
            torch.from_numpy(batch["text_pos"]),
            cond,
            image[rows.clip(min=0)],
            prefixes,
        ):
            i2t, t2i = batch_loss(
                scale * logits, slot_text, target, args.cell_size, batch["n_cells"]
            )
            losses.append(i2t if t2i is None else (i2t + t2i) / 2)
        loss = sum(losses) / len(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % args.log_every == 0:
            logger.info(
                f"step {step} loss {loss.item():.4f} scale {scale.item():.1f} "
                f"{(time.time() - start) / step:.3f}s/step"
            )
        if step % args.eval_every == 0 or step == args.steps:
            history.append({"step": step, "loss": loss.item(), **run_fit()})
            logger.info(json.dumps(history[-1]))
            save(model, args, text_args, os.path.join(args.out_dir, "geo_head.pt"))
    json.dump(history, open(os.path.join(args.out_dir, "metrics.json"), "w"), indent=1)


def save(
    model: GeoTextHead, args: argparse.Namespace, text_args: dict, path: str
) -> None:
    """Checkpoint with everything ``export`` and the sanity checks need."""
    torch.save(
        {
            "state_dict": model.state_dict(),
            "frequencies": model.frequencies.numpy(),
            "cond_hidden": args.cond_hidden,
            "text_args": text_args,
            "args": vars(args),
        },
        path + ".tmp",
    )
    os.replace(path + ".tmp", path)


def load_head(path: str) -> tuple[GeoTextHead, dict]:
    """A trained geo head and its checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    model = GeoTextHead(
        state["text.0.weight"].shape[1],
        state["text.0.weight"].shape[0],
        state["text.2.weight"].shape[0],
        ckpt["frequencies"],
        ckpt["cond_hidden"],
    )
    model.load_state_dict(state)
    return model.eval(), ckpt


def _f16(t: Tensor) -> str:
    data = t.detach().cpu().numpy().astype("<f2")
    return base64.b64encode(np.ascontiguousarray(data).tobytes()).decode()


def _f64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype="<f8").tobytes()).decode()


def _dense(layer: nn.Linear) -> dict:
    return {
        "inFeatures": layer.in_features,
        "outFeatures": layer.out_features,
        "weight": _f16(layer.weight),
        "bias": _f16(layer.bias),
    }


def as_browser(model: GeoTextHead) -> GeoTextHead:
    """A copy with float16-rounded weights, as the browser decodes them."""
    copy = copy_module.deepcopy(model)
    with torch.no_grad():
        for name, p in copy.named_parameters():
            if name != "logit_scale":
                p.copy_(p.half().float())
    return copy


def export(args: argparse.Namespace) -> None:
    """Write the browser JSON and the parity reference."""
    model, ckpt = load_head(args.head)
    text_args = ckpt["text_args"]
    onnx = text_args["text_model"].startswith("onnx:")
    head = {
        "textModel": args.browser_model,
        "dtype": args.dtype,
        "prefix": text_args.get("text_prefix") or "",
        "output": "sentence_embedding" if onnx else "cls",
        "modelFileName": (text_args.get("text_onnx_file") or "model.onnx")
        .removesuffix(".onnx")
        .removesuffix(f"_{args.dtype}"),
        "textDims": text_args.get("text_dims"),
        "activation": "gelu",
        "layers": [_dense(model.text[0]), _dense(model.text[2])],
        "matryoshkaDims": [128]
        + [int(d) for d in ckpt["args"].get("matryoshka", "").split(",") if d],
        "geo": {
            "frequencies": _f64(ckpt["frequencies"]),
            "numFrequencies": len(ckpt["frequencies"]),
            "years": list(YEARS),
            "scales": list(SCALES),
            "film": [_dense(model.film[0]), _dense(model.film[2])],
        },
    }
    with open(args.out, "w") as f:
        json.dump(head, f, separators=(",", ":"))
    if not args.parity:
        return
    cases = [json.loads(line) for line in open(args.parity_cases) if line.strip()]
    emb = encode_texts(
        [c["text"] for c in cases],
        text_args["text_model"],
        "cpu",
        text_args.get("text_prefix") or "",
        text_args.get("text_dims"),
        text_args.get("text_onnx_file") or "model_q4.onnx",
        args.text_cache or None,
    )
    browser = as_browser(model)
    lat = np.array([np.nan if c.get("lat") is None else c["lat"] for c in cases])
    lon = np.array([np.nan if c.get("lon") is None else c["lon"] for c in cases])
    cond = condition(
        lat,
        lon,
        np.array([c.get("year") or 0 for c in cases]),
        np.array([SCALES.index(c.get("scale", "pixel")) for c in cases]),
        ckpt["frequencies"],
    )
    with torch.no_grad():
        out = {
            dims: browser(torch.from_numpy(emb), torch.from_numpy(cond), dims).numpy()
            for dims in head["matryoshkaDims"]
        }
    reference = [
        {
            **c,
            "textEmbedding": emb[i].round(6).tolist(),
            "condition": cond[i].round(6).tolist(),
            "output": {str(d): out[d][i].round(6).tolist() for d in out},
        }
        for i, c in enumerate(cases)
    ]
    with open(args.parity, "w") as f:
        json.dump(reference, f, indent=0)


def main() -> None:
    """Train or export."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out_dir", required=True)
    t.add_argument("--init", required=True, help="native aligner.pt to start from")
    t.add_argument("--pixels", default="")
    t.add_argument("--pixel_texts", default="")
    t.add_argument(
        "--pixel_weights",
        default="",
        help="per --pixels file, the sampling weight of one sample (default 1 each)",
    )
    t.add_argument("--captions", default="", help="GPT-5.5 captions .jsonl")
    t.add_argument("--caption_chips", default="", help="chips .npz of --captions")
    t.add_argument("--quads", action="store_true", help="also caption quadrants")
    t.add_argument("--osm_regions", default="", help="optional OSM region means")
    t.add_argument("--osm_texts", default="")
    t.add_argument("--osm_weight", type=float, default=1.0)
    t.add_argument("--text_cache", default="")
    t.add_argument("--wavelengths_km", default=",".join(map(str, WAVELENGTHS_KM)))
    t.add_argument("--per_scale", type=int, default=16)
    t.add_argument("--cond_hidden", type=int, default=256)
    t.add_argument("--cell_deg", type=float, default=0.1)
    t.add_argument("--cell_size", type=int, default=6)
    t.add_argument("--batch_images", type=int, default=1024)
    t.add_argument(
        "--area_frac", type=float, default=0.25, help="share of area (caption) samples"
    )
    t.add_argument("--random_negatives", type=int, default=124)
    t.add_argument("--drop_loc", type=float, default=0.1)
    t.add_argument("--drop_year", type=float, default=0.2)
    t.add_argument("--matryoshka", default="64")
    t.add_argument("--steps", type=int, default=5000)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--text_lr", type=float, default=2e-4)
    t.add_argument("--weight_decay", type=float, default=0.01)
    t.add_argument("--fit_samples", type=int, default=2000)
    t.add_argument("--log_every", type=int, default=50)
    t.add_argument("--eval_every", type=int, default=1000)
    t.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("export")
    e.add_argument("--head", required=True)
    e.add_argument("--browser_model", required=True)
    e.add_argument("--dtype", default="q4")
    e.add_argument("--out", required=True)
    e.add_argument("--parity", default="", help="write the parity reference here")
    e.add_argument(
        "--parity_cases",
        default="",
        help='.jsonl of {"text", "lat", "lon", "year", "scale"}; null lat/lon or year '
        "means unknown",
    )
    e.add_argument("--text_cache", default="")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "train":
        train(args)
    else:
        export(args)


if __name__ == "__main__":
    main()
