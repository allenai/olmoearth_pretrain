"""Train a window retriever on cached frozen-v1.3 embeddings (the frozen check).

Reads the npz chunks written by ``window_retrieval_cache.py cache``. Each example
is one pixel of one sample and one window: input the embedding of the sample's
12 months ``z_T``, the input's start month, and the window ``(start, length)``
in 30-day steps relative to the input; target ``z_t``, the embedding fresh
inference on that window gives. Only the retriever trains.

Data:

- ``--cache_dir``: calendar-year caches (start month 0, the case the released
  store holds). Split by whole sample into train / val / test.
- ``--extra_train_cache_dir``: optional caches of samples starting in other
  months. Train only, so val and test stay calendar-year.

Retrievers (``--arch``), all of the form ``ẑ = z + Δ(z, window)`` and returning
``z`` unchanged for the full 12 months:

- ``floor``: Δ = 0, the input embedding used as-is.
- ``lowrank``: ``ẑ = (I + U·diag(f(c))·Vᵀ)·z + b(c)``, U, V shared 128×r.
- ``lowrank_full``: the same with a full r×r ``M(c)`` in place of ``diag(f(c))``.
- ``lowrank_lenonly``: ``lowrank`` whose code sees only the window length (control).
- ``film``: an MLP with FiLM (scale/shift from the window) in every layer and a
  zero-initialized output gate (AdaLN-Zero), so training starts at the floor.
- ``film_lenonly``: ``film`` whose code sees only the window length (control).

The window code ``c`` sums learned embeddings of the input's start month and the
window's start month-of-year, end month-of-year and length, then an MLP, so what
is learned about a season is shared across input start months.

``--max_train_samples`` keeps the first N calendar-year training samples of a
fixed shuffle, so smaller sets are nested in larger ones (a learning curve over
the same val / test). Training runs ``--epochs`` passes over the training pixels
(at least ``--min_steps``), evaluates on val
every ``--eval_every`` steps, keeps the best val checkpoint (overall cosine) and
stops after ``--patience`` evaluations without improvement. Metrics are reported
for val and test with that checkpoint. Per window length: cosine and MSE of the
retrieval and of the floor against the target, the fraction of the cosine gap
closed, and a wrong-window control (retrieval with windows shuffled).

Usage::

    python scripts/tools/window_retrieval_train.py --cache_dir $CAL --out_dir $OUT --arch lowrank
    python scripts/tools/window_retrieval_train.py --cache_dir $CAL --extra_train_cache_dir $ANY \
        --out_dir $OUT --arch lowrank
"""

import argparse
import copy
import glob
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from olmoearth_pretrain.evals.embedding_transforms import (
    dequantize_embeddings,
    quantize_embeddings,
)

logger = logging.getLogger(__name__)

MONTHS = 12


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_cache(
    cache_dirs: list[str], quantize: bool, n_pixels: int | None = None
) -> dict[str, Tensor]:
    """Concatenate all chunks under the given dirs; optionally int8 round-trip.

    Embeddings are kept float16; the round trip is applied per chunk in float32.
    With ``n_pixels``, chunks holding more pixels per sample (e.g. a ``--patch_px``
    cache) keep a fixed random subset of that many, so caches can be concatenated.
    """
    files = sorted(
        f for d in cache_dirs for f in glob.glob(os.path.join(d, "shard*.npz"))
    )
    if not files:
        raise FileNotFoundError(f"no shard*.npz under {cache_dirs}")
    parts: dict[str, list] = {}
    for f in files:
        with np.load(f) as d:
            n = d["z_T"].shape[0]
            z_T = torch.from_numpy(d["z_T"]).float()
            z_w = torch.from_numpy(d["z_w"]).float()
            if n_pixels is not None and z_T.shape[1] > n_pixels:
                g = torch.Generator().manual_seed(len(parts.get("z_T", [])))
                keep = torch.randperm(z_T.shape[1], generator=g)[:n_pixels]
                z_T, z_w = z_T[:, keep], z_w[:, :, keep]
            if quantize:
                z_T = dequantize_embeddings(quantize_embeddings(z_T))
                z_w = dequantize_embeddings(quantize_embeddings(z_w))
            parts.setdefault("z_T", []).append(z_T.half())
            parts.setdefault("z_w", []).append(z_w.half())
            win = torch.from_numpy(d["win"].astype(np.int64))
            parts.setdefault("start", []).append(win[..., 0])
            parts.setdefault("length", []).append(win[..., 1])
            parts.setdefault("valid", []).append(
                torch.from_numpy(d["present"].sum(-1) > 0)
            )
            parts.setdefault("index", []).append(torch.from_numpy(d["index"]).long())
            sm = (
                d["start_month"]
                if "start_month" in d.files
                else np.zeros(n, dtype=np.int64)
            )
            parts.setdefault("start_month", []).append(
                torch.from_numpy(sm.astype(np.int64))
            )
    data = {k: torch.cat(v) for k, v in parts.items()}
    logger.info(
        f"{len(files)} chunks from {cache_dirs}: {data['z_T'].shape[0]} samples, "
        f"{data['z_T'].shape[1]} pixels, {data['z_w'].shape[1]} windows, "
        f"start months {torch.bincount(data['start_month'], minlength=MONTHS).tolist()}"
    )
    return data


def subset(d: dict[str, Tensor], idx: Tensor) -> dict[str, Tensor]:
    """Rows idx of every field."""
    return {k: v[idx] for k, v in d.items()}


def concat(*ds: dict[str, Tensor]) -> dict[str, Tensor]:
    """Concatenate datasets field by field."""
    return {k: torch.cat([d[k] for d in ds]) for k in ds[0]}


def split(
    cal: dict[str, Tensor], val_frac: float, test_frac: float, seed: int
) -> tuple[dict, dict, dict]:
    """Train / val / test by whole calendar-year sample."""
    if (cal["start_month"] != 0).any():
        raise ValueError(
            "--cache_dir must hold calendar-year samples only (start month 0)"
        )
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(cal["z_T"].shape[0], generator=g)
    n_val, n_test = int(val_frac * len(perm)), int(test_frac * len(perm))
    return (
        subset(cal, perm[n_val + n_test :]),
        subset(cal, perm[:n_val]),
        subset(cal, perm[n_val : n_val + n_test]),
    )


def examples(d: dict[str, Tensor]) -> Tensor:
    """(sample, window) pairs with a valid target, as a [M, 2] index tensor."""
    s, k = torch.nonzero(d["valid"], as_tuple=True)
    return torch.stack([s, k], dim=1)


def gather(d: dict[str, Tensor], pairs: Tensor, pix: Tensor) -> tuple[Tensor, ...]:
    """Inputs, targets and window fields for (sample, window, pixel) triples."""
    s, k = pairs[:, 0], pairs[:, 1]
    return (
        d["z_T"][s, pix].float(),
        d["z_w"][s, k, pix].float(),
        d["start_month"][s],
        d["start"][s, k],
        d["length"][s, k],
    )


# ---------------------------------------------------------------------------
# retrievers
# ---------------------------------------------------------------------------


def location_features(latlon: Tensor, n_freqs: int = 4) -> Tensor:
    """Lat/lon (degrees) -> unit vector on the sphere plus its sin/cos Fourier features."""
    lat, lon = (
        torch.deg2rad(latlon[..., 0].float()),
        torch.deg2rad(latlon[..., 1].float()),
    )
    xyz = torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)
    freqs = 2.0 ** torch.arange(n_freqs, device=xyz.device, dtype=xyz.dtype)
    ang = xyz[..., None] * freqs * torch.pi  # [..., 3, F]
    return torch.cat([xyz, ang.sin().flatten(-2), ang.cos().flatten(-2)], dim=-1)


class WindowCode(nn.Module):
    """Embedding of (input start month, window start/end month-of-year, length).

    With ``length_only`` it sees only the window length.
    """

    def __init__(
        self, code_dim: int, length_only: bool = False, use_location: bool = False
    ) -> None:
        """Summed per-field embeddings, then an MLP.

        With ``use_location`` a projection of the pixel's lat/lon (unit vector plus
        Fourier features) is added too, so the same calendar months can mean
        different seasons in different places.
        """
        super().__init__()
        self.length_only = length_only
        self.location = nn.Linear(3 + 2 * 3 * 4, code_dim) if use_location else None
        self.length = nn.Embedding(MONTHS + 1, code_dim)
        self.input_start = nn.Embedding(MONTHS, code_dim)
        self.window_start = nn.Embedding(MONTHS, code_dim)
        self.window_end = nn.Embedding(MONTHS, code_dim)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
        )

    def forward(
        self,
        start_month: Tensor,
        start: Tensor,
        length: Tensor,
        latlon: Tensor | None = None,
    ) -> Tensor:
        """Code for each row's window (and location, when the code uses it)."""
        c = self.length(length)
        if self.location is not None:
            if latlon is None:
                raise ValueError("this retriever needs lat/lon")
            c = c + self.location(location_features(latlon))
        if not self.length_only:
            first = (start_month + start) % MONTHS
            last = (start_month + start + length - 1) % MONTHS
            c = (
                c
                + self.input_start(start_month)
                + self.window_start(first)
                + self.window_end(last)
            )
        return self.mlp(c)


class Floor(nn.Module):
    """The input embedding used as-is."""

    def forward(
        self, z: Tensor, start_month: Tensor, start: Tensor, length: Tensor
    ) -> Tensor:
        """Return z."""
        return z


class LowRank(nn.Module):
    """``ẑ = z + U·(M(c)·(Vᵀz)) + b(c)``, M diagonal or full r×r."""

    def __init__(
        self, dim: int, rank: int, code_dim: int, full: bool, length_only: bool = False
    ) -> None:
        """Initialize at the floor: the window terms start at zero."""
        super().__init__()
        self.rank, self.full = rank, full
        self.code = WindowCode(code_dim, length_only)
        self.U = nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
        self.V = nn.Parameter(torch.randn(dim, rank) / math.sqrt(dim))
        self.m = nn.Linear(code_dim, rank * rank if full else rank)
        self.b = nn.Linear(code_dim, dim)
        for layer in (self.m, self.b):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self, z: Tensor, start_month: Tensor, start: Tensor, length: Tensor
    ) -> Tensor:
        """Apply the window's low-rank update."""
        c = self.code(start_month, start, length)
        proj = z @ self.V  # [B, r]
        if self.full:
            proj = torch.einsum(
                "brs,bs->br", self.m(c).view(-1, self.rank, self.rank), proj
            )
        else:
            proj = proj * self.m(c)
        return z + proj @ self.U.T + self.b(c)


class FiLMBlock(nn.Module):
    """Pre-LN residual MLP block, modulated by the window (AdaLN-Zero)."""

    def __init__(self, width: int, code_dim: int, dropout: float = 0.0) -> None:
        """Modulation starts at zero, so the block starts as identity."""
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * width, width),
        )
        self.mod = nn.Linear(code_dim, 3 * width)  # shift, scale, gate
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h: Tensor, c: Tensor) -> Tensor:
        """H + gate · MLP((1 + scale) · LN(h) + shift)."""
        shift, scale, gate = self.mod(c).chunk(3, dim=-1)
        return h + gate * self.mlp(self.norm(h) * (1 + scale) + shift)


class FiLM(nn.Module):
    """FiLM-modulated MLP with a zero-initialized output gate."""

    def __init__(
        self,
        dim: int,
        width: int,
        depth: int,
        code_dim: int,
        length_only: bool = False,
        dropout: float = 0.0,
        use_location: bool = False,
    ) -> None:
        """Initialize at the floor: the output gate starts at zero."""
        super().__init__()
        self.code = WindowCode(code_dim, length_only, use_location)
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.blocks = nn.ModuleList(
            FiLMBlock(width, code_dim, dropout) for _ in range(depth)
        )
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
        self.alpha = nn.Linear(code_dim, dim)
        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.alpha.bias)

    def forward(
        self,
        z: Tensor,
        start_month: Tensor,
        start: Tensor,
        length: Tensor,
        latlon: Tensor | None = None,
    ) -> Tensor:
        """Z + α(c) · Δ(z, c)."""
        c = self.code(start_month, start, length, latlon)
        h = self.inp(z)
        for block in self.blocks:
            h = block(h, c)
        return z + self.alpha(c) * self.out(h)


def build(args: argparse.Namespace, dim: int) -> nn.Module:
    """The retriever named by --arch."""
    if args.arch == "floor":
        return Floor()
    if args.arch in ("lowrank", "lowrank_full", "lowrank_lenonly"):
        return LowRank(
            dim,
            args.rank,
            args.code_dim,
            full=args.arch == "lowrank_full",
            length_only=args.arch == "lowrank_lenonly",
        )
    if args.arch in ("film", "film_lenonly"):
        return FiLM(
            dim,
            args.width,
            args.depth,
            args.code_dim,
            length_only=args.arch == "film_lenonly",
            dropout=getattr(args, "dropout", 0.0),
            use_location=getattr(args, "latlon", False),
        )
    raise ValueError(args.arch)


def retrieve(
    model: nn.Module,
    z: Tensor,
    start_month: Tensor,
    start: Tensor,
    length: Tensor,
    latlon: Tensor | None = None,
) -> Tensor:
    """Retrieval, with the full 12 months returning z exactly."""
    if latlon is None:
        out = model(z, start_month, start, length)
    else:
        out = model(z, start_month, start, length, latlon)
    return torch.where((length == MONTHS)[:, None], z, out)


# ---------------------------------------------------------------------------
# train / eval
# ---------------------------------------------------------------------------


def loss_fn(pred: Tensor, target: Tensor, kind: str) -> Tensor:
    """MSE or cosine distance."""
    if kind == "mse":
        return F.mse_loss(pred, target)
    return (1 - F.cosine_similarity(pred, target, dim=-1)).mean()


def pixel_batch(d: dict[str, Tensor], n_pixels: int, device: str) -> tuple[Tensor, ...]:
    """All cached windows of ``n_pixels`` random (sample, pixel) pairs.

    Returns ``z_T [P, D]``, ``z_t [P, K, D]``, start month ``[P]``, window start and
    length ``[P, K]`` and the valid mask ``[P, K]``: a pixel's windows stay together,
    so a loss can compare a window's retrieval with the same pixel's other windows.
    """
    n, n_pix = d["z_T"].shape[0], d["z_T"].shape[1]
    s = torch.randint(n, (n_pixels,), device=device)
    pix = torch.randint(n_pix, (n_pixels,), device=device)
    return (
        d["z_T"][s, pix].float(),
        d["z_w"][s, :, pix].float(),
        d["start_month"][s],
        d["start"][s],
        d["length"][s],
        d["valid"][s],
    )


def tile_batch(d: dict[str, Tensor], n_tiles: int, device: str) -> tuple[Tensor, ...]:
    """Every cached pixel and window of ``n_tiles`` random samples.

    Returns whether each tile may take within-tile terms ``[T]``, ``z_T [T, P, D]``, ``z_t [T, P, K, D]``, start month ``[T]``, window start
    and length ``[T, K]`` and the valid mask ``[T, K]``: a tile's pixels stay together,
    so a loss can compare each pixel with its neighbours in the same tile.
    """
    s = torch.randint(d["z_T"].shape[0], (n_tiles,), device=device)
    return (
        d["tile_ok"][s],
        d["z_T"][s].float(),
        d["z_w"][s].float().permute(0, 2, 1, 3),
        d["start_month"][s],
        d["start"][s],
        d["length"][s],
        d["valid"][s],
    )


def tile_rows(x: Tensor, valid: Tensor) -> Tensor:
    """``[T, P, K, D]`` -> ``[V, P, D]``: the pixels of every valid (tile, window)."""
    return x.permute(0, 2, 1, 3)[valid]


def tile_centered_mse(pred: Tensor, tgt: Tensor, valid: Tensor) -> Tensor:
    """MSE on each pixel's deviation from its tile's mean embedding (per window)."""
    p, t = tile_rows(pred, valid), tile_rows(tgt, valid)
    return F.mse_loss(p - p.mean(1, keepdim=True), t - t.mean(1, keepdim=True))


def tile_infonce(pred: Tensor, tgt: Tensor, valid: Tensor, tau: float) -> Tensor:
    """Each retrieved pixel must pick out its own fresh embedding among its tile's pixels."""
    p = F.normalize(tile_rows(pred, valid), dim=-1)
    t = F.normalize(tile_rows(tgt, valid), dim=-1)
    logits = p @ t.transpose(1, 2) / tau  # [V, P, P]
    P = logits.shape[-1]
    eye = torch.eye(P, dtype=torch.bool, device=logits.device)
    # Pixels were drawn with replacement: a repeated pixel is not a negative.
    dup = (t @ t.transpose(1, 2) > 0.9999) & ~eye
    logits = logits.masked_fill(dup, float("-inf"))
    labels = torch.arange(P, device=logits.device).expand(len(logits), P)
    return F.cross_entropy(logits.reshape(-1, P), labels.reshape(-1))


def tile_relational_kl(pred: Tensor, tgt: Tensor, valid: Tensor, tau: float) -> Tensor:
    """KL between each pixel's neighbour distributions within its tile (fresh || retrieved)."""
    p = F.normalize(tile_rows(pred, valid), dim=-1)
    t = F.normalize(tile_rows(tgt, valid), dim=-1)
    sp, st = p @ p.transpose(1, 2) / tau, t @ t.transpose(1, 2) / tau
    eye = torch.eye(sp.shape[-1], dtype=torch.bool, device=sp.device)
    sp, st = sp.masked_fill(eye, float("-inf")), st.masked_fill(eye, float("-inf"))
    log_p, log_t = F.log_softmax(sp, dim=-1), F.log_softmax(st, dim=-1)
    kl = (log_t.exp() * (log_t - log_p)).masked_fill(eye, 0.0)
    return kl.sum(-1).mean()


def tile_window_rows(x: Tensor, valid: Tensor) -> list[Tensor]:
    """Per tile: its (pixel, window) rows over valid windows, centred per window. ``[T, P, K, D]`` -> T x ``[P*Kv, D]``."""
    xc = x - x.mean(1, keepdim=True)
    return [
        xc[t][:, valid[t]].reshape(-1, x.shape[-1])
        for t in range(x.shape[0])
        if valid[t].any()
    ]


def tile_window_cka(pred: Tensor, tgt: Tensor, valid: Tensor) -> Tensor:
    """1 - linear CKA between retrieved and fresh (pixel, window) rows of each tile, centred per window."""
    losses = []
    for p, t in zip(tile_window_rows(pred, valid), tile_window_rows(tgt, valid)):
        gp, gt = p @ p.T, t @ t.T
        cka = (gp * gt).sum() / (gp.norm() * gt.norm() + 1e-8)
        losses.append(1 - cka)
    return torch.stack(losses).mean()


def tile_window_kl(pred: Tensor, tgt: Tensor, valid: Tensor, tau: float) -> Tensor:
    """KL between each (pixel, window) row's neighbour distribution over its tile's rows (fresh || retrieved)."""
    losses = []
    for p, t in zip(tile_window_rows(pred, valid), tile_window_rows(tgt, valid)):
        p, t = F.normalize(p, dim=-1), F.normalize(t, dim=-1)
        eye = torch.eye(len(p), dtype=torch.bool, device=p.device)
        sp = (p @ p.T / tau).masked_fill(eye, float("-inf"))
        st = (t @ t.T / tau).masked_fill(eye, float("-inf"))
        log_p, log_t = F.log_softmax(sp, dim=-1), F.log_softmax(st, dim=-1)
        losses.append(
            (log_t.exp() * (log_t - log_p)).masked_fill(eye, 0.0).sum(-1).mean()
        )
    return torch.stack(losses).mean()


def tile_terms(
    pred: Tensor, tgt: Tensor, valid: Tensor, args: argparse.Namespace
) -> Tensor:
    """The enabled within-tile terms on ``[T, P, K, D]`` tile blocks (0 if none apply)."""
    loss = pred.new_zeros(())
    if not valid.any():
        return loss
    if args.tile_centered_weight:
        loss = loss + args.tile_centered_weight * tile_centered_mse(pred, tgt, valid)
    if args.tile_infonce_weight:
        loss = loss + args.tile_infonce_weight * tile_infonce(
            pred, tgt, valid, args.tau
        )
    if args.tile_kl_weight:
        loss = loss + args.tile_kl_weight * tile_relational_kl(
            pred, tgt, valid, args.tau
        )
    if args.tile_window_cka_weight:
        loss = loss + args.tile_window_cka_weight * tile_window_cka(pred, tgt, valid)
    if args.tile_window_kl_weight:
        loss = loss + args.tile_window_kl_weight * tile_window_kl(
            pred, tgt, valid, args.tau
        )
    return loss


class DenseStream:
    """Batches from a dense cache (``window_retrieval_cache.py --dense_px``) through a shuffle buffer.

    A background thread keeps ``n_chunks`` random chunk files in memory (int8 store
    codes) and swaps one for a fresh random chunk every ``refresh_every`` batches.
    Each batch draws pixels and windows anew, so no two steps repeat rows:

    - ``random_pixels``: pixels from random buffered tiles, all positions;
    - ``tiles`` x ``tile_pixels``: whole-tile blocks for the within-tile terms.

    Every pixel gets ``windows`` of its tile's stored windows (valid ones first).
    Samples whose index is in ``exclude`` (the val / test samples) are dropped.
    """

    def __init__(
        self,
        dirs: list[str],
        exclude: set[int],
        n_chunks: int,
        refresh_every: int,
        seed: int,
    ) -> None:
        """Index the chunk files and fill the buffer."""
        import random
        import threading

        self.files = sorted(
            f for d in dirs for f in glob.glob(os.path.join(d, "shard*.npz"))
        )
        if not self.files:
            raise FileNotFoundError(f"no shard*.npz under {dirs}")
        self.exclude = exclude
        self.refresh_every = refresh_every
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.buffer = [self._load(self.rng.choice(self.files)) for _ in range(n_chunks)]
        self.batches = 0
        self.next_chunk = None
        self.thread = None
        logger.info(
            f"dense stream: {len(self.files)} chunks under {dirs}; buffer {n_chunks} chunks "
            f"({sum(c['z_T'].shape[0] for c in self.buffer)} tiles), refresh every {refresh_every} batches"
        )

    def _load(self, path: str) -> dict[str, np.ndarray]:
        """One chunk, minus excluded samples."""
        with np.load(path) as d:
            keep = (
                ~np.isin(d["index"], list(self.exclude))
                if self.exclude
                else np.ones(len(d["index"]), bool)
            )
            return {
                "z_T": d["z_T"][keep],
                "z_w": d["z_w"][keep],
                "win": d["win"][keep].astype(np.int64),
                "valid": d["present"][keep].sum(-1) > 0,
                "start_month": d["start_month"][keep].astype(np.int64),
                "index": d["index"][keep],
            }

    def _refresh(self) -> None:
        """Swap in the prefetched chunk and start prefetching the next one."""
        import threading

        if self.thread is not None:
            self.thread.join()
            with self.lock:
                self.buffer[self.rng.randrange(len(self.buffer))] = self.next_chunk
        path = self.rng.choice(self.files)

        def fetch() -> None:
            self.next_chunk = self._load(path)

        self.thread = threading.Thread(target=fetch, daemon=True)
        self.thread.start()

    def _tiles(self, n: int) -> list[tuple[dict, int]]:
        """``n`` random (chunk, sample) pairs from the buffer."""
        out = []
        while len(out) < n:
            c = self.buffer[self.rng.randrange(len(self.buffer))]
            if c["z_T"].shape[0]:
                out.append((c, self.rng.randrange(c["z_T"].shape[0])))
        return out

    def _windows(self, c: dict, i: int, k: int, gen: np.random.Generator) -> np.ndarray:
        """``k`` window slots of sample ``i``, valid ones first, random order."""
        order = gen.permutation(c["win"].shape[1])
        order = order[np.argsort(~c["valid"][i][order], kind="stable")]
        return order[:k]

    def batch(
        self,
        random_pixels: int,
        tiles: int,
        tile_pixels: int,
        windows: int,
        device: str,
    ) -> dict:
        """One mixed batch as dequantized tensors on ``device``."""
        if self.batches % self.refresh_every == 0:
            self._refresh()
        self.batches += 1
        gen = np.random.default_rng(self.rng.randrange(2**31))
        out = {}
        with self.lock:
            # Random pixels: one tile and position each.
            if random_pixels:
                zs, ts, sms, sts, lns, vas, ids = [], [], [], [], [], [], []
                for c, i in self._tiles(random_pixels):
                    S = c["z_T"].shape[1]
                    r, q = gen.integers(S), gen.integers(S)
                    w = self._windows(c, i, windows, gen)
                    zs.append(c["z_T"][i, r, q])
                    ts.append(c["z_w"][i, w, r, q])
                    sms.append(c["start_month"][i])
                    sts.append(c["win"][i, w, 0])
                    lns.append(c["win"][i, w, 1])
                    vas.append(c["valid"][i, w])
                    ids.append(c["index"][i])
                out["random"] = (
                    np.stack(zs),
                    np.stack(ts),
                    np.array(sms),
                    np.stack(sts),
                    np.stack(lns),
                    np.stack(vas),
                    np.array(ids),
                )
            # Whole-tile blocks: ``tile_pixels`` distinct positions per tile.
            if tiles:
                zs, ts, sms, sts, lns, vas, ids = [], [], [], [], [], [], []
                for c, i in self._tiles(tiles):
                    S = c["z_T"].shape[1]
                    pos = gen.choice(S * S, size=tile_pixels, replace=False)
                    r, q = pos // S, pos % S
                    w = self._windows(c, i, windows, gen)
                    zs.append(c["z_T"][i, r, q])
                    ts.append(c["z_w"][i][w][:, r, q].transpose(1, 0, 2))
                    sms.append(c["start_month"][i])
                    sts.append(c["win"][i, w, 0])
                    lns.append(c["win"][i, w, 1])
                    vas.append(c["valid"][i, w])
                    ids.append(c["index"][i])
                out["tiles"] = (
                    np.stack(zs),
                    np.stack(ts),
                    np.array(sms),
                    np.stack(sts),
                    np.stack(lns),
                    np.stack(vas),
                    np.array(ids),
                )

        def deq(a: np.ndarray) -> Tensor:
            return dequantize_embeddings(torch.from_numpy(a).to(device)).float()

        def to(arrs: tuple) -> tuple:
            z, t, sm, st, ln, va, ids = arrs
            return (
                deq(z),
                deq(t),
                *(torch.from_numpy(a).to(device) for a in (sm, st, ln, va, ids)),
            )

        return {k: to(v) for k, v in out.items()}


def npz_array_location(path: str, key: str) -> tuple[int, tuple, np.dtype, bool]:
    """Byte offset, shape, dtype and Fortran order of one array in an uncompressed ``.npz``."""
    import zipfile

    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{key}.npy")
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{key} is compressed; cannot memory-map")
    with open(path, "rb") as f:
        # Local file header: 30 bytes + name + extra field, then the .npy bytes.
        f.seek(info.header_offset)
        header = f.read(30)
        name_len = int.from_bytes(header[26:28], "little")
        extra_len = int.from_bytes(header[28:30], "little")
        f.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(f)
        read_header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = read_header(f)
        return f.tell(), shape, dtype, fortran


def npz_memmap(path: str, key: str, location: tuple | None = None) -> np.ndarray:
    """Memory-map one array of an uncompressed ``.npz`` without loading it."""
    offset, shape, dtype, fortran = location or npz_array_location(path, key)
    return np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=offset,
        shape=shape,
        order="F" if fortran else "C",
    )


class _DenseRandomBatches(torch.utils.data.IterableDataset):
    """Endless random batches over every chunk, as numpy arrays.

    Runs inside DataLoader worker processes: each worker memory-maps the chunks
    itself and draws with its own seed, so page faults on scattered reads proceed
    in parallel across processes.
    """

    def __init__(
        self,
        files: list[str],
        small: list[dict],
        samples: np.ndarray,
        spec: tuple,
        seed: int,
        segment: int = 1,
    ) -> None:
        """Keep what the workers need; memory maps are opened per worker."""
        super().__init__()
        self.segment = segment
        self.files, self.small, self.samples, self.spec, self.seed = (
            files,
            small,
            samples,
            spec,
            seed,
        )

    def _windows(
        self, ch: dict, i: int, k: int, gen: np.random.Generator
    ) -> np.ndarray:
        """``k`` window slots of sample ``i``, valid ones first, random order."""
        order = gen.permutation(ch["win"].shape[1])
        order = order[np.argsort(~ch["valid"][i][order], kind="stable")]
        return order[:k]

    def __iter__(self):
        """Yield batches forever."""
        info = torch.utils.data.get_worker_info()
        gen = np.random.default_rng([self.seed, info.id if info is not None else 0])
        # Map each chunk on first use, from locations read once by the parent.
        maps: dict = {}

        def chunk(c: int) -> tuple:
            if c not in maps:
                ch = self.small[c]
                maps[c] = (
                    npz_memmap(self.files[c], "z_T", ch["loc_z_T"]),
                    npz_memmap(self.files[c], "z_w", ch["loc_z_w"]),
                )
            return maps[c]

        random_pixels, tiles, tile_pixels, windows = self.spec
        seg = self.segment
        while True:
            out = {}
            # Random pixels come in runs of ``seg`` along a row of a random tile
            # (one page per array per run); tile blocks are ``tile_pixels / seg``
            # such runs on distinct rows of one tile.
            for kind, count in (("random", random_pixels // seg), ("tiles", tiles)):
                if not count:
                    continue
                zs, ts, sms, sts, lns, vas, ids = [], [], [], [], [], [], []
                for c, i in self.samples[gen.integers(len(self.samples), size=count)]:
                    ch = self.small[c]
                    zT_c, zw_c = chunk(c)
                    S = zT_c.shape[1]
                    w = self._windows(ch, i, windows, gen)
                    rows = (
                        [gen.integers(S)]
                        if kind == "random"
                        else gen.choice(S, size=tile_pixels // seg, replace=False)
                    )
                    starts = gen.integers(S - seg + 1, size=len(rows))
                    z = np.concatenate(
                        [
                            np.asarray(zT_c[i, r, q : q + seg])
                            for r, q in zip(rows, starts)
                        ]
                    )
                    t = np.stack(
                        [
                            np.concatenate(
                                [
                                    np.asarray(zw_c[i, k, r, q : q + seg])
                                    for r, q in zip(rows, starts)
                                ]
                            )
                            for k in w
                        ],
                        axis=1,
                    )
                    n = len(z)
                    if kind == "random":
                        # One entry per pixel.
                        zs.append(z)
                        ts.append(t)
                        sms.append(np.full(n, ch["start_month"][i]))
                        sts.append(np.tile(ch["win"][i, w, 0], (n, 1)))
                        lns.append(np.tile(ch["win"][i, w, 1], (n, 1)))
                        vas.append(np.tile(ch["valid"][i, w], (n, 1)))
                        ids.append(np.full(n, ch["index"][i]))
                    else:
                        # One entry per tile.
                        zs.append(z[None])
                        ts.append(t[None])
                        sms.append(np.array([ch["start_month"][i]]))
                        sts.append(ch["win"][i, w, 0][None])
                        lns.append(ch["win"][i, w, 1][None])
                        vas.append(ch["valid"][i, w][None])
                        ids.append(np.array([ch["index"][i]]))
                out[kind] = tuple(
                    np.concatenate(x) for x in (zs, ts, sms, sts, lns, vas, ids)
                )
            yield out


class DenseRandom:
    """Truly random batches from a whole dense cache, through memory-mapped chunks.

    Same ``batch()`` as :class:`DenseStream`, but every pixel (and every tile block)
    is drawn uniformly from all non-excluded samples of all chunks. The small
    per-sample arrays are indexed once here; DataLoader worker processes read
    the embeddings in place and prefetch batches.
    """

    def __init__(
        self,
        dirs: list[str],
        exclude: set[int],
        seed: int,
        spec: tuple,
        workers: int = 32,
        segment: int = 1,
    ) -> None:
        """Index every chunk and start the workers."""
        from concurrent.futures import ThreadPoolExecutor

        files = sorted(
            f for d in dirs for f in glob.glob(os.path.join(d, "shard*.npz"))
        )
        if not files:
            raise FileNotFoundError(f"no shard*.npz under {dirs}")

        def index(path: str) -> dict:
            with np.load(path) as d:
                out = {
                    "index": d["index"],
                    "win": d["win"].astype(np.int64),
                    "valid": d["present"].sum(-1) > 0,
                    "start_month": d["start_month"].astype(np.int64),
                }
            out["loc_z_T"] = npz_array_location(path, "z_T")
            out["loc_z_w"] = npz_array_location(path, "z_w")
            return out

        with ThreadPoolExecutor(32) as pool:
            small = list(pool.map(index, files))
        samples = np.array(
            [
                (c, i)
                for c, ch in enumerate(small)
                for i, idx in enumerate(ch["index"])
                if int(idx) not in exclude
            ],
            dtype=np.int64,
        )
        self.spec = spec
        loader = torch.utils.data.DataLoader(
            _DenseRandomBatches(files, small, samples, spec, seed, segment),
            batch_size=None,
            num_workers=workers,
            prefetch_factor=4 if workers else None,
            persistent_workers=bool(workers),
        )
        self.it = iter(loader)
        logger.info(
            f"dense random: {len(files)} chunks, {len(samples)} samples (val/test excluded), {workers} workers"
        )

    def batch(
        self,
        random_pixels: int,
        tiles: int,
        tile_pixels: int,
        windows: int,
        device: str,
    ) -> dict:
        """The next prefetched batch as dequantized tensors on ``device``."""
        assert (random_pixels, tiles, tile_pixels, windows) == self.spec
        out = next(self.it)

        def deq(a) -> Tensor:
            return dequantize_embeddings(torch.as_tensor(a).to(device)).float()

        return {
            k: (
                deq(z),
                deq(t),
                *(torch.as_tensor(a).to(device) for a in (sm, st, ln, va, ids)),
            )
            for k, (z, t, sm, st, ln, va, ids) in out.items()
        }


def retrieve_windows(
    model: nn.Module,
    z: Tensor,
    sm: Tensor,
    st: Tensor,
    ln: Tensor,
    latlon: Tensor | None = None,
) -> Tensor:
    """Retrieve every window of every pixel: ``[P, D]`` + ``[P, K]`` fields -> ``[P, K, D]``."""
    P, K = st.shape
    flat = retrieve(
        model,
        z.repeat_interleave(K, dim=0),
        sm.repeat_interleave(K),
        st.reshape(-1),
        ln.reshape(-1),
        None if latlon is None else latlon.repeat_interleave(K, dim=0),
    )
    return flat.reshape(P, K, -1)


def gram_across_pixels(
    pred: Tensor, tgt: Tensor, valid: Tensor, max_rows: int = 2048
) -> Tensor:
    """Gram term across (pixel, window) rows, as in the distillation head.

    MSE between the cosine-similarity matrices of retrieved and fresh embeddings.
    """
    p, t = pred[valid], tgt[valid]
    if len(p) > max_rows:
        idx = torch.randperm(len(p), device=p.device)[:max_rows]
        p, t = p[idx], t[idx]
    p, t = F.normalize(p, dim=-1), F.normalize(t, dim=-1)
    return F.mse_loss(p @ p.T, t @ t.T)


def gram_across_windows(pred: Tensor, tgt: Tensor, valid: Tensor) -> Tensor:
    """Gram term across each pixel's own windows.

    MSE between the window-by-window cosine-similarity matrices of retrieved and
    fresh embeddings, over pairs of distinct valid windows.
    """
    p, t = F.normalize(pred, dim=-1), F.normalize(tgt, dim=-1)
    gp, gt = p @ p.transpose(1, 2), t @ t.transpose(1, 2)  # [P, K, K]
    K = valid.shape[1]
    pair = (
        valid[:, :, None]
        & valid[:, None, :]
        & ~torch.eye(K, dtype=torch.bool, device=valid.device)
    )
    return ((gp - gt) ** 2)[pair].mean()


@torch.no_grad()
def geometry(
    model: nn.Module,
    d: dict[str, Tensor],
    device: str,
    seed: int,
    knn_rows: int = 4096,
    k: int = 10,
) -> dict:
    """Downstream-relevant geometry of retrieved vs fresh embeddings, model and floor.

    - ``var_ratio``: total variance of the embeddings over all (sample, window, pixel)
      rows, relative to the fresh ones'. Below 1 means retrieval blurs toward the mean.
    - ``knn_agreement``: on a random subset of rows, the overlap between each row's
      ``k`` nearest neighbours (cosine) among retrieved and among fresh embeddings.
    - ``window_id``: per pixel, the fraction of windows whose retrieval is closer
      (cosine) to that window's fresh embedding than to the pixel's other windows'.
    """
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, K, n_pix = d["z_w"].shape[0], d["z_w"].shape[1], d["z_T"].shape[1]
    acc = {
        name: {"sum": 0.0, "sq": 0.0, "n": 0, "id_hit": 0, "id_n": 0}
        for name in ("ret", "floor", "fresh")
    }
    keep = max(1, knn_rows // max(1, n))
    knn = {"ret": [], "floor": [], "fresh": []}
    for chunk in torch.arange(n).split(16):
        z = d["z_T"][chunk].float().to(device)  # [S, P, D]
        tgt = d["z_w"][chunk].float().to(device)  # [S, K, P, D]
        valid = d["valid"][chunk].to(device)  # [S, K]
        S = len(chunk)
        sm = d["start_month"][chunk].to(device)
        st, ln = d["start"][chunk].to(device), d["length"][chunk].to(device)
        rows_z = z[:, None].expand(S, K, n_pix, -1).reshape(-1, z.shape[-1])
        ll = None
        if "latlon" in d:
            ll = (
                d["latlon"][chunk]
                .to(device)[:, None, None]
                .expand(S, K, n_pix, 2)
                .reshape(-1, 2)
            )
        pred = retrieve(
            model,
            rows_z,
            sm[:, None, None].expand(S, K, n_pix).reshape(-1),
            st[:, :, None].expand(S, K, n_pix).reshape(-1),
            ln[:, :, None].expand(S, K, n_pix).reshape(-1),
            ll,
        ).reshape(S, K, n_pix, -1)
        floor = z[:, None].expand(S, K, n_pix, -1)
        vmask = valid[:, :, None].expand(S, K, n_pix)
        for name, emb in (("ret", pred), ("floor", floor), ("fresh", tgt)):
            e = emb[vmask]
            acc[name]["sum"] += e.sum(0).double().cpu().numpy()
            acc[name]["sq"] += (e.double() ** 2).sum(0).cpu().numpy()
            acc[name]["n"] += e.shape[0]
            if name != "fresh":
                # window identification, per (sample, pixel)
                pe = F.normalize(emb.permute(0, 2, 1, 3), dim=-1)  # [S, P, K, D]
                te = F.normalize(tgt.permute(0, 2, 1, 3), dim=-1)
                sims = pe @ te.transpose(2, 3)  # [S, P, K(pred), K(fresh)]
                sims = sims.masked_fill(~valid[:, None, None, :], float("-inf"))
                hit = sims.argmax(-1) == torch.arange(K, device=device)
                ok = (
                    valid[:, None, :].expand(S, n_pix, K)
                    & (valid.sum(1) >= 2)[:, None, None]
                )
                acc[name]["id_hit"] += int(hit[ok].sum())
                acc[name]["id_n"] += int(ok.sum())
        # rows for the kNN comparison: a few random valid (window, pixel) per sample
        for i in range(S):
            ks = torch.nonzero(valid[i]).flatten()
            if len(ks) == 0:
                continue
            sel_k = ks[torch.randint(len(ks), (keep,), generator=g).to(device)]
            sel_p = torch.randint(n_pix, (keep,), generator=g).to(device)
            knn["ret"].append(pred[i, sel_k, sel_p].cpu())
            knn["floor"].append(floor[i, sel_k, sel_p].cpu())
            knn["fresh"].append(tgt[i, sel_k, sel_p].cpu())
    model.train()

    out: dict = {}
    total_var = {}
    for name, a in acc.items():
        mean = a["sum"] / a["n"]
        total_var[name] = float((a["sq"] / a["n"] - mean**2).sum())
    fresh_rows = F.normalize(torch.cat(knn["fresh"])[:knn_rows], dim=-1)
    fresh_nn = (fresh_rows @ fresh_rows.T).fill_diagonal_(-2).topk(k, dim=-1).indices
    for name in ("ret", "floor"):
        rows = F.normalize(torch.cat(knn[name])[:knn_rows], dim=-1)
        sims = (rows @ rows.T).fill_diagonal_(-2)
        nn_idx = sims.topk(k, dim=-1).indices
        overlap = (
            (nn_idx[:, :, None] == fresh_nn[:, None, :]).any(-1).float().mean().item()
        )
        out[name] = {
            "var_ratio": total_var[name] / total_var["fresh"],
            "knn_agreement": overlap,
            "window_id": acc[name]["id_hit"] / max(acc[name]["id_n"], 1),
        }
    return out


def log_geometry(label: str, geo: dict) -> None:
    """One line per model."""
    for name, m in geo.items():
        logger.info(
            f"{label} {name:5s}: var_ratio {m['var_ratio']:.3f}  knn@10 agreement {m['knn_agreement']:.3f}  "
            f"window_id {m['window_id']:.3f}"
        )


def centered_per_block(x: Tensor, n_pix: int) -> Tensor:
    """Rows in blocks of one (sample, window)'s pixels, minus each block's mean."""
    blocks = x.view(-1, n_pix, x.shape[-1])
    return (blocks - blocks.mean(1, keepdim=True)).view_as(x)


@torch.no_grad()
def evaluate(model: nn.Module, d: dict[str, Tensor], device: str, seed: int) -> dict:
    """Per-window-length metrics on every valid (sample, window, pixel) in d."""
    model.eval()
    pairs = examples(d)
    n_pix = d["z_T"].shape[1]
    g = torch.Generator(device="cpu").manual_seed(seed)
    stats: dict[int, dict[str, list]] = {}
    for chunk in pairs.split(256):
        p = chunk.repeat_interleave(n_pix, dim=0)
        pix = torch.arange(n_pix).repeat(len(chunk))
        z, tgt, sm, st, ln = (t.to(device) for t in gather(d, p, pix))
        ll = d["latlon"][p[:, 0]].to(device) if "latlon" in d else None
        pred = retrieve(model, z, sm, st, ln, ll)
        # Wrong-window control: same inputs, (start, length) shuffled within the chunk.
        perm = torch.randperm(len(st), generator=g).to(device)
        wrong = retrieve(model, z, sm, st[perm], ln[perm], ll)
        rows = {
            "cos_floor": F.cosine_similarity(z, tgt, dim=-1),
            "cos_ret": F.cosine_similarity(pred, tgt, dim=-1),
            "cos_wrong": F.cosine_similarity(wrong, tgt, dim=-1),
            "mse_floor": ((z - tgt) ** 2).mean(-1),
            "mse_ret": ((pred - tgt) ** 2).mean(-1),
        }
        # Within-tile error: deviations from each (sample, window)'s mean over its pixels.
        lt = centered_per_block(tgt, n_pix)
        rows["local_mse_floor"] = ((centered_per_block(z, n_pix) - lt) ** 2).mean(-1)
        rows["local_mse_ret"] = ((centered_per_block(pred, n_pix) - lt) ** 2).mean(-1)
        for L in ln.unique().tolist():
            sel = ln == L
            entry = stats.setdefault(L, {k: [] for k in rows})
            for k, v in rows.items():
                entry[k].append(v[sel].float().cpu())
    model.train()

    out: dict = {}
    for L in sorted(stats):
        m = {k: torch.cat(v).mean().item() for k, v in stats[L].items()}
        m["gap_closed"] = (m["cos_ret"] - m["cos_floor"]) / max(
            1 - m["cos_floor"], 1e-8
        )
        m["n"] = int(sum(len(x) for x in stats[L]["cos_ret"]))
        out[L] = m
    total = sum(m["n"] for m in out.values())
    out["all"] = {
        k: sum(m[k] * m["n"] for m in out.values()) / total
        for k in (
            "cos_floor",
            "cos_ret",
            "cos_wrong",
            "mse_floor",
            "mse_ret",
            "gap_closed",
            "local_mse_floor",
            "local_mse_ret",
        )
    }
    out["all"]["local_closed"] = 1 - out["all"]["local_mse_ret"] / max(
        out["all"]["local_mse_floor"], 1e-12
    )
    return out


def log_metrics(label: str, metrics: dict) -> None:
    """One line per window length."""
    logger.info(
        f"{label}: len  n        cos_floor cos_ret  cos_wrong gap_closed mse_floor mse_ret"
    )
    for L, m in metrics.items():
        n = m.get("n", "")
        logger.info(
            f"          {str(L):4s} {str(n):8s} {m['cos_floor']:.4f}    {m['cos_ret']:.4f}   "
            f"{m['cos_wrong']:.4f}    {m['gap_closed']:+.3f}     {m['mse_floor']:.5f}   {m['mse_ret']:.5f}"
        )
    if "local_closed" in metrics.get("all", {}):
        a = metrics["all"]
        logger.info(
            f"{label}: within-tile MSE floor {a['local_mse_floor']:.5f} ret {a['local_mse_ret']:.5f} "
            f"({100 * a['local_closed']:.1f}% closed)"
        )


def main() -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache_dir", nargs="+", required=True, help="calendar-year caches"
    )
    parser.add_argument(
        "--extra_train_cache_dir",
        nargs="*",
        default=[],
        help="other start months, train only",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--arch",
        choices=[
            "floor",
            "lowrank",
            "lowrank_full",
            "lowrank_lenonly",
            "film",
            "film_lenonly",
        ],
        required=True,
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--code_dim", type=int, default=64)
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout inside FiLM blocks"
    )
    parser.add_argument("--loss", choices=["mse", "cos"], default="mse")
    parser.add_argument(
        "--pixel_batches",
        action="store_true",
        help="batch whole pixels (all their windows) instead of independent (pixel, window) rows",
    )
    parser.add_argument(
        "--gram_pixels_weight",
        type=float,
        default=0.0,
        help="Gram across (pixel, window) rows",
    )
    parser.add_argument(
        "--gram_windows_weight",
        type=float,
        default=0.0,
        help="Gram across a pixel's own windows",
    )
    parser.add_argument(
        "--geometry",
        action="store_true",
        help="report var ratio, kNN agreement, window id",
    )
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument(
        "--min_steps", type=int, default=0, help="train at least this many steps"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="nested calendar-year subset (0 = all)",
    )
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument(
        "--tile_batches",
        type=int,
        default=0,
        help="batch this many whole tiles (all cached pixels and windows) instead of random pixels",
    )
    parser.add_argument(
        "--tile_centered_weight",
        type=float,
        default=0.0,
        help="tile batches: within-tile MSE",
    )
    parser.add_argument(
        "--tile_infonce_weight",
        type=float,
        default=0.0,
        help="tile batches: within-tile InfoNCE",
    )
    parser.add_argument(
        "--tile_kl_weight",
        type=float,
        default=0.0,
        help="tile batches: within-tile relational KL",
    )
    parser.add_argument(
        "--tile_window_cka_weight",
        type=float,
        default=0.0,
        help="tile batches: 1 - CKA over (pixel, window) rows",
    )
    parser.add_argument(
        "--tile_window_kl_weight",
        type=float,
        default=0.0,
        help="tile batches: neighbour KL over (pixel, window) rows",
    )
    parser.add_argument(
        "--tile_terms_primary_only",
        action="store_true",
        help="within-tile terms only on --cache_dir tiles (pixels spread over the whole tile), not --extra_train_cache_dir patches",
    )
    parser.add_argument(
        "--dense_dirs",
        nargs="*",
        default=[],
        help="train from these dense caches (val / test from --cache_dir)",
    )
    parser.add_argument("--dense_steps", type=int, default=100000)
    parser.add_argument(
        "--dense_sampling",
        choices=["buffer", "random"],
        default="buffer",
        help="buffer: shuffle buffer of whole chunks; random: memory-mapped, uniform over every sample",
    )
    parser.add_argument(
        "--dense_segment",
        type=int,
        default=32,
        help="random sampling: pixels are drawn in runs of this many along a row (one 4 KB page per array at 32)",
    )
    parser.add_argument(
        "--latlon",
        action="store_true",
        help="condition the retriever on each pixel's lat/lon",
    )
    parser.add_argument(
        "--latlon_file",
        default="/weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
        "cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_"
        "worldpop_wri_canopy_height_map/1138828/latlon_distribution.npy",
        help="per-sample lat/lon of the pretraining dataset, indexed by sample index",
    )
    parser.add_argument(
        "--dense_workers",
        type=int,
        default=32,
        help="random sampling: DataLoader worker processes",
    )
    parser.add_argument(
        "--eval_only",
        help="skip training: load this model.pt and evaluate on val / test",
    )
    parser.add_argument("--dense_buffer_chunks", type=int, default=32)
    parser.add_argument("--dense_refresh_every", type=int, default=25)
    parser.add_argument(
        "--dense_random_pixels",
        type=int,
        default=1024,
        help="pixels from random tiles per batch",
    )
    parser.add_argument(
        "--dense_tiles", type=int, default=4, help="whole-tile blocks per batch"
    )
    parser.add_argument(
        "--dense_tile_pixels", type=int, default=256, help="pixels per whole-tile block"
    )
    parser.add_argument(
        "--dense_windows",
        type=int,
        default=4,
        help="windows per pixel per batch (of the 8 stored)",
    )
    parser.add_argument(
        "--select_metric",
        choices=["cos", "local", "mean"],
        default="mean",
        help="checkpoint selection on val: cosine, within-tile closed, or the mean of global and within-tile closed",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.07,
        help="temperature for the within-tile InfoNCE / KL",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="evals without val improvement before stopping",
    )
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument(
        "--no_quantize", action="store_true", help="skip the int8 round trip"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantize = not args.no_quantize
    train, val, test = split(
        load_cache(args.cache_dir, quantize), args.val_frac, args.test_frac, args.seed
    )
    latlon_table = None
    if args.latlon:
        if not args.dense_dirs:
            raise ValueError("--latlon is implemented for --dense_dirs training only")
        table = torch.from_numpy(np.load(args.latlon_file)).float()
        for d in (val, test):
            d["latlon"] = table[d["index"]]
        latlon_table = table.to(device)
        logger.info(f"lat/lon input from {args.latlon_file} ({len(table)} samples)")
    stream = None
    if args.dense_dirs:
        # Train on the dense cache; val / test stay the calendar cache's held-out
        # samples (same pixels as every earlier run), excluded from the stream.
        held_out = {int(i) for i in torch.cat([val["index"], test["index"]]).tolist()}
        if args.dense_sampling == "random":
            spec = (
                args.dense_random_pixels,
                args.dense_tiles,
                args.dense_tile_pixels,
                args.dense_windows,
            )
            stream = DenseRandom(
                args.dense_dirs,
                held_out,
                args.seed,
                spec,
                args.dense_workers,
                args.dense_segment,
            )
        else:
            stream = DenseStream(
                args.dense_dirs,
                held_out,
                args.dense_buffer_chunks,
                args.dense_refresh_every,
                args.seed,
            )
        train = None
    if train is not None and args.max_train_samples:
        # split() already shuffled; the first N are nested across N.
        train = subset(
            train, torch.arange(min(args.max_train_samples, train["z_T"].shape[0]))
        )
    if train is not None:
        train["tile_ok"] = torch.ones(train["z_T"].shape[0], dtype=torch.bool)
    if train is not None and args.extra_train_cache_dir:
        extra = load_cache(
            args.extra_train_cache_dir, quantize, n_pixels=train["z_T"].shape[1]
        )
        extra["tile_ok"] = torch.full(
            (extra["z_T"].shape[0],), not args.tile_terms_primary_only
        )
        train = concat(train, extra)
    if train is not None:
        train = {k: v.to(device) for k, v in train.items()}
        pairs = examples(train).to(device)
        n_pix = train["z_T"].shape[1]
        n_examples = len(pairs) * n_pix
        steps = max(
            args.min_steps, math.ceil(args.epochs * n_examples / args.batch_size)
        )
        logger.info(
            f"train: {train['z_T'].shape[0]} samples ({len(pairs)} windows x {n_pix} px = {n_examples} examples); "
            f"val {val['z_T'].shape[0]}, test {test['z_T'].shape[0]} calendar-year samples; "
            f"{steps} steps ({args.epochs} epochs)"
        )
    else:
        steps = args.dense_steps
        logger.info(
            f"train: dense stream; val {val['z_T'].shape[0]}, test {test['z_T'].shape[0]}; {steps} steps"
        )

    model = build(args, val["z_T"].shape[-1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = args.run_name or (
        args.arch if n_params == 0 else f"{args.arch}_{int(time.time())}"
    )
    run_dir = os.path.join(args.out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"arch {args.arch}: {n_params} parameters -> {run_dir}")

    history = []
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device))
    best_step, best_cos, best_state = 0, -1.0, copy.deepcopy(model.state_dict())
    if n_params > 0 and not args.eval_only:
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=steps, pct_start=0.05
        )
        stale = 0
        t_log, data_wait = time.time(), 0.0
        for step in range(1, steps + 1):
            if stream is not None:
                t_wait = time.time()
                b = stream.batch(
                    args.dense_random_pixels,
                    args.dense_tiles,
                    args.dense_tile_pixels,
                    args.dense_windows,
                    device,
                )
                data_wait += time.time() - t_wait
                preds, tgts = [], []
                if "random" in b:
                    z, tgt, sm, st, ln, valid, ids = b["random"]
                    ll = latlon_table[ids] if latlon_table is not None else None
                    pred = retrieve_windows(model, z, sm, st, ln, ll)
                    preds.append(pred[valid])
                    tgts.append(tgt[valid])
                tile_loss = 0.0
                if "tiles" in b:
                    z, tgt, sm, st, ln, valid, ids = b["tiles"]
                    T, P, K, D = tgt.shape
                    ll = (
                        latlon_table[ids].repeat_interleave(P, dim=0)
                        if latlon_table is not None
                        else None
                    )
                    pred = retrieve_windows(
                        model,
                        z.reshape(T * P, D),
                        sm.repeat_interleave(P),
                        st.repeat_interleave(P, dim=0),
                        ln.repeat_interleave(P, dim=0),
                        ll,
                    ).reshape(T, P, K, D)
                    vmask = valid[:, None, :].expand(T, P, K)
                    preds.append(pred[vmask])
                    tgts.append(tgt[vmask])
                    tile_loss = tile_terms(pred, tgt, valid, args)
                loss = loss_fn(torch.cat(preds), torch.cat(tgts), args.loss) + tile_loss
            elif args.tile_batches:
                ok, z, tgt, sm, st, ln, valid = tile_batch(
                    train, args.tile_batches, device
                )
                T, P, K, D = tgt.shape
                pred = retrieve_windows(
                    model,
                    z.reshape(T * P, D),
                    sm.repeat_interleave(P),
                    st.repeat_interleave(P, dim=0),
                    ln.repeat_interleave(P, dim=0),
                ).reshape(T, P, K, D)
                vmask = valid[:, None, :].expand(T, P, K)
                loss = loss_fn(pred[vmask], tgt[vmask], args.loss)
                # Within-tile terms only on tiles whose pixels span the tile (see --tile_terms_primary_only).
                loss = loss + tile_terms(pred[ok], tgt[ok], valid[ok], args)
            elif args.pixel_batches:
                K = train["z_w"].shape[1]
                z, tgt, sm, st, ln, valid = pixel_batch(
                    train, args.batch_size // K, device
                )
                pred = retrieve_windows(model, z, sm, st, ln)
                loss = loss_fn(pred[valid], tgt[valid], args.loss)
                if args.gram_pixels_weight:
                    loss = loss + args.gram_pixels_weight * gram_across_pixels(
                        pred, tgt, valid
                    )
                if args.gram_windows_weight:
                    loss = loss + args.gram_windows_weight * gram_across_windows(
                        pred, tgt, valid
                    )
            else:
                p = pairs[torch.randint(len(pairs), (args.batch_size,), device=device)]
                pix = torch.randint(n_pix, (args.batch_size,), device=device)
                z, tgt, sm, st, ln = gather(train, p, pix)
                loss = loss_fn(retrieve(model, z, sm, st, ln), tgt, args.loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if step % 200 == 0:
                rate = 200 / (time.time() - t_log)
                logger.info(
                    f"step {step}/{steps} loss {loss.item():.5f} ({rate:.1f} steps/s, data wait {100 * data_wait / max(time.time() - t_log, 1e-9):.0f}%)"
                )
                t_log, data_wait = time.time(), 0.0
            if step % args.eval_every == 0 or step == steps:
                m = evaluate(model, val, device, args.seed)
                history.append({"step": step, "val_all": m["all"]})
                cos = {
                    "cos": m["all"]["cos_ret"],
                    "local": m["all"]["local_closed"],
                    "mean": (m["all"]["gap_closed"] + m["all"]["local_closed"]) / 2,
                }[args.select_metric]
                logger.info(
                    f"step {step}: val {args.select_metric} {cos:.4f} (best {max(best_cos, cos):.4f}), "
                    f"within-tile {100 * m['all']['local_closed']:.1f}% closed"
                )
                if cos > best_cos:
                    best_step, best_cos, best_state, stale = (
                        step,
                        cos,
                        copy.deepcopy(model.state_dict()),
                        0,
                    )
                else:
                    stale += 1
                    if stale >= args.patience:
                        logger.info(f"early stop at step {step}; best step {best_step}")
                        break
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(run_dir, "model.pt"))

    val_m = evaluate(model, val, device, args.seed)
    test_m = evaluate(model, test, device, args.seed)
    log_metrics(f"VAL (best step {best_step})", val_m)
    log_metrics(f"TEST (best step {best_step})", test_m)
    geo = None
    if args.geometry:
        geo = {
            "val": geometry(model, val, device, args.seed),
            "test": geometry(model, test, device, args.seed),
        }
        log_geometry("VAL ", geo["val"])
        log_geometry("TEST", geo["test"])
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(
            {
                "args": vars(args),
                "best_step": best_step,
                "steps": steps,
                "val": val_m,
                "test": test_m,
                "geometry": geo,
                "history": history,
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
