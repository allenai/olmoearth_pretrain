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


def load_cache(cache_dirs: list[str], quantize: bool) -> dict[str, Tensor]:
    """Concatenate all chunks under the given dirs; optionally int8 round-trip.

    Embeddings are kept float16; the round trip is applied per chunk in float32.
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


class WindowCode(nn.Module):
    """Embedding of (input start month, window start/end month-of-year, length).

    With ``length_only`` it sees only the window length.
    """

    def __init__(self, code_dim: int, length_only: bool = False) -> None:
        """Summed per-field embeddings, then an MLP."""
        super().__init__()
        self.length_only = length_only
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

    def forward(self, start_month: Tensor, start: Tensor, length: Tensor) -> Tensor:
        """Code for each row's window."""
        c = self.length(length)
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
    ) -> None:
        """Initialize at the floor: the output gate starts at zero."""
        super().__init__()
        self.code = WindowCode(code_dim, length_only)
        self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
        self.blocks = nn.ModuleList(
            FiLMBlock(width, code_dim, dropout) for _ in range(depth)
        )
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, dim))
        self.alpha = nn.Linear(code_dim, dim)
        nn.init.zeros_(self.alpha.weight)
        nn.init.zeros_(self.alpha.bias)

    def forward(
        self, z: Tensor, start_month: Tensor, start: Tensor, length: Tensor
    ) -> Tensor:
        """Z + α(c) · Δ(z, c)."""
        c = self.code(start_month, start, length)
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
        )
    raise ValueError(args.arch)


def retrieve(
    model: nn.Module, z: Tensor, start_month: Tensor, start: Tensor, length: Tensor
) -> Tensor:
    """Retrieval, with the full 12 months returning z exactly."""
    out = model(z, start_month, start, length)
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


def retrieve_windows(
    model: nn.Module, z: Tensor, sm: Tensor, st: Tensor, ln: Tensor
) -> Tensor:
    """Retrieve every window of every pixel: ``[P, D]`` + ``[P, K]`` fields -> ``[P, K, D]``."""
    P, K = st.shape
    flat = retrieve(
        model,
        z.repeat_interleave(K, dim=0),
        sm.repeat_interleave(K),
        st.reshape(-1),
        ln.reshape(-1),
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
        pred = retrieve(
            model,
            rows_z,
            sm[:, None, None].expand(S, K, n_pix).reshape(-1),
            st[:, :, None].expand(S, K, n_pix).reshape(-1),
            ln[:, :, None].expand(S, K, n_pix).reshape(-1),
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
        pred = retrieve(model, z, sm, st, ln)
        # Wrong-window control: same inputs, (start, length) shuffled within the chunk.
        perm = torch.randperm(len(st), generator=g).to(device)
        wrong = retrieve(model, z, sm, st[perm], ln[perm])
        rows = {
            "cos_floor": F.cosine_similarity(z, tgt, dim=-1),
            "cos_ret": F.cosine_similarity(pred, tgt, dim=-1),
            "cos_wrong": F.cosine_similarity(wrong, tgt, dim=-1),
            "mse_floor": ((z - tgt) ** 2).mean(-1),
            "mse_ret": ((pred - tgt) ** 2).mean(-1),
        }
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
        )
    }
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
    if args.max_train_samples:
        # split() already shuffled; the first N are nested across N.
        train = subset(
            train, torch.arange(min(args.max_train_samples, train["z_T"].shape[0]))
        )
    if args.extra_train_cache_dir:
        train = concat(train, load_cache(args.extra_train_cache_dir, quantize))
    train = {k: v.to(device) for k, v in train.items()}
    pairs = examples(train).to(device)
    n_pix = train["z_T"].shape[1]
    n_examples = len(pairs) * n_pix
    steps = max(args.min_steps, math.ceil(args.epochs * n_examples / args.batch_size))
    logger.info(
        f"train: {train['z_T'].shape[0]} samples ({len(pairs)} windows x {n_pix} px = {n_examples} examples); "
        f"val {val['z_T'].shape[0]}, test {test['z_T'].shape[0]} calendar-year samples; "
        f"{steps} steps ({args.epochs} epochs)"
    )

    model = build(args, train["z_T"].shape[-1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    name = args.run_name or (
        args.arch if n_params == 0 else f"{args.arch}_{int(time.time())}"
    )
    run_dir = os.path.join(args.out_dir, name)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"arch {args.arch}: {n_params} parameters -> {run_dir}")

    history = []
    best_step, best_cos, best_state = 0, -1.0, copy.deepcopy(model.state_dict())
    if n_params > 0:
        opt = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=steps, pct_start=0.05
        )
        stale = 0
        for step in range(1, steps + 1):
            if args.pixel_batches:
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
                logger.info(f"step {step}/{steps} loss {loss.item():.5f}")
            if step % args.eval_every == 0 or step == steps:
                m = evaluate(model, val, device, args.seed)
                history.append({"step": step, "val_all": m["all"]})
                cos = m["all"]["cos_ret"]
                logger.info(
                    f"step {step}: val cos {cos:.4f} (best {max(best_cos, cos):.4f})"
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
