"""Test metrics for the PASTIS changed-year benchmark (pastis_changed_s2).

The test split mixes three sets of windows: the official 2019 PASTIS test fold, and
the train-fold and test-fold places re-imaged in a year when their crop changed. The
plain test mIoU averages over all of them; these metrics score each set on its own and
measure memorisation: on pixels whose crop changed, how often the model still outputs
the 2019 class, at places it trained on versus places it never saw.

The dataset root holds a sidecar, ``changed_year_scoring.npz``, with one row per test
window:
  label            int8 (N, S, S)  the window's test label as the loader yields it
                                   (void = -1); used to match predictions to windows
  label_2019       int8 (N, S, S)  2019 crop class (1..) where the 2019 label is
                                   trusted for the change test, else -1
  bench            str  (N,)       ref2019 | changed_trainloc | changed_testloc
  year             int  (N,)       label year
Tiles of one window are consecutive in the test predictions, so they are stitched back
to S x S and matched to a row by label content.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial
from logging import getLogger

import numpy as np
import torch
from upath import UPath

logger = getLogger(__name__)

SIDECAR = "changed_year_scoring.npz"
DATASETS = {"pastis_changed_s2"}
BENCH_KEYS = {
    "ref2019": "ref2019",
    "changed_trainloc": "trainloc",
    "changed_testloc": "testloc",
}

ExtraTestMetrics = Callable[[torch.Tensor, torch.Tensor, int], dict[str, float]]


def scorer_for_dataset(dataset: str) -> ExtraTestMetrics | None:
    """Return the changed-year scorer for a dataset that has one, else None."""
    if dataset not in DATASETS:
        return None
    from olmoearth_pretrain.evals.studio_ingest import get_dataset_entry

    path = UPath(get_dataset_entry(dataset).weka_path) / SIDECAR
    return partial(changed_year_metrics, sidecar_path=str(path))


def _key(a: np.ndarray) -> str:
    return hashlib.md5(a.astype(np.int8).tobytes(), usedforsecurity=False).hexdigest()


def _stitch(a: np.ndarray, size: int) -> np.ndarray:
    ws = a.shape[-1]
    t = size // ws
    return a.reshape(-1, t, t, ws, ws).transpose(0, 1, 3, 2, 4).reshape(-1, size, size)


def _miou(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(float)
    union = confusion.sum(0) + confusion.sum(1) - tp
    return (
        float((tp[union > 0] / union[union > 0]).mean()) if (union > 0).any() else 0.0
    )


def changed_year_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    sidecar_path: str,
) -> dict[str, float]:
    """Per-set mIoU and changed/unchanged-pixel scores, keyed cy_<set>[_<year>]_<metric>."""
    with UPath(sidecar_path).open("rb") as f:
        side = dict(np.load(f))
    size = side["label"].shape[-1]
    P, L = preds.cpu().numpy(), labels.cpu().numpy()
    ws = P.shape[-1]
    t = size // ws if ws and size % ws == 0 else 0
    if P.ndim != 3 or not t or len(P) % (t * t):
        logger.warning(
            f"changed-year metrics skipped: predictions {P.shape} do not tile {size}x{size}"
        )
        return {}
    P, L = _stitch(P, size), _stitch(L, size)

    rows: dict[str, list[int]] = {}
    for i, lab in enumerate(side["label"]):
        rows.setdefault(_key(lab), []).append(i)

    n = num_classes
    acc: dict[str, dict[str, np.ndarray | int]] = {}
    unmatched = 0
    for p, lab in zip(P, L):
        hit = rows.get(_key(lab), [])
        if len(hit) != 1:
            unmatched += 1
            continue
        i = hit[0]
        bench = BENCH_KEYS[str(side["bench"][i])]
        l19 = side["label_2019"][i].astype(np.int64)
        lab = lab.astype(np.int64)
        valid = lab >= 0
        scored = (l19 >= 1) & (lab >= 1)
        chg, unc = scored & (lab != l19), scored & (lab == l19)
        groups = (
            [bench]
            if bench == "ref2019"
            else [bench, f"{bench}_{int(side['year'][i])}"]
        )
        for g in groups:
            a = acc.setdefault(
                g,
                dict(
                    cm=np.zeros((n, n), np.int64),
                    windows=0,
                    chg=0,
                    chg_ok=0,
                    chg_19=0,
                    unc=0,
                    unc_ok=0,
                ),
            )
            a["cm"] += np.bincount(lab[valid] * n + p[valid], minlength=n * n).reshape(
                n, n
            )
            a["windows"] += 1
            a["chg"] += int(chg.sum())
            a["chg_ok"] += int((chg & (p == lab)).sum())
            a["chg_19"] += int((chg & (p == l19)).sum())
            a["unc"] += int(unc.sum())
            a["unc_ok"] += int((unc & (p == lab)).sum())

    out: dict[str, float] = {"cy_unmatched_windows": float(unmatched)}
    if unmatched:
        logger.warning(
            f"changed-year metrics: {unmatched} of {len(P)} windows not matched"
        )
    for g, a in sorted(acc.items()):
        per_year = g.count("_") == 1
        if not per_year:
            out[f"cy_{g}_miou"] = _miou(a["cm"])  # type: ignore[arg-type]
            out[f"cy_{g}_windows"] = float(a["windows"])  # type: ignore[arg-type]
        if g != "ref2019":
            chg = max(int(a["chg"]), 1)  # type: ignore[call-overload]
            out[f"cy_{g}_changed_acc"] = a["chg_ok"] / chg  # type: ignore[operator]
            out[f"cy_{g}_changed_pred_2019"] = a["chg_19"] / chg  # type: ignore[operator]
            if not per_year:
                out[f"cy_{g}_unchanged_acc"] = a["unc_ok"] / max(int(a["unc"]), 1)  # type: ignore[operator, call-overload]
    return out
