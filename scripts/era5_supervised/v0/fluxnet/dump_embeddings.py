"""Dump frozen ERA5 encoder embeddings of the FLUXNET eval windows (CPU).

So new target definitions can be probed offline in seconds instead of
re-running encoders: for each checkpoint (or random init) of a queue, the 7000
``eval`` windows go through the encoder once and three poolings are saved:
``mean`` (the encoder's own pooled vector, what every eval probes), ``last1``
(the last patch token, final 14 days) and ``last4`` (mean of the last 4 tokens,
final 35 days), the token poolings with the pooled vector's parameter-free
LayerNorm.

Windows are read from the per-site ERA5 series the rslearn windows were
written from (identical values) and preprocessed as ``Era5TaskDataset`` does:
z-score with the pretraining stats, no-data cells zeroed after normalization,
``[day-of-year, month0, year]`` per day. Output:
``<root>/embeddings/<name>.npz`` (float16 ``mean``/``last1``/``last4``) and
``<root>/embeddings/windows.parquet`` (row order: site, anchor, split).

Usage (repo root)::

    python scripts/era5_supervised/v0/fluxnet/dump_embeddings.py --queue <queue.json> --shard 0/2
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cpu_probe import CKPT_ROOT, FLUX_ROOT, load_weights, model_config
from olmo_core.utils import seed_all

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.normalize import Normalizer, Strategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NODATA = -9999.0
WINDOW_DAYS = 448
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def eval_inputs(
    root: Path,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocessed encoder inputs of every eval window.

    Args:
        root: Dataset root.

    Returns:
        ``(windows, era5 [N, 448, 14], timestamps [N, 448, 3], valid [N, 448, 14])``.
    """
    windows = pd.read_parquet(root / "meta" / "windows_eval.parquet").reset_index(
        drop=True
    )
    normalizer = Normalizer(Strategy.ZSCORE)
    era5 = np.zeros((len(windows), WINDOW_DAYS, 14), np.float32)
    valid = np.zeros_like(era5, dtype=bool)
    for site_id, rows in windows.groupby("site_id"):
        series = np.load(root / "meta" / "era5" / f"{site_id}.npy")
        for i, t0 in zip(rows.index, rows["t0"]):
            raw = series[int(t0) : int(t0) + WINDOW_DAYS]
            mask = raw != NODATA
            norm = np.asarray(
                normalizer.normalize(Modality.ERA5L_DAY_10, raw.copy()), np.float32
            )
            norm[~mask] = 0.0
            era5[i], valid[i] = norm, mask
    stamps = np.zeros((len(windows), WINDOW_DAYS, 3), np.int64)
    for i, anchor in enumerate(windows["anchor"]):
        start = EPOCH + timedelta(days=int(anchor) - WINDOW_DAYS)
        days = [start + timedelta(days=k) for k in range(WINDOW_DAYS)]
        stamps[i] = [[d.timetuple().tm_yday, d.month - 1, d.year] for d in days]
    return (
        windows,
        torch.from_numpy(era5),
        torch.from_numpy(stamps),
        torch.from_numpy(valid),
    )


def embed(encoder: torch.nn.Module, era5: torch.Tensor, stamps: torch.Tensor,
          valid: torch.Tensor, batch: int) -> dict[str, np.ndarray]:  # fmt: skip
    """Pooled embeddings of all windows.

    Args:
        encoder: Frozen encoder.
        era5: ``[N, 448, 14]`` normalized inputs.
        stamps: ``[N, 448, 3]`` timestamps.
        valid: ``[N, 448, 14]`` validity mask.
        batch: Batch size.

    Returns:
        ``{"mean", "last1", "last4"}`` float16 arrays ``[N, D]``.
    """
    out: dict[str, list[torch.Tensor]] = {"mean": [], "last1": [], "last4": []}
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(era5), batch):
            with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
                res = encoder(era5=era5[i : i + batch], timestamps=stamps[i : i + batch],
                              valid_mask=valid[i : i + batch])  # fmt: skip
            tokens = res["tokens"].float()
            width = tokens.shape[-1]
            out["mean"].append(res["pooled"].float())
            out["last1"].append(F.layer_norm(tokens[:, -1], (width,)))
            out["last4"].append(F.layer_norm(tokens[:, -4:].mean(dim=1), (width,)))
    return {k: torch.cat(v).numpy().astype(np.float16) for k, v in out.items()}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--queue", required=True)
    ap.add_argument("--root", default=str(FLUX_ROOT))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--threads", type=int, default=56)
    ap.add_argument("--batch", type=int, default=256)
    # Accepted for launcher compatibility; unused.
    ap.add_argument("--tasks", default="")
    ap.add_argument("--lr-grid", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    root = Path(args.root)
    out_dir = root / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    windows, era5, stamps, valid = eval_inputs(root)
    windows[["site_id", "anchor", "split"]].to_parquet(out_dir / "windows.parquet")
    index, count = (int(v) for v in args.shard.split("/"))
    for entry in json.loads(Path(args.queue).read_text())[index::count]:
        target = out_dir / f"{entry['name']}.npz"
        if target.exists():
            continue
        run_dir = CKPT_ROOT / entry["source"]
        config = model_config(
            run_dir / f"step{int(entry.get('load_step', 50000))}" / "config.json"
        )
        seed_all(int(entry.get("init_seed", 0)))
        model = config.build()
        if not entry.get("random_init", False):
            load_weights(model, run_dir / f"step{int(entry['step'])}")
        np.savez(target, **embed(model.encoder, era5, stamps, valid, args.batch))
        logger.info("wrote %s", target)


if __name__ == "__main__":
    main()
