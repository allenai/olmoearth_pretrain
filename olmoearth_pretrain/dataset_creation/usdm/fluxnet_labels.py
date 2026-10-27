"""USDM drought state and evolution at the FLUXNET eval's tower-anchors.

For every window of the FLUXNET dataset (``meta/windows_all.parquet``: the
target is the mean over ``[anchor - 8, anchor)``) whose tower lies in the USDM
CONUS domain: the severity of the tower's 0.1 deg cell in the map valid for the
week containing ``anchor - 1`` (the latest map dated on or before that day),
and its changes over the 4 / 8 weeks before and after that map. Lets the FLUXNET
probes' errors (``rgpp``, ``gpp_anom``) be split by drought state.

Output: ``<usdm root>/fluxnet_drought.parquet`` keyed on ``(site_id, anchor)``
with ``map_date``, ``sev``, ``dpast4``, ``dpast8``, ``dfut4``, ``dfut8``,
``drought`` (``sev >= 2``) and the tower's grid ``row`` / ``col``.

Usage::

    python -m olmoearth_pretrain.dataset_creation.usdm.fluxnet_labels
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .common import DEFAULT_ROOT, MIN_INSIDE_SUBCELLS, grid_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLUXNET_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/fluxnet")
LAGS = (-8, -4, 4, 8)


def run(root: Path, flux_root: Path) -> pd.DataFrame:
    """Join USDM severities onto the FLUXNET windows.

    Args:
        root: USDM dataset root.
        flux_root: FLUXNET dataset root (read only).

    Returns:
        The written table.
    """
    meta = root / "meta"
    sums = np.load(meta / "usdm_sum.npy", mmap_mode="r")
    inside = np.load(meta / "usdm_inside.npy")
    dates = np.load(meta / "usdm_dates.npy")
    windows = pd.read_parquet(flux_root / "meta" / "windows_all.parquet")[
        ["site_id", "anchor"]
    ]
    sites = pd.read_parquet(flux_root / "meta" / "sites_split.parquet")[
        ["site_id", "lat", "lon"]
    ]
    cells = {}
    for site in sites.itertuples():
        idx = grid_index(float(site.lat), float(site.lon))
        if idx is not None and inside[idx] >= MIN_INSIDE_SUBCELLS:
            cells[site.site_id] = idx
    frame = windows[windows["site_id"].isin(cells)].reset_index(drop=True)
    rows = np.array([cells[s][0] for s in frame["site_id"]])
    cols = np.array([cells[s][1] for s in frame["site_id"]])
    k = np.searchsorted(dates, frame["anchor"].to_numpy() - 1, side="right") - 1
    ok = (k >= 0) & (frame["anchor"].to_numpy() - 1 - dates[np.maximum(k, 0)] < 7)
    frame, rows, cols, k = frame[ok].reset_index(drop=True), rows[ok], cols[ok], k[ok]

    def severity(index: np.ndarray) -> np.ndarray:
        """Cell severity in map ``index`` per row.

        Args:
            index: Map index per row (inside the record).

        Returns:
            Severity per row.
        """
        return (
            np.asarray(sums[index, rows, cols], dtype=np.float32) / inside[rows, cols]
        )

    frame["map_date"] = pd.to_datetime(dates[k], unit="D").date
    frame["sev"] = severity(k)
    for lag in LAGS:
        j = np.clip(k + lag, 0, len(dates) - 1)
        # Defined only when the map ``lag`` weeks away exists (no gap).
        exact = (k + lag == j) & (dates[j] - dates[k] == 7 * lag)
        shifted = np.where(exact, severity(j), np.nan)
        name = f"dpast{-lag}" if lag < 0 else f"dfut{lag}"
        frame[name] = frame["sev"] - shifted if lag < 0 else shifted - frame["sev"]
    frame["drought"] = frame["sev"] >= 2.0
    frame["row"], frame["col"] = rows, cols
    out = root / "fluxnet_drought.parquet"
    frame.to_parquet(out)
    logger.info(
        "%d of %d FLUXNET windows (%d of %d towers) in the USDM domain; drought share %.3f; wrote %s",
        len(frame), len(windows), frame["site_id"].nunique(), windows["site_id"].nunique(),
        float(frame["drought"].mean()), out,
    )  # fmt: skip
    return frame


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--flux-root", default=str(FLUXNET_ROOT))
    args = ap.parse_args()
    run(Path(args.root), Path(args.flux_root))


if __name__ == "__main__":
    main()
