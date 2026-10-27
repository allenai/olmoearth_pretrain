"""Snow phenology labels for the USDM and FLUXNET evals.

* ``usdm``: every seasonal-snow cell of the phenology table inside the USDM
  CONUS grid, per water year (October-September, the USDM eval's convention):
  ``sdd``, ``sad``, ``scd`` and their leave-one-year-out anomalies, with the
  USDM grid ``row`` / ``col`` (``usdm.common.grid_index``). Lets the USDM
  probes' errors be split by snow drought.
* ``fluxnet``: per FLUXNET tower (``meta/sites_split.parquet``) whose 0.1 deg
  cell has seasonal snow, per water year: the same columns. Lets the FLUXNET
  probes' spring errors be split by melt timing.

Both cover every cell of the phenology table (snow season in at least half the
water years), not only the snow eval's sampled cells.

Outputs: ``<root>/usdm_snow.parquet`` and ``<root>/fluxnet_snow.parquet``.

Usage::

    python -m olmoearth_pretrain.dataset_creation.snow.partner_labels usdm fluxnet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..usdm.common import grid_index
from .common import DEFAULT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FLUXNET_ROOT = Path("/weka/dfive-default/rslearn-eai/datasets/fluxnet")
COLUMNS = [
    "cell", "lat10", "lon10", "wy", "sdd", "sad", "scd", "sdd_dowy", "sad_dowy",
    "season", "peak", "sdd_gap", "sdd_dowy_clim", "sdd_dowy_anom", "sad_dowy_anom",
    "scd_clim", "scd_anom", "season_anom",
]  # fmt: skip


def usdm(root: Path) -> pd.DataFrame:
    """Snow labels of every CONUS seasonal-snow cell.

    Args:
        root: Snow dataset root.

    Returns:
        The written table.
    """
    phen = pd.read_parquet(root / "meta" / "phenology.parquet", columns=COLUMNS)
    cells = phen.drop_duplicates("cell")[["cell", "lat10", "lon10"]]
    index = {c: grid_index(a / 10.0, b / 10.0)
             for c, a, b in zip(cells["cell"], cells["lat10"], cells["lon10"])}  # fmt: skip
    keep = {c: rc for c, rc in index.items() if rc is not None}
    out = phen[phen["cell"].isin(keep)].copy()
    out["row"] = out["cell"].map(lambda c: keep[c][0])
    out["col"] = out["cell"].map(lambda c: keep[c][1])
    out["lat"], out["lon"] = out["lat10"] / 10.0, out["lon10"] / 10.0
    out.to_parquet(root / "usdm_snow.parquet")
    logger.info(
        "usdm: %d cells x water years, %d cells", len(out), out["cell"].nunique()
    )
    return out


def fluxnet(root: Path, flux_root: Path) -> pd.DataFrame:
    """Snow labels at the FLUXNET towers' cells.

    Args:
        root: Snow dataset root.
        flux_root: FLUXNET dataset root (read only).

    Returns:
        The written table.
    """
    phen = pd.read_parquet(root / "meta" / "phenology.parquet", columns=COLUMNS)
    sites = pd.read_parquet(flux_root / "meta" / "sites_split.parquet")
    sites = sites[["site_id", "lat", "lon"]].dropna()
    lat10 = np.rint(sites["lat"].to_numpy() * 10).astype(int)
    lon10 = np.rint(((sites["lon"].to_numpy() + 180) % 360 - 180) * 10).astype(int)
    lon10 = np.where(lon10 >= 1800, lon10 - 3600, lon10)
    sites["cell"] = [f"c{a:+05d}{b:+06d}" for a, b in zip(lat10, lon10)]
    out = sites.merge(phen, on="cell")
    out.to_parquet(root / "fluxnet_snow.parquet")
    logger.info("fluxnet: %d tower x water years, %d towers with seasonal snow",
                len(out), out["site_id"].nunique())  # fmt: skip
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("targets", nargs="+", choices=["usdm", "fluxnet"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--fluxnet-root", default=str(FLUXNET_ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    if "usdm" in args.targets:
        usdm(root)
    if "fluxnet" in args.targets:
        fluxnet(root, Path(args.fluxnet_root))


if __name__ == "__main__":
    main()
