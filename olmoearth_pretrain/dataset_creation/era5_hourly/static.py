"""Stage 3: the time-invariant ERA5-Land fields on the hourly store's grid.

The hourly store has none of them (except ``lshf``), so they come from the
EarthDataHub daily store (1801 latitudes, 90N..90S), cropped to the hourly
store's 1472 rows (90N..57.1S): a window's ``ilat, ilon`` indexes them
directly. ``lshf`` has a time axis there; its first step is taken.

Outputs under ``<root>/static``: ``static.npy`` ``(n_fields, 1472, 3600)``
float32 (``NODATA`` where the store has no value) and ``static.json``
(field order, units, names).

Usage::

    python -m olmoearth_pretrain.dataset_creation.era5_hourly.static
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np

from .common import (
    DAILY_URL,
    DEFAULT_ROOT,
    GRID_DEG,
    N_LAT,
    NODATA,
    STATIC_FIELDS,
    open_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(root: Path) -> None:
    """Download the static fields (kept if already there).

    Args:
        root: Output root.
    """
    out_dir = root / "static"
    if (out_dir / "static.npy").exists():
        logger.info("%s exists; nothing to do", out_dir / "static.npy")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    group = open_store(DAILY_URL)
    lat = np.asarray(group["latitude"][:N_LAT])
    assert np.allclose(lat, 90.0 - GRID_DEG * np.arange(N_LAT), atol=1e-6), "daily grid"
    fields, meta = [], []
    for name in STATIC_FIELDS:
        array = group[name]
        values = array[0, :N_LAT, :] if array.ndim == 3 else array[:N_LAT, :]
        values = np.asarray(values, dtype=np.float32)
        fields.append(np.where(np.isfinite(values), values, NODATA))
        attrs = dict(array.attrs)
        meta.append(
            dict(name=name, units=attrs.get("units"), long_name=attrs.get("long_name"))
        )
        logger.info("%s: %s, %.1f%% with values", name, meta[-1]["long_name"],
                    100 * np.isfinite(values).mean())  # fmt: skip
    tmp = out_dir / "static.tmp.npy"
    np.save(tmp, np.stack(fields))
    os.replace(tmp, out_dir / "static.npy")
    (out_dir / "static.json").write_text(json.dumps(
        dict(source=DAILY_URL, grid="hourly store rows 0..1471 (90N..57.1S), lon 0..359.9",
             nodata=NODATA, fields=meta), indent=2))  # fmt: skip
    logger.info("wrote %s", out_dir / "static.npy")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    args = ap.parse_args()
    run(Path(args.root))


if __name__ == "__main__":
    main()
