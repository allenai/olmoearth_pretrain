"""Stage 1: download the weekly USDM shapefiles and the CONUS state boundaries.

Every Tuesday from 2000-01-04 to today is tried once (the archive is complete
as far as we know; a missing week is recorded, not fatal). Files already on
disk that open as zip archives are kept, so the stage resumes.

Outputs under ``<root>/raw``: ``maps/USDM_<YYYYMMDD>_M.zip``,
``cb_2018_us_state_20m.zip`` and ``download_status.csv``.

Usage::

    python -m olmoearth_pretrain.dataset_creation.usdm.download --workers 8
"""

from __future__ import annotations

import argparse
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from .common import DEFAULT_ROOT, FIRST_MAP, STATES_URL, USDM_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HTTP_TIMEOUT = (30, 120)
MAX_ATTEMPTS = 4


def map_dates(last: date) -> list[date]:
    """Every USDM map date (Tuesdays) from the first map to ``last``.

    Args:
        last: Last date to include.

    Returns:
        Dates in order.
    """
    out, d = [], FIRST_MAP
    while d <= last:
        out.append(d)
        d += timedelta(days=7)
    return out


def is_zip(path: Path) -> bool:
    """Whether ``path`` exists and is a readable zip archive.

    Args:
        path: File path.

    Returns:
        True for a valid archive.
    """
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def fetch(url: str, dest: Path) -> str:
    """Download ``url`` to ``dest`` (atomically), with retries.

    Args:
        url: Source URL.
        dest: Target path.

    Returns:
        ``ok``, ``cached`` or ``missing`` (HTTP 404), or the last error.
    """
    if is_zip(dest):
        return "cached"
    error = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(url, timeout=HTTP_TIMEOUT)
            if response.status_code == 404:
                return "missing"
            response.raise_for_status()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(response.content)
            tmp.rename(dest)
            if is_zip(dest):
                return "ok"
            error = "bad zip"
        except requests.RequestException as exc:
            error = str(exc)
        time.sleep(2**attempt)
    return error


def download(root: Path, workers: int, last: date) -> None:
    """Fetch every weekly map up to ``last`` plus the state boundaries.

    Args:
        root: Dataset root.
        workers: Concurrent downloads.
        last: Last map date to try.
    """
    raw = root / "raw"
    (raw / "maps").mkdir(parents=True, exist_ok=True)
    logger.info("states: %s", fetch(STATES_URL, raw / "cb_2018_us_state_20m.zip"))
    dates = map_dates(last)

    def one(d: date) -> dict[str, str]:
        """Fetch one weekly map.

        Args:
            d: Map date.

        Returns:
            Status row.
        """
        ymd = d.strftime("%Y%m%d")
        status = fetch(USDM_URL.format(ymd=ymd), raw / "maps" / f"USDM_{ymd}_M.zip")
        return {"date": d.isoformat(), "status": status}

    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(one, dates))
    status = pd.DataFrame(rows)
    status.to_csv(raw / "download_status.csv", index=False)
    logger.info(
        "%d map dates %s..%s: %s",
        len(dates), dates[0], dates[-1], status["status"].value_counts().to_dict(),
    )  # fmt: skip


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--last", default=None, help="last map date (YYYY-MM-DD; default today)"
    )
    args = ap.parse_args()
    last = date.fromisoformat(args.last) if args.last else date.today()
    download(Path(args.root), args.workers, last)


if __name__ == "__main__":
    main()
