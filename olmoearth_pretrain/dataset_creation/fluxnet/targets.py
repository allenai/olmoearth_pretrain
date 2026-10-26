"""Stage 2: parse every downloaded archive into a daily flux table and a site table.

Reads two members of each ONEFlux archive without extracting it:

* ``*_BIF_*.csv`` (BADM site info, long format): coordinates, elevation, IGBP,
  network, UTC offset. When a site moved, the latest ``LOCATION`` group wins.
* ``*_FLUXMET_DD_*.csv`` (daily, local standard time; ``*_FULLSET_DD_*`` in
  the older FLUXNET2015-style archives): the columns in ``DAILY_COLUMNS``,
  -9999 turned into NaN.

Outputs under ``<root>/meta``: ``daily/<site_id>.parquet`` and ``sites.parquet``
(one row per site with record length and valid-day counts). A site listed by two
hubs keeps the archive with more valid GPP days.

Usage::

    python -m olmoearth_pretrain.dataset_creation.fluxnet.targets --workers 32
"""

from __future__ import annotations

import argparse
import io
import logging
import multiprocessing
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import DEFAULT_ROOT, NODATA
from .download import DAILY_MARKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Daily ONEFlux variables kept per site. GPP/RECO/NEE in gC m-2 d-1, LE/H in
# W m-2; *_QC at daily resolution is the fraction (0-1) of measured or
# good-quality gap-filled half-hours behind the daily value.
DAILY_COLUMNS = [
    "GPP_NT_VUT_REF",
    "GPP_DT_VUT_REF",
    "RECO_NT_VUT_REF",
    "NEE_VUT_REF",
    "NEE_VUT_REF_QC",
    "LE_F_MDS",
    "LE_F_MDS_QC",
    "H_F_MDS",
    "H_F_MDS_QC",
    "SW_IN_F",
    "TA_F",
    "VPD_F",
    "P_F",
]
# A day counts as valid for GPP when at least half of its NEE half-hours are
# measured or good-quality gap-filled.
MIN_DAILY_QC = 0.5


def _member(names: list[str], marker: str) -> str | None:
    """First archive member whose name contains ``marker``.

    Args:
        names: Archive member names.
        marker: Substring to look for, e.g. ``_FLUXMET_DD_``.

    Returns:
        The member name, or None.
    """
    matches = sorted(n for n in names if marker in n and n.endswith(".csv"))
    return matches[0] if matches else None


def parse_bif(text: str) -> dict[str, Any]:
    """Site metadata from a BIF (BADM long-format) csv.

    Args:
        text: The BIF csv content.

    Returns:
        ``lat, lon, elev, igbp, network, source_network, utc_offset`` (NaN or
        empty string when absent).
    """
    bif = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    info: dict[str, Any] = dict(
        lat=np.nan, lon=np.nan, elev=np.nan, igbp="", network="",
        source_network="", utc_offset=np.nan,
    )  # fmt: skip
    location = bif[bif["VARIABLE_GROUP"] == "LOCATION"]
    if not location.empty:
        # Latest location group (a moved tower lists several).
        starts = location[location["VARIABLE"] == "LOCATION_DATE_START"]
        group = (
            starts.sort_values("DATAVALUE")["GROUP_ID"].iloc[-1]
            if not starts.empty
            else location["GROUP_ID"].iloc[-1]
        )
        values = dict(
            location.loc[
                location["GROUP_ID"] == group, ["VARIABLE", "DATAVALUE"]
            ].values
        )
        info["lat"] = pd.to_numeric(values.get("LOCATION_LAT"), errors="coerce")
        info["lon"] = pd.to_numeric(values.get("LOCATION_LONG"), errors="coerce")
        info["elev"] = pd.to_numeric(values.get("LOCATION_ELEV"), errors="coerce")
    simple = {
        "IGBP": "igbp",
        "NETWORK": "network",
        "PRODUCT_SOURCE_NETWORK": "source_network",
        "UTC_OFFSET": "utc_offset",
    }
    for variable, key in simple.items():
        rows = bif.loc[bif["VARIABLE"] == variable, "DATAVALUE"]
        if rows.empty:
            continue
        if key == "network":
            info[key] = ";".join(sorted(set(rows)))
        elif key == "utc_offset":
            info[key] = pd.to_numeric(rows.iloc[-1], errors="coerce")
        else:
            info[key] = rows.iloc[-1]
    return info


def parse_daily(text: str) -> pd.DataFrame:
    """The kept daily variables from a FLUXMET_DD csv.

    Args:
        text: The DD csv content.

    Returns:
        Frame indexed by ``date`` with ``DAILY_COLUMNS`` (missing columns as NaN).
    """
    header = text.split("\n", 1)[0].split(",")
    usecols = ["TIMESTAMP"] + [c for c in DAILY_COLUMNS if c in header]
    daily = pd.read_csv(io.StringIO(text), usecols=usecols, dtype={"TIMESTAMP": str})
    daily = daily.replace(NODATA, np.nan)
    daily["date"] = pd.to_datetime(daily.pop("TIMESTAMP"), format="%Y%m%d")
    for column in DAILY_COLUMNS:
        if column not in daily:
            daily[column] = np.nan
    return daily.set_index("date")[DAILY_COLUMNS].astype("float32")


def valid_gpp_days(daily: pd.DataFrame) -> pd.Series:
    """Days whose GPP is usable as a target.

    Args:
        daily: Frame from :func:`parse_daily`.

    Returns:
        Boolean series: NT GPP present and NEE daily QC >= ``MIN_DAILY_QC``.
    """
    return daily["GPP_NT_VUT_REF"].notna() & (daily["NEE_VUT_REF_QC"] >= MIN_DAILY_QC)


def parse_archive(job: tuple[str, str, str]) -> dict[str, Any] | None:
    """Worker: parse one archive and write its daily parquet.

    Args:
        job: ``(archive path, hub, output dir)``.

    Returns:
        The site row for ``sites.parquet``, or None if the archive has no DD file.
    """
    path, hub, out_dir = job
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        dd_name = next(
            (m for m in (_member(names, k) for k in DAILY_MARKERS) if m), None
        )
        bif_name = _member(names, "_BIF_")
        if dd_name is None:
            logger.warning("%s: no daily member", path)
            return None
        daily = parse_daily(archive.read(dd_name).decode("utf-8", "replace"))
        info = (
            parse_bif(archive.read(bif_name).decode("utf-8", "replace"))
            if bif_name
            else {}
        )
    site_id = Path(path).name.split("_")[1]
    valid = valid_gpp_days(daily)
    le_valid = daily["LE_F_MDS"].notna() & (daily["LE_F_MDS_QC"] >= MIN_DAILY_QC)
    out = Path(out_dir) / f"{site_id}__{hub}.parquet"
    daily.to_parquet(out)
    return dict(
        site_id=site_id,
        hub=hub,
        archive=Path(path).name,
        daily_file=out.name,
        first_date=daily.index.min(),
        last_date=daily.index.max(),
        n_days=len(daily),
        n_gpp_valid=int(valid.sum()),
        n_le_valid=int(le_valid.sum()),
        n_years_gpp=int((valid.groupby(daily.index.year).sum() >= 180).sum()),
        gpp_mean=float(daily.loc[valid, "GPP_NT_VUT_REF"].mean()),
        **info,
    )


def build_targets(root: Path, workers: int) -> pd.DataFrame:
    """Parse every downloaded archive and write ``meta/sites.parquet``.

    Args:
        root: Dataset root.
        workers: Parser processes.

    Returns:
        The site table (one row per site after hub deduplication).
    """
    status = pd.read_csv(root / "raw" / "download_status.csv", keep_default_na=False)
    status = status[status["status"].isin(["exists", "downloaded"])]
    out_dir = root / "meta" / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(root / "raw" / row.hub / row.filename), row.hub, str(out_dir))
        for row in status.itertuples(index=False)
    ]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        rows = [row for row in pool.imap_unordered(parse_archive, jobs) if row]
    sites = pd.DataFrame(rows)
    # Catalogue coordinates and IGBP fill sites whose archive has no BIF.
    catalog = status.set_index("filename")
    for column in ("lat", "lon"):
        if column not in sites:
            sites[column] = np.nan
        missing = sites[column].isna()
        values = pd.to_numeric(catalog[column], errors="coerce")
        sites.loc[missing, column] = sites.loc[missing, "archive"].map(values)
    if "igbp" not in sites:
        sites["igbp"] = ""
    if "igbp" in catalog:
        missing = sites["igbp"].fillna("") == ""
        sites.loc[missing, "igbp"] = sites.loc[missing, "archive"].map(catalog["igbp"])
    sites = (
        sites.sort_values("n_gpp_valid", ascending=False)
        .drop_duplicates("site_id")
        .sort_values("site_id")
        .reset_index(drop=True)
    )
    sites.to_parquet(root / "meta" / "sites.parquet")
    logger.info(
        "%d sites (%d with >=1 year of valid GPP), %d valid GPP days; by hub:\n%s",
        len(sites),
        int((sites["n_years_gpp"] >= 1).sum()),
        int(sites["n_gpp_valid"].sum()),
        sites.groupby("hub")[["n_gpp_valid"]].agg(["count", "sum"]),
    )
    return sites


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    build_targets(Path(args.root), args.workers)


if __name__ == "__main__":
    main()
