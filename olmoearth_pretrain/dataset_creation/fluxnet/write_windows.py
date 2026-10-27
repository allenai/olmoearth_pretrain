"""Stage 4: anchor dates, targets, spatial splits and the rslearn windows.

``candidates``: for every site, anchor dates ``d`` on a global 8-day grid (so
the 8-day target periods never overlap) whose ERA5 window ``[d - 448, d)`` is
inside the mirror and has no all-missing day, with targets over the window's
last 8 days ``[d - 8, d)``:

* ``gpp``: mean GPP (gC m-2 d-1; per day the mean of the night- and day-time
  partitionings ``GPP_NT_VUT_REF`` / ``GPP_DT_VUT_REF``, or the one that
  exists) over days with NEE daily QC >= 0.5; at least 6 such days.
* ``le``: mean latent heat (``LE_F_MDS``, W m-2), LE daily QC >= 0.5, >= 6 days.
* ``nee``: mean ``NEE_VUT_REF`` on the GPP days.
* ``rgpp``: ``gpp`` divided by the site's 95th percentile of ``gpp`` (relative
  GPP; undefined when that percentile is below 0.5 gC m-2 d-1; capped at 1.5). The headline
  target: dividing out the site's productivity level, which the weather does not
  determine, leaves the phenological / stress state.
* ``gpp_anom``: ``gpp`` minus the site's leave-one-year-out climatology at the
  same day of year (8-day GPP within +-7 days in every other year), when at
  least two other years contribute.

Splits are spatial: sites within ``GROUP_KM`` of each other (single linkage)
form one group, and groups go to train / val / test (70 / 15 / 15 % of sites)
within each coarse region, so towers sharing a landscape (or an ERA5 cell)
never straddle splits.

``windows``: two groups of pre-materialized windows (``era5_daily`` raster
``(14, 448, 1, 1)`` + ``label`` point with the targets as properties):

* ``all``: every candidate with ``gpp`` on a ``--spacing``-day grid, for
  supervised training;
* ``eval``: a fixed subset recorded in ``meta/eval_keys.parquet`` (reused on
  every rebuild; delete it to draw a new one), originally a site-balanced draw
  (``--n-train/--n-val/--n-test``) of the
  candidates where ``gpp``, ``le`` and ``nee`` are all defined, for
  the in-loop linear probes (the task configs use ``allow_invalid: false``, so a
  missing property fails loudly instead of reading as 0). It is a separate
  group so the evaluator does not scan the full set.

Tags: ``split``, ``site_id``, ``igbp``, ``region``, ``hub``, ``site_group``,
``anchor_date``, and ``has_anom`` / ``has_le`` / ``has_rgpp`` (present when
``gpp_anom`` / ``le`` / ``rgpp`` is defined).

Usage::

    python -m olmoearth_pretrain.dataset_creation.fluxnet.write_windows all --workers 32
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rasterio.crs import CRS
from rslearn.dataset import Dataset, Window
from rslearn.utils.feature import Feature
from rslearn.utils.geometry import WGS84_PROJECTION, Projection, STGeometry
from rslearn.utils.raster_array import RasterArray, RasterMetadata
from rslearn.utils.raster_format import NumpyRasterFormat
from rslearn.utils.vector_format import GeojsonCoordinateMode, GeojsonVectorFormat
from shapely.geometry import Point
from upath import UPath

from .common import (
    DEFAULT_ROOT,
    ERA5_LAYER,
    ERA5L_BANDS,
    LABEL_LAYER,
    NODATA,
    WINDOW_DAYS,
    haversine_km,
)
from .targets import MIN_DAILY_QC, daily_gpp, valid_gpp_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_DAYS = 8
MIN_TARGET_DAYS = 6
ANCHOR_GRID_DAYS = 8
CLIM_HALF_WIDTH_DAYS = 7
MIN_CLIM_YEARS = 2
GROUP_KM = 50.0
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
TARGETS = ("gpp", "rgpp", "le", "nee", "gpp_anom")
# Targets every eval window carries; gpp_anom is optional (has_anom tag).
EVAL_REQUIRED = ("gpp", "le", "nee")
RGPP_QUANTILE = 0.95
MIN_RGPP_SCALE = 0.5  # gC m-2 d-1; below that a site has no growing season to scale by
# Dryland towers with a 95th percentile barely above MIN_RGPP_SCALE reach rgpp
# 3-14 on rare GPP spikes (225 of 212k candidates exceed 1.5), which would
# dominate an MSE probe: cap there.
RGPP_MAX = 1.5
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# --- splits -------------------------------------------------------------------
def region_of(lat: float, lon: float) -> str:
    """Coarse continental region used to stratify the split.

    Args:
        lat: Latitude (deg).
        lon: Longitude (deg, -180..180).

    Returns:
        One of north_america, south_america, europe, africa, asia, oceania.
    """
    if lon < -30:
        return "north_america" if lat >= 13 else "south_america"
    if lon < 60:
        return "europe" if lat >= 35 else "africa"
    if lat < -10 and lon >= 110:
        return "oceania"
    return "asia"


def spatial_groups(sites: pd.DataFrame) -> np.ndarray:
    """Single-linkage groups of sites closer than ``GROUP_KM``.

    Args:
        sites: Frame with ``lat`` and ``lon``.

    Returns:
        Group id per site (index of the group's first site).
    """
    lat, lon = sites["lat"].to_numpy(), sites["lon"].to_numpy()
    parent = np.arange(len(sites))

    def find(i: int) -> int:
        """Union-find root with path halving.

        Args:
            i: Site index.

        Returns:
            Root index of the site's group.
        """
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            if haversine_km(lat[i], lon[i], lat[j], lon[j]) < GROUP_KM:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(len(sites))])


def assign_splits(sites: pd.DataFrame, seed: int) -> pd.Series:
    """Train / val / test per site, by whole spatial groups within each region.

    Args:
        sites: Frame with ``region`` and ``site_group``.
        seed: Shuffle seed.

    Returns:
        Split name per site (same index as ``sites``).
    """
    rng = np.random.default_rng(seed)
    split = pd.Series("", index=sites.index)
    for _, region in sites.groupby("region"):
        groups = region.groupby("site_group").size()
        order = rng.permutation(groups.index.to_numpy())
        cumulative = groups.loc[order].cumsum() / groups.sum()
        for group, frac in cumulative.items():
            if frac <= SPLIT_FRACTIONS["train"] + 1e-9 or len(groups) < 3:
                name = "train"
            elif frac <= SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"] + 1e-9:
                name = "val"
            else:
                name = "test"
            split[region.index[region["site_group"] == group]] = name
    return split


# --- targets ------------------------------------------------------------------
def _rolling_target(values: pd.Series, valid: pd.Series) -> pd.Series:
    """Trailing 8-day mean over valid days, defined with >= 6 of them.

    Args:
        values: Daily values on a complete daily index.
        valid: Daily validity on the same index.

    Returns:
        Series whose value at day ``e`` is the mean over ``[e - 7, e]``.
    """
    return values.where(valid).rolling(TARGET_DAYS, min_periods=MIN_TARGET_DAYS).mean()


def loo_climatology(target: pd.Series) -> pd.Series:
    """Leave-one-year-out day-of-year climatology of an 8-day target series.

    Args:
        target: 8-day target on a complete daily index.

    Returns:
        Per day, the mean of the target within +-``CLIM_HALF_WIDTH_DAYS`` days of
        year in every *other* year, NaN with fewer than ``MIN_CLIM_YEARS`` years.
    """
    frame = pd.DataFrame(
        {
            "v": target,
            "year": target.index.year,
            "doy": np.minimum(target.index.dayofyear, 365) - 1,
        }
    )
    years = np.sort(frame["year"].unique())
    year_index = {y: k for k, y in enumerate(years)}
    values = np.full((len(years), 365), np.nan)
    rows = frame["year"].map(year_index).to_numpy()
    values[rows, frame["doy"].to_numpy()] = frame["v"].to_numpy()
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    # Circular +-W day-of-year window sums per year.
    sums = np.zeros_like(filled)
    counts = np.zeros_like(filled)
    for shift in range(-CLIM_HALF_WIDTH_DAYS, CLIM_HALF_WIDTH_DAYS + 1):
        sums += np.roll(filled, shift, axis=1)
        counts += np.roll(finite, shift, axis=1)
    other_sums = sums.sum(axis=0, keepdims=True) - sums
    other_counts = counts.sum(axis=0, keepdims=True) - counts
    other_years = (counts > 0).sum(axis=0, keepdims=True) - (counts > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        clim = np.where(
            other_years >= MIN_CLIM_YEARS, other_sums / other_counts, np.nan
        )
    return pd.Series(clim[rows, frame["doy"].to_numpy()], index=target.index)


def site_targets(daily: pd.DataFrame) -> pd.DataFrame:
    """All 8-day targets of one site, indexed by the target period's last day.

    Args:
        daily: Site's daily table (``targets.DAILY_COLUMNS``).

    Returns:
        Frame with ``TARGETS`` columns on a complete daily index.
    """
    daily = daily.asfreq("D")
    gpp_valid = valid_gpp_days(daily)
    le_valid = daily["LE_F_MDS"].notna() & (daily["LE_F_MDS_QC"] >= MIN_DAILY_QC)
    out = pd.DataFrame(index=daily.index)
    out["gpp"] = _rolling_target(daily_gpp(daily), gpp_valid)
    # Relative GPP: the site's 8-day GPP over its 95th percentile, so the
    # vegetation's productivity level (not inferable from weather) drops out
    # and the target is the phenological / stress state.
    scale = out["gpp"].quantile(RGPP_QUANTILE)
    out["rgpp"] = (
        (out["gpp"] / scale).clip(upper=RGPP_MAX) if scale > MIN_RGPP_SCALE else np.nan
    )
    out["le"] = _rolling_target(daily["LE_F_MDS"], le_valid)
    out["nee"] = _rolling_target(daily["NEE_VUT_REF"], gpp_valid)
    out["gpp_anom"] = out["gpp"] - loo_climatology(out["gpp"])
    return out


def site_candidates(job: tuple[dict[str, Any], str]) -> pd.DataFrame:
    """Worker: candidate anchors of one site with their targets.

    Args:
        job: ``(site row incl. ERA5 cell and split, meta dir)``.

    Returns:
        One row per anchor: ``site_id, anchor (days since epoch), t0`` (first
        ERA5 row of the window) and the targets.
    """
    site, meta_dir = job
    meta = Path(meta_dir)
    daily = pd.read_parquet(meta / "daily" / site["daily_file"])
    targets = site_targets(daily)
    dates = np.load(meta / "era5_dates.npy")
    series = np.load(meta / "era5" / f"{site['site_id']}.npy")
    missing_day = (series == NODATA).all(axis=1)
    bad_prefix = np.concatenate([[0], np.cumsum(missing_day)])

    last_day = (targets.index - pd.Timestamp("1970-01-01")).days.to_numpy()
    anchor = last_day + 1
    keep = (anchor % ANCHOR_GRID_DAYS == 0) & targets["gpp"].notna().to_numpy()
    t0 = np.searchsorted(dates, anchor - WINDOW_DAYS)
    inside = (t0 + WINDOW_DAYS <= len(dates)) & (anchor - WINDOW_DAYS >= dates[0])
    keep &= inside
    t0_safe = np.where(inside, t0, 0)
    keep &= (bad_prefix[t0_safe + WINDOW_DAYS] - bad_prefix[t0_safe]) == 0
    rows = targets[keep].reset_index(drop=True)
    rows.insert(0, "t0", t0[keep])
    rows.insert(0, "anchor", anchor[keep])
    rows.insert(0, "site_id", site["site_id"])
    return rows


def build_candidates(
    root: Path, workers: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Site splits and every candidate anchor; writes both tables under ``meta``.

    Args:
        root: Dataset root.
        workers: Processes.
        seed: Split seed.

    Returns:
        ``(sites, candidates)``.
    """
    meta = root / "meta"
    sites = pd.read_parquet(meta / "sites.parquet").merge(
        pd.read_parquet(meta / "era5_cells.parquet"), on="site_id"
    )
    sites = sites[(sites["ilat"] >= 0) & (sites["n_gpp_valid"] > 0)].reset_index(
        drop=True
    )
    sites["region"] = [region_of(r.lat, r.lon) for r in sites.itertuples()]
    sites["site_group"] = spatial_groups(sites)
    sites["split"] = assign_splits(sites, seed)

    jobs = [(row, str(meta)) for row in sites.to_dict("records")]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        parts = list(pool.imap_unordered(site_candidates, jobs))
    # Sorted, so the seeded eval selection does not depend on worker order.
    candidates = pd.concat(parts).sort_values(["site_id", "anchor"], ignore_index=True)
    candidates = candidates.merge(
        sites[["site_id", "split", "igbp", "region", "hub", "site_group"]], on="site_id"
    )
    has_data = sites["site_id"].isin(candidates["site_id"])
    sites = sites[has_data].reset_index(drop=True)
    sites.to_parquet(meta / "sites_split.parquet")
    candidates.to_parquet(meta / "candidates.parquet")
    summary = candidates.groupby("split").agg(
        sites=("site_id", "nunique"), windows=("anchor", "size"),
        le=("le", lambda s: int(s.notna().sum())), anom=("gpp_anom", lambda s: int(s.notna().sum())),
        gpp_mean=("gpp", "mean"), gpp_std=("gpp", "std"),
    )  # fmt: skip
    logger.info("candidates (8-day grid):\n%s", summary)
    return sites, candidates


# --- subsets ------------------------------------------------------------------
def balanced_subset(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Up to ``n`` rows spread as evenly as possible over sites.

    Takes rows round-robin over sites (each site's rows in random order), so
    every site contributes before any site contributes twice.

    Args:
        frame: Candidate rows of one split.
        n: Rows wanted.
        seed: Shuffle seed.

    Returns:
        The selected rows.
    """
    shuffled = frame.sample(frac=1.0, random_state=seed)
    shuffled = shuffled.assign(_rank=shuffled.groupby("site_id").cumcount())
    site_order = {s: k for k, s in enumerate(shuffled["site_id"].drop_duplicates())}
    shuffled = shuffled.assign(_site=shuffled["site_id"].map(site_order))
    picked = shuffled.sort_values(["_rank", "_site"]).head(n)
    return picked.drop(columns=["_rank", "_site"])


def select_windows(
    candidates: pd.DataFrame,
    spacing: int,
    quotas: dict[str, int],
    seed: int,
    eval_keys: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Rows of the ``all`` and ``eval`` groups.

    Args:
        candidates: Output of :func:`build_candidates`.
        spacing: Anchor spacing (days, multiple of 8) of the ``all`` group.
        quotas: Eval subset size per split.
        seed: Sampling seed.
        eval_keys: ``site_id, anchor`` of a fixed eval subset (``meta/eval_keys.parquet``);
            when given, it replaces the seeded selection.

    Returns:
        ``{"all": rows, "eval": rows}``.
    """
    full = candidates[candidates["anchor"] % spacing == 0]
    if eval_keys is not None:
        eval_rows = candidates.merge(
            eval_keys[["site_id", "anchor"]], on=["site_id", "anchor"]
        )
        if len(eval_rows) != len(eval_keys):
            raise ValueError(
                f"{len(eval_keys) - len(eval_rows)} eval keys are not candidates"
            )
        return {"all": full, "eval": eval_rows}
    eligible = candidates[candidates[list(EVAL_REQUIRED)].notna().all(axis=1)]
    eval_rows = pd.concat(
        [
            balanced_subset(eligible[eligible["split"] == split], n, seed)
            for split, n in quotas.items()
        ],
        ignore_index=True,
    )
    return {"all": full, "eval": eval_rows}


# --- windows ------------------------------------------------------------------
class WindowWriter:
    """Writes pre-materialized rslearn windows (ERA5 raster + target point)."""

    def __init__(self, ds_path: str) -> None:
        """Open the dataset and set up formats and projection.

        Args:
            ds_path: rslearn dataset root.
        """
        self.dataset = Dataset(UPath(ds_path))
        self.projection = Projection(CRS.from_epsg(4326), 0.1, -0.1)
        self.raster_format = NumpyRasterFormat()
        self.vector_format = GeojsonVectorFormat(
            coordinate_mode=GeojsonCoordinateMode.WGS84
        )

    def write(
        self, group: str, row: dict[str, Any], site: dict[str, Any], block: np.ndarray
    ) -> None:
        """Save one window with its ERA5 block and target feature.

        Args:
            group: Window group (``all`` or ``eval``).
            row: Candidate row.
            site: Site row (coordinates, ERA5 cell).
            block: ``(448, 14)`` ERA5 rows of the window.
        """
        start = EPOCH + timedelta(days=int(row["anchor"]) - WINDOW_DAYS)
        end = EPOCH + timedelta(days=int(row["anchor"]))
        # The 0.1 deg pixel holding the tower: rslearn only reads label features
        # inside the window bounds (the raster is pre-materialized, so the
        # bounds are otherwise just metadata).
        col = int(np.floor(site["lon"] / 0.1))
        pix_row = int(np.floor(site["lat"] / -0.1))
        targets = {t: float(row[t]) for t in TARGETS if np.isfinite(row[t])}
        options: dict[str, Any] = dict(
            split=row["split"], site_id=row["site_id"], igbp=str(site["igbp"]),
            region=row["region"], hub=row["hub"], site_group=str(row["site_group"]),
            anchor_date=end.strftime("%Y-%m-%d"), **targets,
        )  # fmt: skip
        if "gpp_anom" in targets:
            options["has_anom"] = "1"
        if "le" in targets:
            options["has_le"] = "1"
        if "rgpp" in targets:
            options["has_rgpp"] = "1"
        window = Window(
            storage=self.dataset.storage, group=group,
            name=f"{row['site_id']}_{end:%Y%m%d}", projection=self.projection,
            bounds=(col, pix_row, col + 1, pix_row + 1), time_range=(start, end),
            options=options,
        )  # fmt: skip
        window.save()
        raster = RasterArray(
            array=np.ascontiguousarray(block.T)[:, :, None, None].astype(np.float32),
            timestamps=[
                (start + timedelta(days=k), start + timedelta(days=k + 1))
                for k in range(WINDOW_DAYS)
            ],
            metadata=RasterMetadata(nodata_value=NODATA),
        )
        self.raster_format.encode_raster(
            window.get_raster_dir(ERA5_LAYER, list(ERA5L_BANDS)),
            self.projection,
            window.bounds,
            raster,
        )
        window.mark_layer_completed(ERA5_LAYER)
        feature = Feature(
            STGeometry(WGS84_PROJECTION, Point(site["lon"], site["lat"]), (start, end)),
            targets,
        )
        self.vector_format.encode_vector(window.get_layer_dir(LABEL_LAYER), [feature])
        window.mark_layer_completed(LABEL_LAYER)


def dataset_config() -> dict[str, Any]:
    """Rslearn ``config.json`` (pre-materialized layers, no data sources).

    Returns:
        The config dict.
    """
    band_set = {
        "bands": list(ERA5L_BANDS),
        "dtype": "float32",
        "nodata_value": NODATA,
        "format": {"class_path": "rslearn.utils.raster_format.NumpyRasterFormat"},
    }
    return {
        "layers": {
            ERA5_LAYER: {"type": "raster", "band_sets": [band_set]},
            LABEL_LAYER: {"type": "vector"},
        }
    }


def write_site(
    job: tuple[dict[str, Any], dict[str, list[dict[str, Any]]], str, str],
) -> dict[str, int]:
    """Worker: write every selected window of one site.

    Args:
        job: ``(site row, {group: candidate rows}, meta dir, dataset path)``.

    Returns:
        Windows written per group.
    """
    site, rows_by_group, meta_dir, ds_path = job
    series = np.load(Path(meta_dir) / "era5" / f"{site['site_id']}.npy")
    writer = WindowWriter(ds_path)
    counts: dict[str, int] = {}
    for group, rows in rows_by_group.items():
        for row in rows:
            t0 = int(row["t0"])
            writer.write(group, row, site, series[t0 : t0 + WINDOW_DAYS])
        counts[group] = len(rows)
    return counts


def write_all(
    root: Path, selected: dict[str, pd.DataFrame], sites: pd.DataFrame, workers: int
) -> None:
    """Write the selected windows of both groups and their summaries.

    Args:
        root: Dataset root.
        selected: Rows per group from :func:`select_windows`.
        sites: Site table with ERA5 cells.
        workers: Processes.
    """
    ds_path = root / "rslearn_dataset"
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "config.json").write_text(json.dumps(dataset_config(), indent=2))
    site_rows = sites.set_index("site_id").to_dict("index")
    jobs = []
    for site_id in sorted(set().union(*(set(f["site_id"]) for f in selected.values()))):
        site = dict(site_rows[site_id], site_id=site_id)
        rows_by_group = {
            group: frame[frame["site_id"] == site_id].to_dict("records")
            for group, frame in selected.items()
        }
        jobs.append((site, rows_by_group, str(root / "meta"), str(ds_path)))
    totals: dict[str, int] = {}
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for k, counts in enumerate(pool.imap_unordered(write_site, jobs), start=1):
            for group, n in counts.items():
                totals[group] = totals.get(group, 0) + n
            if k % 50 == 0:
                logger.info("[%d/%d sites] %s", k, len(jobs), totals)
    stats: dict[str, Any] = {}
    for group, frame in selected.items():
        frame.to_parquet(root / "meta" / f"windows_{group}.parquet")
        stats[group] = {
            split: {
                "windows": len(part), "sites": int(part["site_id"].nunique()),
                **{t: dict(n=int(part[t].notna().sum()), mean=float(part[t].mean()),
                           std=float(part[t].std())) for t in TARGETS},
            }
            for split, part in frame.groupby("split")
        }  # fmt: skip
    (root / "meta" / "label_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        "windows written: %s\nlabel stats:\n%s", totals, json.dumps(stats, indent=2)
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["candidates", "windows", "all"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--spacing", type=int, default=16, help="anchor spacing of the 'all' group"
    )
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument(
        "--eval-keys",
        default=None,
        help="parquet of site_id, anchor fixing the eval subset (default: <root>/meta/eval_keys.parquet if it exists)",
    )
    args = ap.parse_args()
    root = Path(args.root)
    if args.step in ("candidates", "all"):
        sites, candidates = build_candidates(root, args.workers, args.seed)
    else:
        sites = pd.read_parquet(root / "meta" / "sites_split.parquet")
        candidates = pd.read_parquet(root / "meta" / "candidates.parquet")
    if args.step in ("windows", "all"):
        quotas = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
        keys_path = (
            Path(args.eval_keys)
            if args.eval_keys
            else root / "meta" / "eval_keys.parquet"
        )
        eval_keys = pd.read_parquet(keys_path) if keys_path.exists() else None
        selected = select_windows(
            candidates, args.spacing, quotas, args.seed, eval_keys
        )
        selected["eval"][["site_id", "anchor", "split"]].to_parquet(
            root / "meta" / "eval_keys.parquet"
        )
        write_all(root, selected, sites, args.workers)


if __name__ == "__main__":
    main()
