"""Tag the weather-triggered landslide windows into a leakage-safe train/val split.

This is the train/val tagging step that follows ``dedup_windows.py``. It:

  1. Loads the deduped, leakage-safe universe (``deduped_windows.parquet``).
  2. Keeps only **weather-triggered** events (rain / cyclone / monsoon / ...),
     dropping earthquake- and unknown-trigger windows whose pos/neg contrast
     carries no interannual weather signal (see the project discussion).
  3. Splits into train/val at the **ERA5-cell** level — the only leakage-safe
     unit. We do NOT split on individual windows or pos/neg pairs, because:
       - a positive and its negative twin share a lat/lon (and thus an ERA5
         cell), so they must stay in the same split, and
       - many distinct landslide locations fall inside the *same* ERA5 cell
         (~half of the cells here), so they share the same coarse ERA5 input
         and would leak across splits if separated.
     Assigning whole ERA5 cells to a split guarantees no two windows with the
     same ERA5 forcing straddle the train/val boundary.
  4. Stratifies the assignment to mirror the overall distribution across
     event_type, event_year, and coarse geography as closely as possible,
     while hitting a target val fraction.
  5. Writes the split by **copying each selected window's metadata.json** into a
     standalone rslearn-style dataset tree at ``--data-root``
     (``<data-root>/windows/<group>/<name>/metadata.json``), with
     ``options.split`` set to "train"/"val" (the original value is preserved
     under ``options.orig_split``). Unless ``--no-labels`` is passed, it also
     writes a per-window vector ``label`` layer
     (``layers/label/data.geojson`` + ``completed``) holding the pos/neg target
     as a single ``category`` feature, and registers the ``label`` vector layer
     in ``<data-root>/config.json`` — making the dataset directly consumable as a
     binary classification eval task via rslearn's ``ClassificationTask``. The
     ERA5 raster layer is materialized separately (not copied here).
  6. Emits ``eval_split.parquet`` (authoritative table), ``split_summary.md``,
     and ``subsampling.png`` (same layout as ``visualize_dedup`` but colored by
     train/val instead of source group) into ``--out-dir`` (the repo splits dir).

Usage:
    python eval_tagging.py [--ds-root PATH] [--out-dir PATH] [--data-root PATH] \
        [--val-frac 0.294] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from rslearn.utils import Feature, Projection, STGeometry
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DS_ROOT = (
    "/weka/dfive-default/piperw/rslearn_projects/data/landslide/"
    "sen12landslides/all_positives"
)
DEFAULT_PARQUET = SCRIPT_DIR / "deduped_windows.parquet"
DEFAULT_OUT_DIR = SCRIPT_DIR

# Where the tagged window metadata tree (the actual eval dataset) is written.
DEFAULT_DATA_ROOT = "/weka/dfive-default/olmoearth/eval_datasets/landslide_era5"

# Vector label layer written per-window for the binary pos/neg classification
# task. ``classes`` order is significant: "positive" maps to class id 1 so that
# F1-on-the-positive-class reporting matches the other ERA5 eval tasks.
LABEL_LAYER_NAME = "label"
LABEL_PROPERTY_NAME = "category"
LABEL_CLASSES = ["negative", "positive"]

# Weather-triggered event types across BOTH source vocabularies (sen12 uses
# title-case, glc uses lowercase). Earthquake/unknown/nan are intentionally
# excluded: their pos/neg contrast carries no interannual weather signal.
WEATHER_TYPES = {
    # sen12_landslides vocabulary
    "Rainfall",
    "Hurricane",
    "Cyclone",
    "Tropical Storm",
    # glc vocabulary
    "rain",
    "continuous_rain",
    "downpour",
    "tropical_cyclone",
    "monsoon",
    "flooding",
    "freeze_thaw",
    "snowfall_snowmelt",
}

# Target share of windows held out for validation (≈ 125 / (300+125) pairs).
DEFAULT_VAL_FRAC = 125.0 / (300.0 + 125.0)

# Coarse geographic bin size (degrees) for the geography stratification axis.
GEO_BIN_DEG = 10.0

SPLIT_COLORS = {"train": "#4c72b0", "val": "#dd8452"}


# ---------------------------------------------------------------------------
# Stratified, leakage-safe cell-level assignment
# ---------------------------------------------------------------------------


def _geo_region(lat: float, lon: float) -> str:
    """Coarse lat/lon tile id used as the geography stratification axis."""
    lat_bin = int(np.floor(lat / GEO_BIN_DEG) * GEO_BIN_DEG)
    lon_bin = int(np.floor(lon / GEO_BIN_DEG) * GEO_BIN_DEG)
    return f"{lat_bin:+03d},{lon_bin:+04d}"


def _cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate windows into one row per ERA5 cell (the split unit).

    Each cell carries its window count and a representative value on each
    stratification axis (the modal event_type / year, and a geo region from the
    cell's mean location).
    """
    df = df.copy()
    df["geo_region"] = [
        _geo_region(la, lo) for la, lo in zip(df["latitude"], df["longitude"])
    ]
    has_curated_val = df.get("split")
    if has_curated_val is not None:
        df["is_curated_val"] = df["split"] == "val"
    else:
        df["is_curated_val"] = False

    def _mode(s: pd.Series) -> Any:
        m = s.mode()
        return m.iloc[0] if len(m) else None

    is_pos = (df["window_type"] == "positive").astype(int)
    is_neg = (df["window_type"] == "negative").astype(int)
    grp = df.groupby("era5_cell_id")
    cells = pd.DataFrame(
        {
            "n": grp.size(),
            "n_pos": is_pos.groupby(df["era5_cell_id"]).sum(),
            "n_neg": is_neg.groupby(df["era5_cell_id"]).sum(),
            "event_type": grp["event_type"].agg(_mode),
            "event_year": grp["event_year"].agg(_mode),
            "geo_region": grp["geo_region"].agg(_mode),
            "has_curated_val": grp["is_curated_val"].any(),
        }
    ).reset_index()
    return cells


def assign_splits(
    df: pd.DataFrame, val_frac: float, seed: int
) -> dict[int, str]:
    """Greedily assign whole ERA5 cells to train/val to mirror the overall
    distribution across (event_type, event_year, geo_region).

    Strategy: cells that contain a hand-curated val window are seeded into val
    first (they are the trusted held-out anchors). Remaining cells are then
    assigned val one at a time, always picking the cell that most reduces the
    current val *deficit* summed over its three stratum categories, until the
    target val window count is reached. Ties are broken deterministically by a
    seeded shuffle. The result keeps every ERA5 cell intact → no leakage.

    Returns a mapping era5_cell_id -> "train" | "val".
    """
    df = df.copy()
    df["geo_region"] = [
        _geo_region(la, lo) for la, lo in zip(df["latitude"], df["longitude"])
    ]
    cells = _cell_table(df)
    rng = np.random.default_rng(seed)
    cells = cells.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    total_windows = int(cells["n"].sum())
    val_target = int(round(val_frac * total_windows))

    axes = ["event_type", "event_year", "geo_region"]
    # Per-category target val window counts on each axis.
    cat_target: dict[str, dict[Any, float]] = {}
    for axis in axes:
        totals = df.groupby(axis).size()
        cat_target[axis] = {k: val_frac * v for k, v in totals.items()}

    cat_val: dict[str, dict[Any, float]] = {
        axis: defaultdict(float) for axis in axes
    }
    assignment: dict[int, str] = {}
    val_count = 0

    def _put_val(row: pd.Series) -> None:
        nonlocal val_count
        assignment[int(row["era5_cell_id"])] = "val"
        val_count += int(row["n"])
        for axis in axes:
            cat_val[axis][row[axis]] += int(row["n"])

    # 1) Seed curated-val cells into val (without overshooting the target).
    curated = cells[cells["has_curated_val"]]
    for _, row in curated.iterrows():
        if val_count + int(row["n"]) <= val_target:
            _put_val(row)

    # 2) Greedy fill of the remaining val budget by largest summed deficit.
    remaining = cells[~cells["era5_cell_id"].isin(assignment.keys())].copy()

    def _deficit(row: pd.Series) -> float:
        return sum(
            cat_target[axis].get(row[axis], 0.0) - cat_val[axis].get(row[axis], 0.0)
            for axis in axes
        )

    remaining_idx = list(remaining.index)
    while val_count < val_target and remaining_idx:
        # jitter keeps ties from always resolving the same way, but seeded.
        best_idx = max(
            remaining_idx,
            key=lambda i: (
                _deficit(remaining.loc[i]),
                rng.random(),
            ),
        )
        row = remaining.loc[best_idx]
        # Stop if the smallest remaining cell would overshoot badly; otherwise add.
        _put_val(row)
        remaining_idx.remove(best_idx)

    # 3) Everything not in val is train.
    for _, row in cells.iterrows():
        assignment.setdefault(int(row["era5_cell_id"]), "train")

    logger.info(
        "Assigned %d cells: val target=%d windows, achieved=%d (%.1f%%), train=%d",
        len(assignment),
        val_target,
        val_count,
        100.0 * val_count / total_windows,
        total_windows - val_count,
    )
    return assignment


# ---------------------------------------------------------------------------
# Metadata copying
# ---------------------------------------------------------------------------


def _write_label_layer(dst_dir: Path, meta: dict[str, Any], window_type: str) -> None:
    """Write a one-feature vector label layer for a single window.

    Creates ``<dst_dir>/layers/label/data.geojson`` (a GeoJSON FeatureCollection
    with a single center point carrying ``category = positive|negative``) plus an
    empty ``completed`` marker, matching rslearn's ``GeojsonVectorFormat`` +
    ``FileWindowStorage`` on-disk layout so the layer reads back exactly like a
    normally-materialized one. The point is placed at the window center in the
    window's own projection (pixel coordinates), so after rslearn reprojects to
    the window bounds it falls inside and ``ClassificationTask`` picks it up.
    """
    proj = Projection.deserialize(meta["projection"])
    bounds = meta["bounds"]
    cx = (bounds[0] + bounds[2]) / 2.0
    cy = (bounds[1] + bounds[3]) / 2.0
    category = "positive" if window_type == "positive" else "negative"
    feat = Feature(STGeometry(proj, shapely.Point(cx, cy), None), {LABEL_PROPERTY_NAME: category})

    layer_dir = UPath(str(dst_dir / "layers" / LABEL_LAYER_NAME))
    GeojsonVectorFormat().encode_vector(layer_dir, [feat])
    (layer_dir / "completed").touch()


def copy_tagged_metadata(
    df: pd.DataFrame, ds_root: Path, data_root: Path, write_labels: bool = True
) -> int:
    """Copy each selected window's metadata.json into the tagged dataset tree.

    Writes to ``<data_root>/windows/<group>/<name>/metadata.json``. Sets
    ``options.split`` to the assigned train/val value and preserves the original
    value under ``options.orig_split``.

    When ``write_labels`` is set, also writes a per-window vector ``label`` layer
    (``layers/label/data.geojson`` + ``completed``) encoding the pos/neg target,
    so the dataset is directly consumable as a binary classification eval task
    (see ``_write_label_layer``). The ERA5 raster layer is never copied here — it
    is materialized separately into ``data_root``.
    """
    n_written = 0
    n_missing = 0
    n_labels = 0
    for _, row in df.iterrows():
        group = row["group"]
        name = row["window_name"]
        src = ds_root / "windows" / group / name / "metadata.json"
        try:
            with open(src) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            n_missing += 1
            continue

        opts = meta.setdefault("options", {})
        opts["orig_split"] = opts.get("split")
        opts["split"] = row["split"]

        dst_dir = data_root / "windows" / group / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        with open(dst_dir / "metadata.json", "w") as f:
            json.dump(meta, f)
        n_written += 1

        if write_labels:
            _write_label_layer(dst_dir, meta, row["window_type"])
            n_labels += 1

    if n_missing:
        logger.warning("Skipped %d windows with missing/unreadable metadata", n_missing)
    logger.info("Wrote %d tagged metadata.json files under %s", n_written, data_root)
    if write_labels:
        logger.info("Wrote %d %r vector label layers", n_labels, LABEL_LAYER_NAME)
        ensure_label_layer_in_config(data_root)
    return n_written


def ensure_label_layer_in_config(data_root: Path) -> None:
    """Ensure the dataset ``config.json`` declares the vector ``label`` layer.

    rslearn only reads layers that are declared in the dataset config, so the
    per-window label geojsons are useless until ``label`` is registered as a
    vector layer. This is idempotent: it leaves an existing entry untouched.
    """
    config_path = data_root / "config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.warning(
            "Could not read %s; skipping label-layer registration. Add a vector "
            "%r layer manually.",
            config_path,
            LABEL_LAYER_NAME,
        )
        return

    layers = config.setdefault("layers", {})
    if LABEL_LAYER_NAME in layers:
        logger.info("config.json already declares the %r layer", LABEL_LAYER_NAME)
        return

    layers[LABEL_LAYER_NAME] = {"type": "vector"}
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info("Registered vector %r layer in %s", LABEL_LAYER_NAME, config_path)


def write_labels_for_materialized(data_root: Path) -> int:
    """Write the vector ``label`` layer for every already-materialized window.

    Unlike ``copy_tagged_metadata`` (which is driven by the deduped parquet and
    also re-copies metadata), this walks the existing
    ``<data_root>/windows/<group>/<name>`` tree and writes one label layer per
    window straight from that window's own ``metadata.json`` (``options`` carries
    ``window_type``; the window name is used as a fallback). This guarantees a
    1:1 match with the materialized ERA5 layers and needs no parquet/rescan.
    """
    windows_root = data_root / "windows"
    if not windows_root.exists():
        raise FileNotFoundError(f"No windows tree found at {windows_root}")

    n_labels = 0
    n_missing = 0
    for group_dir in sorted(p for p in windows_root.iterdir() if p.is_dir()):
        for win_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            meta_path = win_dir / "metadata.json"
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                n_missing += 1
                continue
            window_type = meta.get("options", {}).get("window_type")
            if window_type not in ("positive", "negative"):
                # Fall back to the name encoding (e.g. "..._positive_...").
                window_type = "positive" if "_positive_" in win_dir.name else "negative"
            _write_label_layer(win_dir, meta, window_type)
            n_labels += 1

    if n_missing:
        logger.warning("Skipped %d windows with missing/unreadable metadata", n_missing)
    logger.info(
        "Wrote %d %r vector label layers under %s", n_labels, LABEL_LAYER_NAME, data_root
    )
    ensure_label_layer_in_config(data_root)
    return n_labels


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _split_table(df: pd.DataFrame, col: str) -> str:
    ct = pd.crosstab(df[col].fillna("unknown"), df["split"])
    for s in ("train", "val"):
        if s not in ct.columns:
            ct[s] = 0
    ct = ct[["train", "val"]]
    ct["total"] = ct.sum(axis=1)
    ct["val_%"] = (100.0 * ct["val"] / ct["total"]).round(1)
    ct = ct.sort_values("total", ascending=False)
    lines = [f"| {col} | train | val | total | val % |", "|---|---|---|---|---|"]
    for idx, r in ct.iterrows():
        lines.append(
            f"| {idx} | {int(r['train'])} | {int(r['val'])} | "
            f"{int(r['total'])} | {r['val_%']} |"
        )
    lines.append(
        f"| **total** | **{int(ct['train'].sum())}** | **{int(ct['val'].sum())}** | "
        f"**{int(ct['total'].sum())}** | **{round(100.0 * ct['val'].sum() / ct['total'].sum(), 1)}** |"
    )
    return "\n".join(lines)


def write_summary(df: pd.DataFrame, out_path: Path) -> None:
    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    n_cells = df["era5_cell_id"].nunique()
    sections = [
        "# Landslide ERA5 — Weather-Triggered Train/Val Split\n",
        f"**Weather-triggered windows**: {len(df):,} "
        f"(train {n_train} / val {n_val}) across {n_cells} ERA5 cells.\n",
        "Split unit is the ERA5 cell (leakage-safe): every window in a cell shares "
        "a split, so no two windows with the same ERA5 forcing straddle train/val.\n",
        "## By window_type\n",
        _split_table(df, "window_type"),
        "\n## By event_type\n",
        _split_table(df, "event_type"),
        "\n## By event_year\n",
        _split_table(df, "event_year"),
        "\n## By group\n",
        _split_table(df, "group"),
        "\n## By location\n",
        _split_table(df, "location"),
    ]
    out_path.write_text("\n".join(sections) + "\n")
    logger.info("Summary written to %s", out_path)


# ---------------------------------------------------------------------------
# Plot (same layout as visualize_dedup, colored by split)
# ---------------------------------------------------------------------------


def _splits_in(df: pd.DataFrame) -> list[str]:
    return [s for s in ("train", "val") if s in set(df["split"].unique())]


def _stacked_crosstab(df: pd.DataFrame, category: str, splits: list[str]) -> pd.DataFrame:
    ct = pd.crosstab(df[category].fillna("unknown"), df["split"])
    for s in splits:
        if s not in ct.columns:
            ct[s] = 0
    return ct[splits]


def _stacked_barh(ax, ct, splits, title, cap=None) -> None:
    y = np.arange(len(ct))
    left = np.zeros(len(ct))
    for s in splits:
        vals = ct[s].to_numpy()
        ax.barh(y, vals, left=left, color=SPLIT_COLORS[s],
                edgecolor="white", linewidth=0.5, label=s)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(ct.index, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Window count" + (f" (capped at {cap:,})" if cap else ""))
    ax.set_title(title, fontsize=13, fontweight="bold")
    if cap:
        ax.set_xlim(0, cap)
    totals = ct.sum(axis=1).to_numpy()
    for yi, total in zip(y, totals):
        if cap and total > cap:
            ax.text(cap * 0.98, yi, f"{int(total):,}", va="center", ha="right",
                    fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(total + max(totals) * 0.01, yi, f"{int(total):,}",
                    va="center", ha="left", fontsize=9)


def _stacked_barv(ax, ct, splits, title, cap=None) -> None:
    x = np.arange(len(ct))
    bottom = np.zeros(len(ct))
    for s in splits:
        vals = ct[s].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=SPLIT_COLORS[s],
               edgecolor="white", linewidth=0.5, label=s)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in ct.index], fontsize=10)
    ax.set_xlabel("Event year")
    ax.set_ylabel("Window count" + (f" (capped at {cap:,})" if cap else ""))
    ax.set_title(title, fontsize=13, fontweight="bold")
    if cap:
        ax.set_ylim(0, cap)
    totals = ct.sum(axis=1).to_numpy()
    for xi, total in zip(x, totals):
        if cap and total > cap:
            ax.text(xi, cap * 0.98, f"{int(total):,}", ha="center", va="top",
                    fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(xi, total + max(totals) * 0.01, f"{int(total):,}",
                    ha="center", va="bottom", fontsize=9)


def make_plot(df: pd.DataFrame, out_path: Path) -> None:
    splits = _splits_in(df)
    fig = plt.figure(figsize=(20, 14), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.30)

    # Rows 1-2: world map, colored by split.
    ax_map = fig.add_subplot(gs[0:2, :], projection=ccrs.Robinson())
    ax_map.set_global()
    ax_map.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax_map.add_feature(cfeature.OCEAN, facecolor="#e6f2ff")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.4, color="#888888")
    ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, color="#bbbbbb")

    df_u = df.drop_duplicates(subset=["latitude", "longitude"])
    for s in splits:
        sub = df_u[df_u["split"] == s]
        ax_map.scatter(
            sub["longitude"], sub["latitude"], c=SPLIT_COLORS[s], s=12, alpha=0.75,
            transform=ccrs.PlateCarree(), zorder=3 if s == "val" else 2,
            edgecolors="none", label=s,
        )
    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    ax_map.set_title(
        f"Weather-Triggered Landslide Split — {len(df):,} windows "
        f"(train {n_train} / val {n_val})  •  {len(df_u):,} unique locations",
        fontsize=14, fontweight="bold",
    )

    # Row 3, col 1: by location.
    ax_loc = fig.add_subplot(gs[2, 0])
    loc_totals = df["location"].fillna("unknown").value_counts()
    keep = list(loc_totals.head(12).index)
    loc_ct = _stacked_crosstab(df, "location", splits).reindex(keep)
    if len(loc_totals) > 12:
        rest = _stacked_crosstab(
            df[~df["location"].fillna("unknown").isin(keep)], "location", splits
        ).sum(axis=0)
        loc_ct.loc["other"] = rest
    loc_ct.index = [
        s if len(s) <= 15 else s[:14] + "\u2026" for s in loc_ct.index.astype(str)
    ]
    _stacked_barh(ax_loc, loc_ct, splits, "By location")

    # Row 3, col 2: by event year.
    ax_year = fig.add_subplot(gs[2, 1])
    year_ct = _stacked_crosstab(df, "event_year", splits).sort_index()
    _stacked_barv(ax_year, year_ct, splits, "By event year")

    # Row 3, col 3: by event type.
    ax_type = fig.add_subplot(gs[2, 2])
    type_ct = _stacked_crosstab(df, "event_type", splits)
    type_ct = type_ct.loc[type_ct.sum(axis=1).sort_values(ascending=False).index]
    _stacked_barh(ax_type, type_ct, splits, "By event type")

    handles, labels = ax_loc.get_legend_handles_labels()
    fig.legend(handles, labels, title="Split", loc="lower center",
               ncol=len(splits), fontsize=11, title_fontsize=12,
               bbox_to_anchor=(0.5, -0.02))

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved plot to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds-root", type=str, default=DEFAULT_DS_ROOT)
    parser.add_argument("--parquet", type=str, default=str(DEFAULT_PARQUET))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Where analysis artifacts (parquet/summary/plot) go.")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                        help="Where the tagged window metadata tree (dataset) goes.")
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-copy", action="store_true",
        help="Skip copying metadata.json (only write parquet/summary/plot).",
    )
    parser.add_argument(
        "--no-labels", action="store_true",
        help="Skip writing the per-window vector 'label' layer (pos/neg target).",
    )
    parser.add_argument(
        "--labels-only", action="store_true",
        help=(
            "Only write the vector 'label' layer for windows already materialized "
            "under --data-root (skip parquet/split/plot). Use when the split is "
            "already baked into the materialized metadata.json files."
        ),
    )
    args = parser.parse_args()

    if args.labels_only:
        write_labels_for_materialized(Path(args.data_root))
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.parquet)
    weather = df[df["event_type"].isin(WEATHER_TYPES)].copy()
    logger.info(
        "Weather-triggered windows: %d / %d deduped (%d pos / %d neg, %d ERA5 cells)",
        len(weather), len(df),
        int((weather["window_type"] == "positive").sum()),
        int((weather["window_type"] == "negative").sum()),
        weather["era5_cell_id"].nunique(),
    )

    assignment = assign_splits(weather, val_frac=args.val_frac, seed=args.seed)
    weather["split"] = weather["era5_cell_id"].map(assignment)

    # Outputs.
    split_pq = out_dir / "eval_split.parquet"
    weather.to_parquet(str(split_pq), index=False)
    logger.info("Wrote split table to %s (%d rows)", split_pq, len(weather))

    write_summary(weather, out_dir / "split_summary.md")
    make_plot(weather, out_dir / "subsampling.png")

    if not args.no_copy:
        copy_tagged_metadata(
            weather,
            Path(args.ds_root),
            Path(args.data_root),
            write_labels=not args.no_labels,
        )


if __name__ == "__main__":
    main()
