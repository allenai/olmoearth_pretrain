"""Build the US socioeconomic eval dataset (income, home value, population).

MOSAIKS-style tile regression (Rolf et al. 2021), rebuilt from public-domain
sources so the labels match the imagery year instead of 2015:

* ``log_income``       -- ACS 5-year 2020-2024 median household income (B19013)
* ``log_home_value``   -- ACS 5-year 2020-2024 median home value (B25077); the
                          public stand-in for MOSAIKS' Zillow ZTRAX label, which
                          is not redistributable
* ``log_pop_density``  -- log1p(people per km^2) from ACS total population (B01003)

All three come from block groups. A tile's income / home value is the mean over
the block groups it overlaps, weighted by the people each contributes (block-group
population x the fraction of its area inside the tile, i.e. population assumed
uniform within a block group). MOSAIKS weighted by area alone; weighting by people
keeps a tile that is mostly park from taking the income of an empty block group.
A label is kept only when block groups with a valid estimate supply at least
MIN_VALID_FRAC of the tile's people -- ACS suppresses income/home value for small
or renter-only block groups.

Tiles are 128 x 128 px at 10 m (1.28 km) on the same UTM grid as pretraining,
sampled population-weighted over CONUS (a uniform sample is mostly empty land,
where income is barely defined). Splits are spatially blocked on a BLOCK_DEG grid
so neighbouring tiles never straddle train and test. Imagery is the twelve monthly
S2 / S1 / Landsat layers for IMAGERY_YEAR, the centre of the ACS 2020-2024 period,
copied verbatim from a registered year-aligned eval's config.json (LAYERS_FROM):
Planetary Computer S2/S1 and the AWS Landsat bucket, so the imagery is fetched
exactly as the other year-aligned evals' was. The repo's pretraining configs moved
to the olmoearth_datasets service in July (24ffe0737) and are deliberately not used.

Sources (all US Census Bureau, public domain; no API key needed):
  ACS table-based summary files  www2.census.gov/programs-surveys/acs/summary_file
  Cartographic boundary block groups  www2.census.gov/geo/tiger/GENZ2024

Usage (on a weka-mounted machine; ``build`` downloads ~150 MB)::

    python scripts/tools/build_us_socioeconomic_eval.py build \
        --ds_path /weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/us_socioeconomic \
        --cache_dir /weka/dfive-default/helios/dataset_creation/us_socioeconomic_src
    rslearn dataset prepare --root $DS_PATH --workers 16 --retry-backoff-seconds 2
    rslearn dataset materialize --root $DS_PATH --workers 64

Landsat materialize needs AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (requester-pays
bucket). ``write_config`` rewrites config.json on an existing dataset, e.g. after
changing LAYERS_FROM, as long as no imagery has been prepared yet.
"""

import argparse
import functools
import hashlib
import json
import logging
import math
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from rasterio.crs import CRS
from rslearn.dataset import Dataset, Window
from rslearn.utils import Projection, STGeometry, get_utm_ups_crs
from rslearn.utils.feature import Feature
from rslearn.utils.vector_format import GeojsonVectorFormat
from upath import UPath

logger = logging.getLogger(__name__)

ACS_YEAR = 2024  # 5-year estimates 2020-2024
IMAGERY_YEAR = 2022  # centre of the ACS period
ACS_URL = (
    "https://www2.census.gov/programs-surveys/acs/summary_file/{year}/"
    "table-based-SF/data/5YRData/acsdt5y{year}-{table}.dat"
)
BG_URL = "https://www2.census.gov/geo/tiger/GENZ{year}/shp/cb_{year}_us_bg_500k.zip"
ACS_TABLES = {
    "income": ("b19013", "B19013_E001"),
    "home_value": ("b25077", "B25077_E001"),
    "pop": ("b01003", "B01003_E001"),
}
BG_GEOID_PREFIX = "1500000US"
# Contiguous US: drop Alaska, Hawaii and the territories.
NON_CONUS_STATEFP = {"02", "15", "60", "66", "69", "72", "78"}
EQUAL_AREA_CRS = "EPSG:5070"  # CONUS Albers

RESOLUTION = 10
TILE_PX = 128
TILE_AREA_KM2 = (TILE_PX * RESOLUTION / 1000) ** 2
NUM_TILES = 10_000
MIN_VALID_FRAC = 0.5
BLOCK_DEG = 0.5
SPLIT_FRACS = (("train", 0.6), ("val", 0.2), ("test", 0.2))
SEED = 20261029

# The registered year-aligned eval whose imagery layers are copied.
LAYERS_FROM = (
    "/weka/dfive-default/olmoearth/eval_datasets/us_trees_year_aligned/config.json"
)
IMAGERY_LAYER_PREFIXES = ("sentinel2_l2a_mo", "sentinel1_mo", "landsat_mo")
NUM_IMAGERY_LAYERS = 36

WINDOW_GROUP = "default"
LABEL_LAYER = "label"
LABEL_PROPERTIES = ("log_income", "log_home_value", "log_pop_density")


def _download(url: str, dst: Path) -> Path:
    if not dst.exists():
        logger.info("downloading %s", url)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        # Only ever called with the fixed https census.gov URLs above.
        urllib.request.urlretrieve(url, tmp)  # nosec B310
        tmp.rename(dst)
    return dst


def load_acs_table(path: Path, column: str) -> pd.Series:
    """Read one ACS summary-file table into a block-group GEOID -> value series.

    Suppressed or unavailable estimates (blank, or the negative jam values) are
    NaN.
    """
    df = pd.read_csv(path, sep="|", usecols=["GEO_ID", column], dtype=str)
    df = df[df["GEO_ID"].str.startswith(BG_GEOID_PREFIX)]
    values = pd.to_numeric(df[column], errors="coerce")
    values[values < 0] = np.nan
    return pd.Series(
        values.to_numpy(),
        index=df["GEO_ID"].str.removeprefix(BG_GEOID_PREFIX).to_numpy(),
    )


def load_block_groups(cache_dir: Path) -> gpd.GeoDataFrame:
    """CONUS block groups in EPSG:5070 with income, home_value and pop columns."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = _download(
        BG_URL.format(year=ACS_YEAR), cache_dir / f"cb_{ACS_YEAR}_us_bg_500k.zip"
    )
    with zipfile.ZipFile(zip_path) as zf:
        shp = next(n for n in zf.namelist() if n.endswith(".shp"))
    bgs = gpd.read_file(f"zip://{zip_path}!{shp}", columns=["GEOID", "STATEFP"])
    bgs = bgs[~bgs["STATEFP"].isin(NON_CONUS_STATEFP)].to_crs(EQUAL_AREA_CRS)
    for name, (table, column) in ACS_TABLES.items():
        path = _download(
            ACS_URL.format(year=ACS_YEAR, table=table),
            cache_dir / f"acsdt5y{ACS_YEAR}-{table}.dat",
        )
        bgs[name] = bgs["GEOID"].map(load_acs_table(path, column))
    bgs["pop"] = bgs["pop"].fillna(0.0)
    bgs["area_m2"] = bgs.geometry.area
    bgs = bgs[bgs["area_m2"] > 0].reset_index(drop=True)
    logger.info(
        "%d CONUS block groups, %.1fM people; income missing %.1f%%, home value "
        "missing %.1f%%",
        len(bgs),
        bgs["pop"].sum() / 1e6,
        100 * bgs["income"].isna().mean(),
        100 * bgs["home_value"].isna().mean(),
    )
    return bgs


@functools.cache
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def tile_for_point(lon: float, lat: float) -> tuple[str, int, int]:
    """The (crs, col, row) of the 128 px pretraining-grid tile containing a point."""
    crs = get_utm_ups_crs(lon, lat).to_string()
    x, y = _transformer("EPSG:4326", crs).transform(lon, lat)
    size = TILE_PX * RESOLUTION
    # Pixel rows grow southward (y resolution is -RESOLUTION).
    return crs, math.floor(x / size), math.floor(-y / size)


def tile_polygon(crs: str, col: int, row: int) -> shapely.Polygon:
    """A tile's footprint in projected metres of its own CRS."""
    size = TILE_PX * RESOLUTION
    return shapely.box(col * size, -(row + 1) * size, (col + 1) * size, -row * size)


def sample_tiles(
    bgs: gpd.GeoDataFrame, num_tiles: int, rng: np.random.Generator
) -> list[tuple[str, int, int]]:
    """Population-weighted, de-duplicated tiles (people -> random point -> tile)."""
    weights = bgs["pop"].to_numpy() / bgs["pop"].sum()
    to_lonlat = _transformer(EQUAL_AREA_CRS, "EPSG:4326")
    tiles: dict[tuple[str, int, int], None] = {}
    while len(tiles) < num_tiles:
        for idx in rng.choice(len(bgs), size=num_tiles, p=weights):
            geom = bgs.geometry.iloc[idx]
            point = shapely.Point(*rng.uniform(geom.bounds[:2], geom.bounds[2:]))
            if not geom.contains(point):
                continue  # rejection sampling inside the block group
            tiles.setdefault(tile_for_point(*to_lonlat.transform(point.x, point.y)))
            if len(tiles) == num_tiles:
                break
    return list(tiles)


def tile_labels(
    bgs: gpd.GeoDataFrame, tiles: list[tuple[str, int, int]]
) -> pd.DataFrame:
    """People-weighted block-group labels for each tile."""
    tree = shapely.STRtree(bgs.geometry.to_numpy())
    rows = []
    for crs, col, row in tiles:
        to_albers = _transformer(crs, EQUAL_AREA_CRS)
        footprint = shapely.transform(
            tile_polygon(crs, col, row),
            lambda xy: np.column_stack(to_albers.transform(xy[:, 0], xy[:, 1])),
        )
        idx = tree.query(footprint, predicate="intersects")
        overlap = shapely.area(
            shapely.intersection(bgs.geometry.to_numpy()[idx], footprint)
        )
        sub = bgs.iloc[idx]
        people = sub["pop"].to_numpy() * overlap / sub["area_m2"].to_numpy()
        total = people.sum()
        rec = {"crs": crs, "col": col, "row": row, "people": total}
        rec["log_pop_density"] = math.log1p(total / TILE_AREA_KM2)
        for name in ("income", "home_value"):
            values = sub[name].to_numpy()
            ok = ~np.isnan(values)
            valid_people = people[ok].sum()
            rec[f"{name}_valid_frac"] = valid_people / total if total > 0 else 0.0
            if total > 0 and valid_people / total >= MIN_VALID_FRAC:
                rec[name] = float((people[ok] * values[ok]).sum() / valid_people)
                rec[f"log_{name}"] = math.log(rec[name])
        rows.append(rec)
    return pd.DataFrame(rows)


def split_for(lon: float, lat: float) -> str:
    """Deterministic train/val/test assignment of a BLOCK_DEG spatial block."""
    block = f"{math.floor(lon / BLOCK_DEG)}_{math.floor(lat / BLOCK_DEG)}"
    u = int(hashlib.sha256(block.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, frac in SPLIT_FRACS:
        cumulative += frac
        if u < cumulative:
            return name
    return SPLIT_FRACS[-1][0]


def imagery_layers_from(config_path: UPath) -> dict:
    """The 36 monthly S2 / S1 / Landsat layers of another dataset's config.json."""
    config = json.loads(config_path.read_text())
    layers = {
        name: layer
        for name, layer in config["layers"].items()
        if name.startswith(IMAGERY_LAYER_PREFIXES)
    }
    if len(layers) != NUM_IMAGERY_LAYERS:
        raise ValueError(
            f"expected {NUM_IMAGERY_LAYERS} monthly imagery layers in {config_path}, "
            f"found {len(layers)}"
        )
    return layers


def write_config(ds_path: UPath, layers_from: UPath) -> None:
    """config.json: the year-aligned monthly imagery plus the vector label layer."""
    layers = imagery_layers_from(layers_from)
    layers[LABEL_LAYER] = {"type": "vector"}
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "config.json").write_text(json.dumps({"layers": layers}, indent=2))


def write_windows(ds_path: UPath, labels: pd.DataFrame) -> Counter:
    """One window per tile with its labels as a single vector feature."""
    dataset = Dataset(ds_path)
    time_range = (
        datetime(IMAGERY_YEAR, 1, 1, tzinfo=UTC),
        datetime(IMAGERY_YEAR + 1, 1, 1, tzinfo=UTC),
    )
    splits: Counter = Counter()
    for rec in labels.itertuples(index=False):
        projection = Projection(CRS.from_string(rec.crs), RESOLUTION, -RESOLUTION)
        bounds = (
            rec.col * TILE_PX,
            rec.row * TILE_PX,
            (rec.col + 1) * TILE_PX,
            (rec.row + 1) * TILE_PX,
        )
        to_lonlat = _transformer(rec.crs, "EPSG:4326")
        centre = tile_polygon(rec.crs, rec.col, rec.row).centroid
        split = split_for(*to_lonlat.transform(centre.x, centre.y))
        splits[split] += 1
        window = Window(
            storage=dataset.storage,
            group=WINDOW_GROUP,
            name=f"{rec.crs.replace(':', '_')}_{rec.col}_{rec.row}",
            projection=projection,
            bounds=bounds,
            time_range=time_range,
            options={"split": split},
        )
        window.save()
        properties = {
            k: getattr(rec, k)
            for k in (*LABEL_PROPERTIES, "income", "home_value", "people")
            if k in labels.columns and not pd.isna(getattr(rec, k))
        }
        feature = Feature(
            STGeometry(projection, shapely.box(*bounds), None), properties
        )
        GeojsonVectorFormat().encode_vector(
            window.get_layer_dir(LABEL_LAYER), [feature]
        )
        window.mark_layer_completed(LABEL_LAYER)
    return splits


def summarize(labels: pd.DataFrame) -> None:
    """Print label coverage and distributions."""
    print(f"tiles: {len(labels)}")
    for name in LABEL_PROPERTIES:
        col = labels[name] if name in labels.columns else pd.Series(dtype=float)
        print(
            f"  {name:16} valid {col.notna().sum():6d}  "
            + "  ".join(
                f"p{q:02d}={col.quantile(q / 100):.3f}" for q in (1, 25, 50, 75, 99)
            )
        )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["build", "write_config"])
    parser.add_argument("--ds_path", required=True)
    parser.add_argument("--cache_dir")
    parser.add_argument("--num_tiles", type=int, default=NUM_TILES)
    parser.add_argument("--layers_from", default=LAYERS_FROM)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    ds_path = UPath(args.ds_path)
    layers_from = UPath(args.layers_from)
    if args.command == "write_config":
        prepared = list((ds_path / "windows").glob("*/*/items.json"))
        if prepared:
            raise SystemExit(
                f"{len(prepared)} windows already have prepared imagery; "
                "refusing to swap their data sources"
            )
        write_config(ds_path, layers_from)
        print(f"wrote {ds_path / 'config.json'} from {layers_from}")
        return
    if args.cache_dir is None:
        raise SystemExit("build needs --cache_dir")
    if (ds_path / "windows").exists():
        raise SystemExit(f"{ds_path} already has windows; refusing to add more")
    bgs = load_block_groups(Path(args.cache_dir))
    tiles = sample_tiles(bgs, args.num_tiles, np.random.default_rng(SEED))
    labels = tile_labels(bgs, tiles)
    labels.to_csv(Path(args.cache_dir) / "tile_labels.csv", index=False)
    summarize(labels)
    write_config(ds_path, layers_from)
    splits = write_windows(ds_path, labels)
    print("splits:", dict(splits))


if __name__ == "__main__":
    main()
