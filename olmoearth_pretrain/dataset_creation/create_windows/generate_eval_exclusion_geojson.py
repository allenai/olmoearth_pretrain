"""Generate a WGS84 GeoJSON of eval val/test extents to exclude from pretraining.

Produces one polygon per val/test sample of the PASTIS and yemen_crop evaluations, so
that ``create_windows.from_open_set --exclude_geojson ...`` can drop any pretraining
window whose footprint intersects an eval extent.

Sources (both optional; pass whichever you have local/weka access to):
  * PASTIS: the raw PASTIS-R ``metadata.geojson`` (has ``geometry``, ``ID_PATCH``,
    ``Fold``). Per ``pastis_processor.py`` the official split is folds 1-3 = train,
    fold 4 = val, fold 5 = test. Patch polygons for the val/test folds are reprojected
    to WGS84.
  * yemen_crop: the ingested rslearn dataset. Windows carry a ``split`` tag in
    ``window.options`` ("train"/"val"/"test"); val/test window footprints are
    reprojected to WGS84.

The other three evals in scope are handled differently and are NOT emitted here:
eurosat / so2sat are excluded at the dataset level (see ``EXCLUDED_SLUGS``), and MADOS
is not in the open-set bank and has no recoverable geocoordinates.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset import Dataset
from rslearn.utils.geometry import STGeometry
from upath import UPath

# Default val/test split names (accept both "val" and "valid").
_DEFAULT_SPLITS = ("val", "valid", "test")
# PASTIS official fold -> split mapping (folds 1-3 are train).
_PASTIS_FOLD_TO_SPLIT = {4: "val", 5: "test"}

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_PATH = (
    _REPO_ROOT / "data" / "open_set_segmentation_data" / "eval_exclusion.geojson"
)


def pastis_features(
    metadata_geojson: str, folds: dict[int, str] = _PASTIS_FOLD_TO_SPLIT
) -> list[dict[str, Any]]:
    """Return WGS84 polygon features for PASTIS val/test patches."""
    import geopandas as gpd

    gdf = gpd.read_file(metadata_geojson)
    gdf = gdf[gdf["Fold"].isin(folds.keys())]
    gdf = gdf.to_crs(4326)

    features = []
    for _, row in gdf.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(row.geometry),
                "properties": {
                    "dataset": "pastis",
                    "split": folds[int(row["Fold"])],
                    "sample_id": str(row.get("ID_PATCH")),
                },
            }
        )
    return features


def rslearn_split_features(
    ds_path: str,
    dataset_name: str,
    split_tag_key: str = "split",
    splits: tuple[str, ...] = _DEFAULT_SPLITS,
    workers: int = 32,
) -> list[dict[str, Any]]:
    """Return WGS84 polygon features for val/test windows of an rslearn eval dataset.

    Args:
        ds_path: the ingested rslearn eval dataset path.
        dataset_name: value recorded in each feature's ``dataset`` property.
        split_tag_key: the ``window.options`` key holding the split. yemen_crop uses
            ``eval_split`` (see the studio_ingest registry's ``split_tag_key``).
        splits: split values to keep (val/test).
        workers: window-loading workers.
    """
    dataset = Dataset(UPath(ds_path))
    features = []
    for window in dataset.load_windows(workers=workers, show_progress=True):
        split = window.options.get(split_tag_key)
        if split not in splits:
            continue
        box = shapely.box(*window.bounds)
        geom = STGeometry(window.projection, box, None).to_projection(WGS84_PROJECTION)
        features.append(
            {
                "type": "Feature",
                "geometry": shapely.geometry.mapping(geom.shp),
                "properties": {
                    "dataset": dataset_name,
                    "split": "val" if split == "valid" else split,
                    "sample_id": window.name,
                },
            }
        )
    return features


def write_feature_collection(features: list[dict[str, Any]], output_path: Path) -> None:
    """Write features as a WGS84 FeatureCollection atomically."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fc = {"type": "FeatureCollection", "features": features}
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(fc, f)
    tmp.rename(output_path)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pastis_metadata",
        type=str,
        default=None,
        help="Path to the raw PASTIS-R metadata.geojson",
    )
    parser.add_argument(
        "--yemen_ds_path",
        type=str,
        default=None,
        help=(
            "Path to the ingested yemen_crop rslearn dataset "
            "(e.g. /weka/dfive-default/olmoearth/eval_datasets/yemen_crop)"
        ),
    )
    parser.add_argument(
        "--yemen_split_tag_key",
        type=str,
        default="eval_split",
        help="window.options key holding the split for yemen_crop",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to write the exclusion GeoJSON",
    )
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    features: list[dict[str, Any]] = []
    if args.pastis_metadata:
        pf = pastis_features(args.pastis_metadata)
        print(f"PASTIS: {len(pf)} val/test patch polygons")
        features.extend(pf)
    if args.yemen_ds_path:
        yf = rslearn_split_features(
            args.yemen_ds_path,
            "yemen_crop",
            split_tag_key=args.yemen_split_tag_key,
            workers=args.workers,
        )
        print(f"yemen_crop: {len(yf)} val/test window polygons")
        features.extend(yf)

    if not features:
        raise SystemExit(
            "No sources given. Pass --pastis_metadata and/or --yemen_ds_path."
        )

    write_feature_collection(features, Path(args.output))
    print(f"Wrote {len(features)} features to {args.output}")


if __name__ == "__main__":
    main()
