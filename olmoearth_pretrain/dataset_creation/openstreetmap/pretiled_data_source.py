"""rslearn DataSource for pre-tiled OpenStreetMap GeoJSON data.

Drop-in replacement for rslearn.data_sources.openstreetmap.OpenStreetMap.
Reads from a directory of 1-degree WGS84 GeoJSON tiles produced by pretile_osm.py.

Config usage in rslearn config.json:
    "openstreetmap": {
        "data_source": {
            "class_path": "olmoearth_pretrain.dataset_creation.openstreetmap.pretiled_data_source.PretiledOpenStreetMap",
            "init_args": {
                "tiles_dir": "/weka/.../osm_pretiled",
                "categories": { ... same category config ... }
            }
        }
    }
"""

from __future__ import annotations

import json
import math
from typing import Any

import shapely
from upath import UPath

from rslearn.config import QueryConfig
from rslearn.const import WGS84_PROJECTION
from rslearn.data_sources import DataSource, DataSourceContext, Item
from rslearn.data_sources.openstreetmap import FeatureType, Filter
from rslearn.data_sources.utils import MatchedItemGroup, match_candidate_items_to_window
from rslearn.log_utils import get_logger
from rslearn.tile_stores import TileStoreWithLayer
from rslearn.utils import Feature, STGeometry

logger = get_logger(__name__)

TILE_SIZE_DEG = 1.0


class PretiledOsmItem(Item):
    """An item pointing to a pre-tiled GeoJSON file."""

    def __init__(self, name: str, geometry: STGeometry, tile_path: str):
        super().__init__(name, geometry)
        self.tile_path = tile_path

    def serialize(self) -> dict:
        d = super().serialize()
        d["tile_path"] = self.tile_path
        return d

    @staticmethod
    def deserialize(d: dict) -> "PretiledOsmItem":
        item = super(PretiledOsmItem, PretiledOsmItem).deserialize(d)
        return PretiledOsmItem(
            name=item.name, geometry=item.geometry, tile_path=d["tile_path"]
        )


def _parse_categories(raw: dict) -> dict[str, Filter]:
    """Parse category dict from JSON-style config into Filter objects."""
    categories = {}
    for name, spec in raw.items():
        feature_types = None
        if "feature_types" in spec:
            feature_types = [FeatureType(ft.lower()) for ft in spec["feature_types"]]
        categories[name] = Filter(
            feature_types=feature_types,
            tag_conditions=spec.get("tag_conditions"),
            tag_properties=spec.get("tag_properties"),
            to_geometry=spec.get("to_geometry"),
        )
    return categories


class PretiledOpenStreetMap(DataSource[PretiledOsmItem]):
    """DataSource backed by pre-tiled 1-degree GeoJSON files.

    The pre-tiled data is produced once by pretile_osm.py and shared across runs.
    Ingestion reads small GeoJSON files instead of parsing multi-GB PBF files.
    """

    def __init__(
        self,
        tiles_dir: str,
        categories: dict[str, Any] | None = None,
        context: DataSourceContext = DataSourceContext(),
    ):
        self.tiles_dir = UPath(tiles_dir)

        # Categories can be passed as raw dicts (from jsonargparse) or Filter objects.
        # We keep them for potential downstream filtering but the pre-tiled data
        # already contains only features matching these categories.
        if categories and isinstance(next(iter(categories.values())), dict):
            self.categories = _parse_categories(categories)
        elif categories:
            self.categories = categories
        else:
            self.categories = {}

        # Load manifest for spatial index
        manifest_path = self.tiles_dir / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open() as f:
                self.manifest = json.load(f)
        else:
            # Fall back to scanning the directory
            logger.warning(
                "No manifest.json found, scanning directory for tiles..."
            )
            self.manifest = self._scan_tiles()

        self._tile_index = {}
        for tile_name, info in self.manifest.get("tiles", {}).items():
            bounds = info["bounds"]
            lon = int(bounds[0])
            lat = int(bounds[1])
            self._tile_index[(lon, lat)] = (tile_name, info)

        logger.info(
            f"PretiledOpenStreetMap: loaded {len(self._tile_index)} tiles "
            f"from {self.tiles_dir}"
        )

    def _scan_tiles(self) -> dict:
        """Build manifest by scanning the tiles directory."""
        tiles = {}
        for f in sorted(self.tiles_dir.glob("lon_*_lat_*.geojson")):
            name = f.stem
            parts = name.split("_")
            lon = int(parts[1])
            lat = int(parts[3])
            tiles[name] = {
                "bounds": [lon, lat, lon + TILE_SIZE_DEG, lat + TILE_SIZE_DEG],
                "path": f.name,
            }
        return {"tile_size_deg": TILE_SIZE_DEG, "num_tiles": len(tiles), "tiles": tiles}

    def get_items(
        self, geometries: list[STGeometry], query_config: QueryConfig
    ) -> list[list[MatchedItemGroup[PretiledOsmItem]]]:
        # Collect all unique tile items that overlap any geometry
        all_items: dict[str, PretiledOsmItem] = {}
        for geometry in geometries:
            wgs84 = geometry.to_wgs84()
            bounds = wgs84.shp.bounds  # (minx, miny, maxx, maxy)
            lon_start = int(math.floor(bounds[0]))
            lon_end = int(math.floor(bounds[2]))
            lat_start = int(math.floor(bounds[1]))
            lat_end = int(math.floor(bounds[3]))

            for lon in range(lon_start, lon_end + 1):
                for lat in range(lat_start, lat_end + 1):
                    if (lon, lat) not in self._tile_index:
                        continue
                    tile_name, info = self._tile_index[(lon, lat)]
                    if tile_name in all_items:
                        continue
                    tile_bounds = info["bounds"]
                    tile_box = shapely.box(*tile_bounds)
                    tile_path = str(self.tiles_dir / info["path"])
                    all_items[tile_name] = PretiledOsmItem(
                        name=tile_name,
                        geometry=STGeometry(WGS84_PROJECTION, tile_box, None),
                        tile_path=tile_path,
                    )

        items = list(all_items.values())

        # Match items to each geometry using rslearn's standard matching
        wgs84_geometries = [g.to_wgs84() for g in geometries]
        groups = []
        for geometry in wgs84_geometries:
            cur_groups = match_candidate_items_to_window(geometry, items, query_config)
            groups.append(cur_groups)
        return groups

    def deserialize_item(self, serialized_item: dict) -> PretiledOsmItem:
        return PretiledOsmItem.deserialize(serialized_item)

    def ingest(
        self,
        tile_store: TileStoreWithLayer,
        items: list[PretiledOsmItem],
        geometries: list[list[STGeometry]],
    ) -> None:
        for item, item_geometries in zip(items, geometries):
            if tile_store.is_vector_ready(item):
                continue

            logger.info(
                f"Ingesting pre-tiled OSM: {item.name} "
                f"({len(item_geometries)} geometries)"
            )

            tile_path = UPath(item.tile_path)
            if not tile_path.exists():
                logger.warning(f"Tile file missing: {tile_path}")
                continue

            with tile_path.open() as f:
                fc = json.load(f)

            # Convert GeoJSON features to rslearn Feature objects
            features = []
            for gj_feat in fc.get("features", []):
                shp = shapely.geometry.shape(gj_feat["geometry"])
                props = gj_feat.get("properties", {})
                features.append(
                    Feature(STGeometry(WGS84_PROJECTION, shp, None), props)
                )

            tile_store.write_vector(item, features)
