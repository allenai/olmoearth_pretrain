# OpenStreetMap Salt Ponds / Salt Works

- **Slug**: `openstreetmap_salt_ponds_salt_works`
- **Status**: completed
- **Task type**: classification (single foreground class, positive-only)
- **Num samples**: 1000 label tiles
- **Source**: OpenStreetMap (ODbL), accessed live via the **Overpass API**
- **Manifest label_type**: polygons (~21,000)

## Source & access method

OSM salt features were fetched **by tag** from the Overpass API
(`https://overpass-api.de/api/interpreter`) — deliberately **NOT** from bulk
Geofabrik/planet regional extracts. A sibling OSM dataset
(`openstreetmap_leisure_tourism_extracts`) had been rejected for a ~14 GB
whole-region download runaway, so per spec §8 (impractical-download) we pull only
the thin label layer. The query unions ways + relations carrying:

```
landuse=salt_pond
man_made=salt_pond | man_made=salt_works | man_made=saltern
```

Global `out geom;` returns **21,109 features** (20,793 ways + 316 relations),
~**33 MB** of JSON, in ~9 s. In practice every matched feature also carries
`landuse=salt_pond`, so this is genuinely one class. The raw response is cached at
`raw/openstreetmap_salt_ponds_salt_works/osm_salt_features.json` and re-used on re-run.

## Class mapping

Single unified class (spec §5, manifest "salt pond / salt works"):

| id | name | definition |
|----|------|------------|
| 0 | `salt_pond` | Human-made solar salt evaporation / production ponds and salt works (OSM `landuse=salt_pond`, `man_made=salt_pond/salt_works/saltern`). Large geometric pond complexes clearly discernible at 10 m, distinct from natural salars. |

`255` = nodata/ignore (all outside-polygon pixels).

## Geometry handling

- **Ways** (98.5%) are closed area polygons built directly from the returned node
  coordinates (auto-closed; `buffer(0)` repair on invalidity).
- **Relations** (multipolygons) are reconstructed from member ways: outer members are
  polygonized and unioned; inner members (holes) are polygonized and differenced out.

## Processing decisions

- **Positive-only** (spec §5): OSM tags presence, not absence. Each tile rasterizes its
  polygon footprint to class 0 and leaves all other pixels as nodata (255). No synthetic
  negatives are fabricated; the assembly step supplies negatives from other datasets.
- **Resolvability filter** (spec §4): polygons < `MIN_AREA_M2 = 2500` m² (0.25 ha ≈ 25 px
  at 10 m) are dropped as unresolvable, computed in equal-area EPSG:6933. This removed
  21,109 → 15,067 candidates (~29%); the survivors are the large, clearly-observable
  complexes the manifest describes.
- **Tiling**: one tile per kept polygon, local UTM at 10 m/pixel, `all_touched`
  rasterization. Tile sized to the polygon footprint (padded, `MIN_TILE=8`), capped at
  64×64. Footprints > 640 m yield a 64×64 window centered on the centroid (a
  representative chunk of the complex).
- **Sampling** (spec §5): classification cap is up to 1000 locations per class; with one
  class this yields **1000 tiles**, drawn by seeded shuffle (`balance_by_class`,
  seed=42) from the 15,067 candidates → naturally globally diverse and deterministic.
- **Time range**: salt ponds are persistent land use (static) → representative 1-year
  Sentinel-era window, `REP_YEAR = 2024` → `[2024-01-01, 2025-01-01)`. `change_time` is
  null.

## Verification

- All 1000 `.tif`: single-band, `uint8`, local UTM CRS at 10 m, size ≤ 64×64 (max dim
  64), pixel values ∈ {0, 255}, nodata=255. Each has a matching `.json` with a 1-year
  `time_range`.
- Georeferencing validated by round-tripping tile pixel-center back to WGS84 for several
  samples — coordinates land in known coastal salt-producing regions (e.g. Nicaragua
  −87.49,12.82; Sumbawa/Timor Indonesia 118.77,−8.69 and 125.75,−8.52). Because labels
  are projected from OSM geometry via rslearn's exact projection math, per-tile alignment
  is correct by construction; a live Sentinel-2 overlay was not run (would require imagery
  ingestion) but the coordinate round-trip confirms CRS/orientation correctness.
- Idempotent: re-running reuses the cached Overpass response, re-selects the same 1000
  (seeded), and skips already-written tiles.

## Caveats

- Single positive class only; downstream assembly must supply negatives.
- 1000 of 15,067 resolvable polygons are used (classification per-class cap). The
  remaining candidates are available if a higher cap is ever adopted.
- Overpass is a live endpoint; a future re-download could differ slightly as OSM evolves.
  Delete the cached `raw/.../osm_salt_features.json` to force a fresh pull.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.openstreetmap_salt_ponds_salt_works
```
