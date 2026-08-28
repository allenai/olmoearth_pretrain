# Colorado Alluvial & Debris Fan Mapping

- **Slug:** `colorado_alluvial_debris_fan_mapping`
- **Status:** `completed`
- **Task type:** classification (alluvial-fan / debris-fan landform segmentation)
- **Source:** Colorado Geological Survey (CGS), Online Series **ON-006** "Alluvial Fan
  Mapping of Colorado" — a statewide effort of county-specific, LiDAR-derived polygon
  inventories of alluvial fans and high-angle / debris fans. Manifest publication (Teller
  County): <https://coloradogeologicalsurvey.org/publications/alluvial-fan-map-data-teller-colorado/>
  ([doi:10.58783/cgs.on00629d.bfqt5710](https://doi.org/10.58783/cgs.on00629d.bfqt5710)).
  **License: free public.**
- **Num samples:** 2000 label tiles.

## What the source is

CGS maps alluvial fans and high-angle / debris fans across Colorado to support land-use and
post-wildfire hazard planning (these landforms are prone to debris flows / mudflows,
especially after fire). Fans were compiled and delineated from Colorado LiDAR datasets
(2–5 ft contours + terrain metrics such as mean slope). Each county is published as its own
"ON-006-##" data product; the effort is ongoing (Teller County = ON-006-29D, v20260121).

### Access method (why the ArcGIS REST service, not the ZIPs)

The per-county data ZIPs / ArcGIS Pro packages (`.ppkx`) on the CGS website are gated behind
a **Gravity Form email-capture download** (no API credential, but an interactive form we
can't script). CGS also serves the *entire statewide inventory* as a single public **ArcGIS
REST MapServer**, which returns the same polygons directly with **no credential**:

```
https://cgsarcimage.mines.edu/arcgis/rest/services/Hazards/ON_006_All_Current_Alluvial_Fan_Mapping_Colorado/MapServer
```

We pull, per county, the two fan-**polygon** layers ("… Alluvial Fans" and "… High Angle /
Debris Fans") via the REST `query` endpoint (`f=geojson`, `outSR=4326`, paged at 2000).
Point layers ("… Debris Fan Points") and county-outline / grouping layers are ignored (the
polygons carry the full footprint; the points are redundant apex markers). Label-only
extraction — no imagery is downloaded. Raw GeoJSON per layer lands in
`raw/colorado_alluvial_debris_fan_mapping/layer_*.geojson`.

Counties covered (as of 2026-07): Boulder, Chaffee, Clear Creek, Fremont, Garfield, Gilpin,
Lake, Pitkin, Summit, Teller.

## Classes

Polygon → per-pixel classification. **uint8**, 255 = nodata (declared, unused — every tile
pixel is observed terrain). Manifest lists two classes; we add a `background` class exactly
as the analogous USGS karst closed-depression dataset does, because CGS maps fans
comprehensively within each county study area, so out-of-polygon pixels are a *genuine
observed negative* ("not a fan"), not a fabricated synthetic negative.

| id | name | definition |
|----|------|------------|
| 0 | `background` | Mapped terrain that is not a delineated fan (real non-fan context around a fan). |
| 1 | `alluvial_fan` | Gently-sloping alluvial fan: fan-shaped sediment deposit where a channel emerges onto lower-gradient ground. |
| 2 | `high_angle_debris_fan` | Steeper (mean slope typically >20°) high-angle / debris fan, downslope of the fan apex / feeder channel; higher debris-flow hazard. |

**Source polygon counts (10 counties, 8,577 total):** 7,208 alluvial fans + 1,369
high-angle/debris fans.

## How labels/classes map to tiles

- **Tiling:** each fan polygon seeds one **64×64** (640 m) tile in a **local UTM** projection
  at **10 m/pixel**, centered on the fan. **All** fan polygons of either class that overlap
  the tile are rasterized (`all_touched=True`, so even a sub-640 m fan yields ≥1 positive
  pixel; high-angle fans burned last so the steeper class wins overlaps). This means a tile
  centered on an alluvial fan also correctly labels an adjacent high-angle fan at its apex,
  and vice-versa. A fan larger than 64 px on an axis (rare) is split into up to 16
  non-overlapping 64×64 windows via bounded random sampling.
- **Selection:** class-balanced by the **seed fan's** class up to **1000 tiles per fan
  class** (spec §5), well under the 25k cap → **2000 tiles** (1000 alluvial-seeded + 1000
  high-angle-seeded). Alluvial fans (the far larger pool) are subsampled; nearly all
  high-angle fans are candidates.
- **Tiles containing each class** (a tile counts toward every class present): background
  2000, alluvial_fan 1210, high_angle_debris_fan 1050.
- **Observability:** fans are 10³–10⁶ m² landforms (median ≈ 8,600 m², p95 ≈ 115,000 m²),
  readily resolved at 10–30 m. A small **900 m² floor** dropped 49 tiny slivers.

## Time-range & change handling

Alluvial / debris fans are **static topographic landforms**, persistent across the Sentinel
era. There is no per-fan event date (LiDAR compiled ~2016–2026), so each sample gets a
**representative 1-year window (2020)** and `change_time = null`. Not a change dataset.

## Caveats

- **Fremont County alluvial fans excluded (geometry unavailable).** The service reports 721
  Fremont "Alluvial Fans" features but returns **null geometry** for all of them via every
  query format (`f=geojson`, `f=json`, `objectIds`, native/other SR) — a server-side data
  issue in that one layer. A handful more (9 Fremont high-angle, 6 Garfield alluvial, 6
  Summit alluvial, plus 3 other malformed) are likewise geometry-less. Net: **742 of 8,577
  source features (~9%) have no usable geometry and are dropped**, leaving **7,832** fans
  (6,433 alluvial + 1,350 high-angle after the area floor). This does **not** constrain the
  final dataset — both classes still have far more than the 1000/class target across the
  other 9 counties. (Fremont's fans could be recovered later from the county ZIP / `.ppkx`
  behind the email form if desired.)
- `background` is an added class not in the 2-class manifest (documented judgment call,
  mirroring `usgs_closed_depressions_in_karst_regions`).
- Adjacent fans of the same class that fall in a neighboring tile are labeled per-tile; the
  neighbor-rendering step captures fans overlapping each tile, so cross-class adjacency
  (alluvial + high-angle in one tile) is preserved.

## Verification (spec §9)

- 2000 `.tif` + 2000 `.json`; all single-band **64×64 uint8**, **EPSG:326xx UTM @ 10 m**,
  values ∈ {0,1,2} with 255 declared nodata; `metadata.json` classes cover all raster
  values.
- Every sample JSON has a **1-year** `time_range` (2020) and `change_time = null` (0 with
  >366-day ranges).
- **Spatial sanity:** all 2000 tile centers fall inside Colorado (lon −109.0…−104.9, lat
  38.4…40.3), matching the mountain counties covered; source_ids trace back to
  county/layer/OID. Georeferencing round-trips through the shared io helpers. (A full
  Sentinel-2 optical overlay was not run — fans are subtle terrain features better seen in a
  DEM/hillshade than in a single optical scene — but the UTM/10 m georeferencing is exact.)

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.colorado_alluvial_debris_fan_mapping
```

Idempotent: raw per-layer GeoJSON is skipped if present; existing `locations/{id}.tif` are
skipped. Requires only public internet access to `cgsarcimage.mines.edu` (no credentials).
A reusable `download.download_arcgis_layer(base_url, layer_id, dst, ...)` helper (paged
ArcGIS REST → GeoJSON) was added to the shared module for this and future ArcGIS sources.
