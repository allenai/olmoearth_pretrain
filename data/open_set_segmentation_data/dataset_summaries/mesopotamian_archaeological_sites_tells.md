# Mesopotamian Archaeological Sites (tells)

- **Slug:** `mesopotamian_archaeological_sites_tells`
- **Status:** completed
- **Task type:** classification (single presence class)
- **Num samples:** 1000 GeoTIFF tiles
- **Label type:** polygons

## Source

The **FloodPlains Web GIS** (University of Bologna / OrientLab,
<https://floodplains.orientlab.net>), a compilation of all published archaeological
surveys of the southern/central Mesopotamian floodplain (~66,000 km²). The core
ground-truth layer `vw_site_survey_poly` holds **4,934 georeferenced polygons** tracing
the contours of known archaeological occupation mounds ("tells"), drawn from 16 published
survey projects (1950s–present) and confirmed by ground survey / surface-scatter study.

Published CC-BY alongside the human–AI collaboration site-detection work:
- Sci. Rep. 2023 — <https://www.nature.com/articles/s41598-023-36015-5>
- PLOS One 2025 "AI-ming backwards" — <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0330419>

## Access method

The live GeoServer WFS at `floodplains.orientlab.net/geoserver` publishes
GetCapabilities but returns **HTTP 401 on every GetFeature** (even demo layers) — the
web app proxies feature access behind a session, so the WFS is effectively
credential-gated. Instead we used the **published shapefile mirror** shipped in the
Sci. Rep. paper's code repo, resolved via `bit.ly/NSR_floodplains`:

    https://raw.githubusercontent.com/mister-magpie/tell_segmentation/main/shapefiles.zip
    -> shapefiles/site_shape/vw_site_survey_poly.shp  (4,934 polygons, EPSG:4326)

This shapefile carries the same `vw_site_survey_poly` attributes as the WFS layer and
matches the manifest's "~4,934 georeferenced polygons". Raw archive + `SOURCE.txt` are
stored under `raw/mesopotamian_archaeological_sites_tells/`.

## Suitability at 10 m (observability judgment)

Tells are man-made occupation mounds, **not sub-pixel points**. Mapped footprints
(reprojected to UTM): median footprint ≈136 m across (~19 px at 10 m), 90th pct max
dimension ≈506 m; **98.8 % span ≥30 m** and **98 % cover ≥9 pixels** at 10 m. The
persistent topographic/soil/vegetation signature of a mound is detectable in
Sentinel-2/Landsat, so the dataset is accepted and rasterized as polygon masks (not the
1×1 point path the manifest note flagged as a possibility). Only 11/4,934 polygons are
sub-pixel (<10 m); `all_touched=True` plus a center-pixel fallback guarantees each tile
has ≥1 positive pixel.

## Class / label mapping

Single presence class:

| id | name | description |
|----|------|-------------|
| 0 | archaeological mound/tell | rasterized survey-polygon footprint of a known tell |

**Presence-only** (no background/negative class). Following AGENT_SUMMARY §5, outside-
polygon pixels are left as **nodata/ignore (255)**; no synthetic background is fabricated
— the pretraining-assembly step supplies negatives from other datasets. `nodata_value = 255`.

## Tiling / GeoTIFF spec

- Single band, uint8, local UTM (EPSG:32638 / 32639), 10 m/pixel, north-up.
- Each polygon rasterized (`rasterize.rasterize_shapes`, `all_touched=True`) into a tile
  centered on the polygon and **sized to its pixel footprint, capped at 64×64**.
- **326 of 4,934 polygons exceed 640 m** (the great tell-cities — Uruk/Warka, Lagash,
  Girsu, Adab, and a 46 km Samarra-area survey megashape); these overflow a 64 px tile and
  are **center-cropped** to their interior — still a valid all-positive mask. (Whether any
  land in the 1000-sample draw is random.)
- Positive-fraction across tiles: mean 0.73, min 0.03, max 1.0.

## Time range

Sites are persistent/static → a fixed representative 1-year Sentinel-era window
**2020-01-01 … 2021-01-01**. `source_id` carries the site `entry_id` (e.g. `QD001`,
`AKK.1444`). `change_time` is null.

## Sampling

Single class; spec per-class cap is 1000 → **1000 tiles** drawn (seeded, `balance_by_class`)
from the 4,934 polygons. The remaining 3,934 polygons are not emitted (per the 1000/class
corpus-balancing rule), not because of any quality issue.

## Verification

- 1000 `.tif` + 1000 `.json`; every tif single-band uint8, UTM @10 m, ≤64×64, values ⊆ {0, 255}.
- All sample `time_range`s are exactly 1 year; `classes_present == [0]`; metadata class ids
  cover all tif values.
- Georeferencing sanity: all 1000 tile centers reproject to lon 42.0–49.7, lat 30.4–34.4
  (southern/central Iraq floodplain) — consistent with the source. (Full S2 overlay not
  fetched; rasterization is done in the tile's own UTM projection so alignment is exact.)
- Idempotent: re-running skips existing `{sample_id}.tif`.

## Caveats

- Presence-only: no in-tile negatives (handled downstream).
- Very large tell-cities are center-cropped, losing their boundary; acceptable for a
  presence mask.
- Site footprints reflect surveyors' digitized mound extents, which can be approximate at
  the meter level; the ≤64 px tiles and 10 m grid absorb this.

## Reproduce

    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.mesopotamian_archaeological_sites_tells
