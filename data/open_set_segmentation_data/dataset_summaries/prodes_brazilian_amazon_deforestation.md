# PRODES (Brazilian Amazon deforestation)

- **Slug**: `prodes_brazilian_amazon_deforestation`
- **Status**: completed
- **Task type**: classification (change dataset: forest → clear-cut)
- **Samples**: 1000 tiles (64×64, UTM, 10 m)
- **Source**: INPE TerraBrasilis PRODES, WFS GeoServer
  `https://terrabrasilis.dpi.inpe.br/geoserver/ows`
- **Layer**: `prodes-legal-amz:yearly_deforestation` (Brazilian Legal Amazon yearly
  clear-cut increment)
- **License**: CC-BY-SA-4.0
- **Annotation**: manual photointerpretation by INPE analysts (Landsat-class / CBERS /
  Sentinel-2 imagery)

## What the source is

PRODES is Brazil's official, annual, wall-to-wall monitoring of **clear-cut (corte raso)
deforestation** in the Legal Amazon. Each yearly increment is a set of polygons of newly
clear-cut primary forest. Relevant attributes: `year` (PRODES reference year, Aug–Jul
cycle), `image_date` (day-precise satellite image date on which the clear-cut was
confirmed), `main_class` (always `DESMATAMENTO`), `sub_class` (exposed-soil vs
residual-vegetation clear-cut for recent years, a `d{year}` placeholder for older ones),
`state`, `area_km`. Native CRS EPSG:4674 (SIRGAS 2000); WFS reprojects to EPSG:4326.

## Why accepted / triage

This is a good-fit **change** dataset. PRODES clear-cut is directly observable at 10–30 m
and the polygons carry a **day-precise `image_date`**, which places the event well within
the spec's ~1–2 month change-timing precision requirement (spec §5) — so it is a genuine
`change_time` dataset, not a rejected year-only change label. No credentials required
(public WFS).

## Class scheme (uint8)

| id | name | meaning |
|----|------|---------|
| 0 | background | no PRODES clear-cut in this pixel (standing forest / other cover) |
| 1 | deforestation | PRODES annual clear-cut / corte raso |
| 255 | nodata | (not used here; reserved) |

The yearly increment layer contains only one phenomenon (clear-cut deforestation), so the
scheme is binary. Both classes are present in every tile (mean tile has ample background
context), so under tiles-per-class balancing both classes reach 1000 with 1000 tiles.

## Encoding & change handling

- One tile per selected polygon: a 64×64 (640 m) UTM 10 m tile centered on the polygon
  centroid; the polygon rasterized (`all_touched=True`) as class 1, background 0 elsewhere.
  Only the target polygon is drawn (co-located clear-cuts of other dates left as
  background). Mirrors the DETER-B recipe / shared `rasterize` + `io` utilities.
- **`change_time` = `image_date`** (day-precise), which splits the sample into two adjacent
  six-month windows (via `io.pre_post_time_ranges`): **`pre_time_range`** = the ~6 months
  (≤183 days) immediately before the clear-cut and **`post_time_range`** = the ~6 months
  (≤183 days) immediately after, with **`time_range` = null** (total span still ~1 year). A
  completed clear-cut persists in imagery, so the pre/post pairing split at the confirmation
  date is well-posed; pretraining pairs the "before" stack with the "after" stack and probes
  on their difference.
- **Post-2016 filter**: only polygons with `image_date >= 2016-01-01` kept (Sentinel era).
  PRODES-year-2016 polygons imaged in late 2015 are dropped.
- **Giant-polygon guard**: selected polygons filtered to `0.002 <= area_km <= 0.4` km² so
  the tile keeps visible background context (a 640 m tile is 0.41 km²) and pathological
  huge clearings do not fill the whole tile.

## Sampling

Fetched candidates per (state, year) via WFS with a CQL `area_km` filter (9 Legal Amazon
states × 10 PRODES years 2016–2025, `FETCH_PER_QUERY=80`). Selected up to 1000 tiles
round-robin across years (shuffled within year, seed 42), giving **100 tiles/year**
spanning 2016–2025 and geographic spread across states. Well under the 25k cap and the
254-class uint8 cap.

## Verification (spec §9)

- 1000 `.tif` + 1000 `.json`; every `.tif` single-band, uint8, 64×64, UTM at 10 m
  (10 distinct UTM zones across the Amazon).
- Pixel values ∈ {0, 1} only; nodata=255 declared; both classes present in every tile.
- Every `.json` has a non-null `change_time` with adjacent ≤183-day
  `pre_time_range`/`post_time_range` windows split at it and `time_range` = null.
- All 1000 tile centroids fall inside the Legal Amazon bounding box (lon −73.1…−44.0,
  lat −16.5…+5.0).
- Georeferencing validated via WGS84→UTM reprojection round-trips + centroid checks; a
  full Sentinel-2 raster overlay was not rendered here, but the encoding path is identical
  to the validated DETER-B dataset (same source/CRS/recipe).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.prodes_brazilian_amazon_deforestation
```

Idempotent: existing `locations/{id}.tif` and cached `raw/` GeoJSONs are skipped.

## Caveats

- Single-phenomenon dataset (clear-cut only); PRODES degradation/other classes live in the
  separate DETER dataset. Deliberately kept binary to match the yearly increment semantics.
- Area filter biases toward small/medium clearings (needed for in-tile background); very
  large mechanized clearings are excluded rather than cropped.
