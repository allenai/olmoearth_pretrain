# AI Dataset for Solar Energy Locations in India

- **Slug:** `ai_dataset_for_solar_energy_locations_in_india`
- **Status:** completed — classification (polygons), 1363 samples
- **Source:** Microsoft / The Nature Conservancy — "An Artificial Intelligence Dataset for
  Solar Energy Locations in India" (Ortiz et al. 2022, arXiv:2202.01340).
  GitHub: https://github.com/microsoft/solar-farms-mapping
- **License:** dataset released under **CDLA-Permissive-2.0** (open, permissive) → usable.
  (The repo *code* is MIT; the *dataset* files are CDLA-Permissive-2.0 per the README.)
- **Access:** unauthenticated HTTPS download of the GeoJSON files from `raw.githubusercontent.com`.

## Source data
A spatially-explicit ML model mapped utility-scale solar PV across India from Sentinel-2
imagery; predictions were **fully validated by human experts** to yield **1363 solar PV
farms** for the year 2021. Three GeoJSONs are published (all EPSG:4326, MultiPolygon,
props: `State`, `Area` m², `Latitude`, `Longitude` = farm center, `fid`):
- `solar_farms_india_2021.geojson` — 4158 raw individual polygon parts (1389 fids).
- `solar_farms_india_2021_merged.geojson` — **used**; raw parts clustered by proximity into
  **1363 farms**, one MultiPolygon per `fid`. This matches the paper's human-validated count
  and gives a clean one-farm = one-sample mapping.
- `..._merged_simplified.geojson` — simplified geometry (not used).

Both the raw and merged files are saved to `raw/{slug}/`; `SOURCE.txt` records provenance.

## Label encoding (polygons recipe, spec §4)
Positive-only, single foreground class. Each merged farm is rasterized into **one** ≤64×64
single-band uint8 UTM tile at **10 m/pixel**:
- `1` = **utility_scale_pv_farm** (panel-array footprint)
- `0` = **background** — the real non-solar land in/around the tile (spatially meaningful,
  NOT a fabricated negative; analogous to the Global Renewables Watch solar path).
- `255` = nodata (declared; not used here — every pixel is 0 or 1).

Tile geometry: centered on the geometry's `representative_point()` (guaranteed inside a
panel polygon — robust to farms whose merged parts are scattered, where a bbox center could
land on empty land), sized to the farm footprint **+ 8 px margin, capped at 64×64**. Farms
larger than 640 m (≈41% of farms exceed 64 px on their long axis) are represented by a 64×64
crop around that point (local footprint + boundary). `all_touched=True` so small/thin farms
stay visible at 10 m. UTM zone picked per-farm from lon/lat (32643/32644 dominate).

## Sampling & time
- **One tile per farm → all 1363 human-validated farms kept.** Single positive class, far
  under the 25k hard cap. (Judgment call: this is slightly above the 1000/class *soft*
  guidance in §5, but I intentionally kept every validated farm — they are all high-quality
  and there is no domination risk with a single class well under 25k.)
- **Time range:** 1-year window anchored on **2021** (`[2021-01-01, 2022-01-01)`); solar
  farms are persistent, so an annual window is valid. No change labels.
- No negatives fabricated (spec §5, positive-only). Downstream assembly supplies negatives.

## Class / value statistics (verification)
- 1363 `.tif` + 1363 `.json` written. All single-band uint8, UTM CRS @ 10 m, sizes 10–64 px
  (all ≤64), values ⊆ {0,1}, nodata 255, `time_range` = 1 year.
- Within-tile solar fraction: median 0.36, mean 0.39; only 5/1363 tiles are 100% solar (the
  rest carry background context for boundary learning).
- **Spatial sanity:** all 1363 tile centers fall inside India's bbox; tile-center vs source
  farm-center distance is median 0.04 km, p90 0.65 km. One outlier (407 km) is a merged
  "farm" whose clustered parts span a large area — its representative point sits on a real
  panel cluster far from the area-weighted center; the tile is still valid solar footprint.
- State coverage recorded in `metadata.json` (`state_counts`); farms span many Indian states
  (Karnataka, Telangana, Rajasthan, Andhra Pradesh, etc.).
- A pixel-perfect Sentinel-2 image overlay was not rendered, but the labels are **natively
  Sentinel-2-derived** (mapped from S2 then human-validated), so co-registration to the S2
  grid is inherent; coordinate checks confirm correct placement.

## Caveats / judgment calls (for review)
- Kept all 1363 farms rather than downsampling to 1000/class (see above).
- Background pixels within footprint-sized tiles are treated as real class-0 background
  (matches the sibling Global Renewables Watch solar encoding), not fabricated negatives.
- A handful of merged farms are geographically scattered; each is represented by one tile on
  its largest/representative cluster rather than tiling every cluster (keeps one-farm =
  one-sample and avoids a few huge farms dominating).
- Label year is 2021 (validated for that year); some farms may have been built earlier — the
  annual 2021 window is a persistent-feature anchor, consistent with the dataset's design.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ai_dataset_for_solar_energy_locations_in_india
```
Idempotent: skips any `locations/{id}.tif` already written. Re-downloads source GeoJSONs
only if absent.
