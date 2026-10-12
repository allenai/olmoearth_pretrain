# Colombia SIMCI Coca Monitoring

- **Slug**: `colombia_simci_coca_monitoring`
- **Status**: completed
- **Task type**: classification (weak coca presence/absence)
- **Label type**: dense_raster (homogeneous 1 km cell -> uniform 640 m tile)
- **Family / region**: illicit_crop / Colombia
- **num_samples**: 1578 (coca=578, no_coca=1000)

## Source

UNODC-Colombia **SIMCI** (Sistema Integrado de Monitoreo de Cultivos Ilícitos),
*"Densidad de Cultivos de Coca"*, published on the Colombian open-data portal
**datos.gov.co** (Socrata dataset id `v3rx-q7t3`).

- Portal: https://www.datos.gov.co/Justicia-y-Derecho/Densidad-de-Cultivos-de-Coca-Subdirecci-n-Estrat-g/v3rx-q7t3
- License: Colombia open government data (datos.gov.co).
- Annotation method: UNODC-SIMCI annual coca census — coca cultivation detected from
  very-high-resolution satellite imagery interpretation with field verification,
  aggregated to a **1 km grid** (hectares of coca per cell per year).

The product is a national 1 km grid: each feature is one ~1 km × 1 km cell (a MultiPolygon
in WGS84) with a column `areaCoca_YYYY` = area (hectares, 0–100) of coca detected inside the
cell that census year, for every year **2001–2024**. It has **119,154 cells** covering the
monitored coca belt of Colombia (not the whole country — so cells are within coca-suitable
terrain). Verified cell size ≈ 1001 × 1001 m.

## Access / download

Openly downloadable, no credential. Socrata GeoJSON export (one FeatureCollection):

```
https://www.datos.gov.co/resource/v3rx-q7t3.geojson?$limit=150000
```

Cached at `raw/colombia_simci_coca_monitoring/coca_grid.geojson` (~101 MB, 119,154 cells).
Label-only; no imagery is downloaded.

## Design decisions

**Classification, not regression.** The source is a *coarse 1 km density aggregate*; coca is
not localized within a cell, so a per-pixel regression value would be spuriously precise.
The manifest offered either treatment; per the SOP derived-product-map guidance (prefer
homogeneous / high-confidence areas), a **weak presence/absence classification** is the
honest representation.

**Two classes** (both high-confidence; a wide gap between them is excluded on purpose):

| id | name    | definition |
|----|---------|------------|
| 0  | no_coca | cell `areaCoca == 0 ha` that year — a **hard negative**: monitored coca belt, no coca detected. |
| 1  | coca    | cell `areaCoca >= 50 ha` that year (≥ 50 % of the 1 km² cell) — a **coca-dominated / homogeneous** cell. |

The 1–49 ha cells are excluded so both classes stay homogeneous/high-confidence. Threshold
of **50 ha** chosen from the distribution (candidate counts: ≥40 ha → 1479 cell-years / 619
cells; **≥50 ha → 578 / 262**; ≥60 ha → 227 / 139). 50 ha keeps a strong-majority ("≥50 %
of the cell is coca") positive definition with a reasonable count; per-year max density is
only ~55–83 ha so higher thresholds thin out fast.

**Tile.** Each qualifying (cell, year) → one **64×64 (640 m)** local-UTM 10 m tile, centered
on the cell centroid and **filled uniformly** with the class id (the 1 km cell is larger and
homogeneous, so the 640 m tile sits inside it). uint8, nodata=255 (unused; tiles are fully
filled). UTM zone picked per-cell from the centroid (EPSG:326xx, mostly 32618/32619).

**Time (SOP §5).** Annual product → **1-year window anchored on each labeled year**
(`year_range(y)` = [Jan 1 y, Jan 1 y+1)). Only **post-2016** years kept (2016–2024, Sentinel
era). `change_time = null` (annual presence *state*, not a dated event). One sample per
qualifying (cell, year), so a persistent hotspot cell contributes multiple yearly samples
with distinct windows.

**Sampling (SOP §5).** Up to 1000/class, balanced by class (25k cap). Positives (578) are
below the target → **all kept** (SOP: keep sparse classes; downstream assembly filters/adds
negatives). Negatives drawn from a bounded random pool (seed 42) of the ~721k zero
cell-years, balanced to 1000. Deterministic (fixed seed + stable ordering).

## RESOLUTION CAVEAT (important)

This is a **coarse weak label**. The source resolves coca only to a 1 km cell, so the
uniform per-tile label is **region-level**, not per-pixel exact — a "coca" tile means "this
~640 m area lies in a coca-cultivation-*dominated* 1 km cell" (≥50 % coca), not that every
10 m pixel is coca. Suitable as a weak pretraining label, not as a precise coca segmentation
mask.

## Sample counts

- no_coca: **1000**
- coca: **578** — per year: 2016:97, 2017:43, 2018:27, 2019:4, 2020:2, 2021:59, 2022:59,
  2023:78, 2024:209 (reflects the SIMCI coca-area trend: dip ~2019–2020, rise to a 2024 peak).

## Verification (SOP §9)

- 1578 `.tif` each with a matching `.json`. Opened several: single band, **uint8**, UTM CRS
  at **10 m**, **64×64**, nodata 255, pixel values ∈ {0, 1}; sampled unique values across the
  dataset = {0, 1} (covered by the class map).
- Every sample JSON has a **1-year** `time_range` and `change_time = null`.
- Geographic sanity: all tile centers fall inside Colombia (lon −78.7…−69.4, lat 0.6…9.1);
  coca positives cluster in known intensive-coca regions (e.g. Nariño/Tumaco Pacific coast at
  −78.6, 1.5), no_coca in the eastern plains. Cross-checked source densities: coca cells
  58.5 / 55.7 ha, no_coca cell 0.0 ha — consistent with the labels. A full Sentinel-2 pixel
  overlay was not performed because the label is a 1 km region-level weak label (per-pixel
  overlay is not meaningful at this granularity); alignment is confirmed via correct
  georeferencing and placement in known coca zones.
- Idempotent: re-running skips existing outputs (2nd run finished in ~6 s, no rewrites).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.colombia_simci_coca_monitoring
```

Outputs on weka under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/`:
- `raw/colombia_simci_coca_monitoring/coca_grid.geojson` (+ `SOURCE.txt`)
- `datasets/colombia_simci_coca_monitoring/metadata.json`
- `datasets/colombia_simci_coca_monitoring/locations/{000000..001577}.tif` + `.json`
- `datasets/colombia_simci_coca_monitoring/registry_entry.json` (status=completed)
