# TPRoGI (Tibetan Plateau Rock Glacier Inventory)

- **Slug:** `tprogi_tibetan_plateau_rock_glacier_inventory`
- **Status:** completed
- **Task type:** classification (positive-only, single class)
- **Num samples:** 1000
- **Source:** Zenodo record 10732042 (Wang et al., "TPRoGI: a comprehensive rock glacier
  inventory for the Tibetan Plateau using deep learning", ESSD).
  https://doi.org/10.5281/zenodo.10732042
- **License:** CC-BY-4.0

## Source

Inventory of **44,273 rock glaciers** (~6,000 km²) across the Tibetan Plateau, compiled
with remote sensing + DeepLabv3+ and manual verification, following the IPA RGIK
guidelines v1.0 (RGIK, 2023). Two shapefiles in EPSG:4326:

- `TPRoGI_Extended_Footprint.shp` — 44,273 extended-outline (footprint) polygons of each
  rock glacier. **This is what we rasterize.** Attributes: `ID`, `SUBREGION`, `AREA`,
  elevation/slope/aspect stats, `MAAT`/`MAGT`/`AP`/`PISR` climate covariates, `MAP_DATE`.
- `TPRoGI_Primary_Marker.shp` — 44,273 primary-marker points (point location). **Not
  used** — the footprint layer already carries geometry + LAT/LON.

Downloaded via `download.download_zenodo("10732042", ...)` (footprint shapefile set +
README). `raw/<slug>/SOURCE.txt` records provenance.

## Access method

Public, no credentials. Zenodo HTTP.

## Class mapping

Unlike the RGIK RoGI precedent (which has active/transitional/relict activity
sub-classes), **TPRoGI has no activity classification** — the attribute table carries only
morphometric/climate covariates. So this is a **single-class, positive-only landform**:

| id | name         | meaning                                                        |
|----|--------------|----------------------------------------------------------------|
| 0  | rock_glacier | ice-rich/debris-mantled creeping permafrost landform footprint |

Per spec §5, no negatives are fabricated: pixels inside each footprint are class 0,
everything outside is **255 (nodata/ignore)**. The assembly step supplies negatives from
other datasets.

## Processing

- Each extended-footprint polygon rasterized (`rasterio.features.rasterize`,
  `all_touched=True`) into a **64×64 UTM 10 m** tile centered on the polygon's
  representative point. UTM zone picked per-sample from lon/lat.
- Median footprint ≈ 300 m across (fits the 640 m tile); the largest (~2.1 km) are clipped
  to the window, yielding a homogeneous interior tile — same behaviour as the RGIK
  precedent (which noted ~21% of outlines exceed 640 m).
- **Sampling:** single class, `balance_by_class(per_class=1000)` (seeded, deterministic) →
  **1000 of 44,273** footprints. Well under the 25k per-dataset cap. Sorted by
  `(cid, rg_id)` for stable sample ids. Re-running is idempotent (skips existing `.tif`).
- **Time range:** rock glaciers are slow, persistent landforms; `MAP_DATE` = Q3/Q4 2021.
  Uniform **2021** 1-year window (`change_time=null`).

## Outputs (on weka)

- `datasets/<slug>/metadata.json` — single-class scheme, `nodata_value=255`.
- `datasets/<slug>/locations/{000000..000999}.tif` — uint8, single band, UTM 10 m, 64×64,
  values {0, 255}.
- `datasets/<slug>/locations/{000000..000999}.json` — crs/pixel_bounds, 2021 time range,
  `source_id = SUBREGION/ID`.

## Verification

- 1000 `.tif` + 1000 `.json`. Spot-checked tifs: single-band uint8, EPSG:326xx at 10 m,
  64×64, only values 0 (rock glacier) and 255 (nodata).
- Every `.tif` has a matching `.json` with a 1-year (2021) `time_range`.
- Spatial sanity: tile centers reprojected to WGS84 land inside the plateau bbox
  (70–104°E, 27–40°N) and match the lat/lon encoded in the `RGU########N#########E` IDs to
  ~3 decimals (e.g. `RGU281527N0972426E` → 97.243°E, 28.153°N) — georeferencing exact.
  (A full Sentinel-2 image overlay was not rendered; CRS/zone + ID coordinate agreement is
  strong confirmation.)

## Caveats / judgment calls

- **Single class vs RGIK's three:** TPRoGI does not classify activity, so it contributes
  one class (`rock_glacier`), not active/transitional/relict. Kept as its own positive-only
  dataset rather than merged with RGIK RoGI (different regions, different class schemes).
- **1000 of 44,273:** followed the §5 "up to 1000 per class" rule and the RGIK precedent
  (`PER_CLASS=1000`). The full inventory is far larger but sampling 1000 keeps class-balance
  consistent with the rest of the corpus. Raise `PER_CLASS` if more presence tiles are
  desired later (idempotent re-run adds them).
- Used the Extended footprint only (TPRoGI has no Restricted outline layer, unlike RGIK).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.tprogi_tibetan_plateau_rock_glacier_inventory
```
(Downloads run separately via `download.download_zenodo("10732042", raw_dir, filenames=[...])`;
the script expects `raw/<slug>/TPRoGI_Extended_Footprint.shp`.)
