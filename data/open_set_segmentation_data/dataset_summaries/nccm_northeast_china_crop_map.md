# NCCM (Northeast China Crop Map)

- **Slug**: `nccm_northeast_china_crop_map`
- **Task type**: classification (dense_raster)
- **Status**: completed
- **Samples**: 2,402 label patches (64×64, uint8, local UTM @ 10 m)
- **Classes**: 4 (maize, soybean, rice, other)

## Source

You, N., Dong, J., et al. (2021), *"The 10-m crop type maps in Northeast China during
2017–2019"*, Scientific Data (<https://doi.org/10.1038/s41597-021-00827-9>). Data DOI
<https://doi.org/10.5281/zenodo.8175171>; also distributed as the TorchGeo `NCCM` dataset.
License: **CC-BY-4.0**.

Annual 10 m crop-type **maps** of Northeast China for 2017, 2018 and 2019, produced by
hierarchical **random-forest** classification of interpolated/smoothed 10-day Sentinel-2
time series (spectral + temporal + textural features), validated against ground-truth
reference samples. `annotation_method`: derived-product map with field/reference validation —
so per spec §4 (dense_raster) / §5 (derived-product) it is sampled at
high-confidence/homogeneous windows and tiles-per-class balanced.

## Access (cached raw — not re-downloaded)

The three source GeoTIFFs were already cached on weka by a previous (interrupted) run and
were **re-used as-is** (no re-download):

```
raw/nccm_northeast_china_crop_map/CDL2017_clip.tif   (~800 MB)
raw/nccm_northeast_china_crop_map/CDL2018_clip1.tif  (~793 MB)
raw/nccm_northeast_china_crop_map/CDL2019_clip.tif   (~788 MB)
```

Each is single-band **uint8, EPSG:4326** at ~8.983e-5°/px (~10 m), 216,985 × 164,926 px,
covering lon 115.48–134.98 E, lat 38.72–53.53 N (Northeast China). Native fill/nodata = 15.
(TorchGeo's canonical download URLs are the figshare files behind the same Zenodo record;
re-running the script only reads these cached files.)

## Class mapping

Native NCCM codes → compact uint8 ids (aligned with the manifest class order; each class's
`native_code` is recorded in `metadata.json`):

| id | name | native code | notes |
|---|---|---|---|
| 0 | maize | 1 | maize (corn) cropland |
| 1 | soybean | 2 | soybean cropland |
| 2 | rice | 0 | paddy rice cropland |
| 3 | other | 3 | "others crops and lands" — the product's residual class (all non-maize/soy/rice crops + non-cropland) |
| — | nodata | 15 | mapped to 255 (ignore) |

`other` (code 3) is the product's catch-all residual class and is present in the great
majority of windows; **rice is the rarest crop** (~0.05 % of pixels overall, concentrated in
the Sanjiang/Liaohe plains).

## Processing

- **Scan**: each year's raster is scanned in parallel over 4096×4096 native super-windows
  (6,519 tasks total; no overviews, so windowed reads decompress only the needed LZW tiles —
  full parallel scan ≈ 20 s on 64 workers). Each super-window is subdivided into 64×64 native
  blocks (~one 640 m UTM-tile footprint). A block becomes a candidate only if it is
  **well-observed** (≥ 70 % of pixels not nodata) and a class covers **≥ 25 % of the valid
  pixels** (high-confidence/homogeneous preference, spec §4). A per-task per-class cap
  (16) bounds candidate memory. 143,599 candidate windows were produced
  (maize 50,926 / soybean 46,686 / rice 25,264 / other 104,841 windows containing each class).
- **Selection**: tiles-per-class balanced, **rarest class first**, up to 1000 tiles/class,
  under the 25k per-dataset cap (`select_tiles_per_class`). A tile counts toward every class
  present in it. → **2,402** windows selected.
- **Write**: each selected native block is reprojected from EPSG:4326 to a **local UTM**
  projection at **10 m** with **nearest** resampling (categorical labels) into a 64×64 uint8
  tile; native codes are remapped via LUT to compact ids, nodata → 255. Written atomically
  with rslearn `GeotiffRasterFormat` so georeferencing is exact.
- **Time range**: 1-year window anchored on each map's year (2017 → [2017-01-01, 2018-01-01),
  etc.). No change labels (`change_time=null`). Static/annual crop-type maps, spec §5.

## Sample counts

- **Total**: 2,402 patches. **By year**: 2017 = 825, 2018 = 813, 2019 = 764.
- **Tiles per class** (a tile counts toward every class it contains):
  maize = 1000, soybean = 1147, rice = 1090, other = 1308.
- All four classes exceed 1000 tiles; none is sparse, so no downstream rare-class filtering
  is expected to drop any class.

## Verification (spec §9)

- Opened output tifs: all single-band **uint8**, **64×64**, local **UTM @ 10 m**
  (e.g. EPSG:32651 = UTM 51N, correct for ~120–126 E NE China), nodata **255**.
- Pixel values are valid class ids **{0,1,2,3}** plus 255 (ignore); a small nodata fraction
  (~0.15 % of pixels) arises at reprojection edges / native-fill pixels — expected.
- Every `.tif` has a matching `.json` with a ≤1-year `time_range`; `metadata.json` class ids
  (0–3) cover all values appearing in the tifs.
- CRS/bounds are derived exactly from each tile's lon/lat via rslearn UTM projection; a live
  Sentinel-2 overlay was **not** rendered (labels are a validated derived product and
  georeferencing is exact by construction), so no misalignment was observed.
- Re-running is **idempotent**: `_write_one` skips any `{sample_id}.tif` already present.

## Caveats

- Derived-product **map** (not in-situ points): mitigated by the high-confidence/homogeneous
  window filter (≥70 % observed, dominant class ≥25 %).
- `other` (code 3) is a heterogeneous residual class (mixed non-target crops + non-cropland);
  treat it as a coarse background-ish class rather than a semantically pure crop type.
- Native grid is EPSG:4326 (~10 m); a 64×64 UTM tile (640 m) draws from ~64 × ~90 native px,
  so candidate composition is computed on a slightly lon-narrower native block than the final
  UTM footprint — an approximation used only for filtering; the written tile reprojects the
  correct footprint.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.nccm_northeast_china_crop_map
```

Reads the three cached rasters under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/nccm_northeast_china_crop_map/`
and writes `datasets/nccm_northeast_china_crop_map/{metadata.json, locations/*.tif+*.json}`.
