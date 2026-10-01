# AI4Arctic / ASIP Sea Ice Dataset — sea-ice concentration (regression)

- **slug**: `ai4arctic_asip_sea_ice_dataset`
- **task_type**: regression
- **num_samples**: 5000
- **source**: Technical University of Denmark (DTU) Data / figshare record **13011134**,
  "AI4Arctic / ASIP Sea Ice Dataset - version 2" (ASID-v2).
  https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134
- **license**: CC-BY.

## What the source is

ASID-v2 is a benchmark of **452 NetCDF scenes** (~333 GB) over the waters around Greenland,
2018–2019. Each scene pairs a **Sentinel-1 EW SAR** acquisition (HH/HV, ~40 m, in native
swath geometry) with the **operational Greenland ice chart** manually drawn by ice analysts
at the Danish Meteorological Institute (DMI), gridded onto the SAR grid, plus **AMSR2**
passive-microwave brightness temperatures. It is the basis of the AI4Arctic / AutoICE sea-ice
mapping challenge.

The ice chart is stored as `polygon_icechart` (a raster of polygon ids) + `polygon_codes` (a
per-polygon **SIGRID-3** attribute table:
`id;CT;CA;SA;FA;CB;SB;FB;CC;SC;FC;CN;CD;CF;POLY_TYPE`). `CT` is the **total sea-ice
concentration** of the polygon; `SA/SB/SC` are stage-of-development (ice type) and `FA/FB/FC`
are floe/form, each with partial concentrations `CA/CB/CC`.

## Label choice — SIC concentration REGRESSION

We produce **per-pixel total sea-ice concentration (SIC), 0–100 %**, as a **regression**
target. This is the recommended primary AI4Arctic label because it maps **unambiguously from
the single `CT` field** of each polygon, with no partial-concentration bookkeeping:

| SIGRID-3 `CT` | meaning | value |
|---|---|---|
| `00` / `01` / `02` | ice-free / <1/10 / bergy water | `0` |
| `10`…`90` (`k0`) | k/10 | `k*10` |
| `91` | 9+/10 … <10/10 | `95` |
| `92` | 10/10 (incl. fast / compact ice) | `100` |
| `ab` (a≤b, e.g. `46`) | range 4/10…6/10 | midpoint → `50` |
| `99` / negative / undetermined | unknown | nodata |

**Stage-of-development (SOD, ice type)** and **form/floe (FLOE)** classifications are also
recoverable from the same polygon codes (they require weighting the partial concentrations
`CA/SA`, `CB/SB`, `CC/SC`) and could be produced later as a classification companion dataset;
SIC was chosen as the single cleanest primary product per the task guidance.

Output tiles are single-band **float32**, local UTM, **10 m/pixel**, **64×64**, nodata
**-99999** (`io.REGRESSION_NODATA`). Observed value range across all tiles: **[0.0, 100.0] %**.

## Resolution / label-generalization caveat (important)

Ice charts are **manually drawn generalized polygons** covering large marine areas; the
**effective native resolution is coarse (kilometre-scale)**, not the SAR pixel. The "10 m"
label here is therefore a **coarse polygon field upsampled to 10 m** (nearest resampling),
not a fine per-pixel measurement — treat it as a smooth regional concentration field. This is
recorded in `metadata.json` (`regression.description` + `notes`).

## Georeferencing

The ice chart lives in **SAR swath geometry**, not a regular map grid. Each NetCDF carries a
coarse geolocation grid (`sar_grid_line`/`sar_grid_sample` → `sar_grid_latitude`/
`sar_grid_longitude`). We build **GCPs** from that grid and warp the concentration raster to a
**scene-local UTM** projection (zone from the scene-mean lon/lat) at 10 m with `rasterio`
`reproject` + **nearest** resampling (categorical polygon field), then tile into 64×64 patches.
Output CRSs are northern UTM zones (EPSG:326xx). Tile centroids fall in Greenland marine
waters (~66–75°N, ~26–61°W); unobserved / land pixels (polygon id 0) stay nodata and are not
sampled, so tiles are marine.

## Access / download (no imagery, no bulk download)

The full 333 GB archive bundles SAR + AMSR2 imagery, which we do **not** need — pretraining
supplies its own imagery. NetCDF4 is HDF5 and the ice-chart variable is gzip-compressed to
~30 KB, so for a bounded scene sample we open each remote NetCDF over **HTTP Range requests**
(`download.HttpRangeFile` + `h5py`) and read **only** the label vars (`polygon_icechart`,
`polygon_codes`) and the geolocation grid — **~60 KB per scene instead of ~500 MB**. Extracted
arrays are cached to a small per-scene `.npz` in `raw/{slug}/` so re-runs are offline and
idempotent. (If figshare listing is unreachable, the script falls back to the cached `.npz`.)

## Scene sampling & coverage

Bounded, **month × region stratified** sample (smallest file per month/region cell first) →
**36 scenes** used, spanning:
- **Years**: 2018 (4193 tiles), 2019 (807) — all post-2016 (Sentinel era).
- **Months**: all 12 represented (winter freeze-up, melt, and summer minimum), e.g. Jan 281,
  Feb 181 … Jun 834, Jul 740, Aug 943 … Dec 231.
- **Regions**: all 9 Greenland ice-chart regions (NorthWest 1159, CentralEast 860, SouthEast
  798, CentralWest, NorthEast, Qaanaaq, CapeFarewell, SouthWest, NorthAndCentralEast).

## Tile sampling (regression)

216,000 candidate 64×64 tiles were scanned (tiles >50 % nodata dropped). Because the SIC
distribution is strongly **bimodal** (open water 0 % and compact pack ice 100 % dominate,
intermediate concentrations rarer), tiles are **fixed-bucket balanced** across the tile
**mean concentration** into 10 buckets `[0,10,20,…,100]` → **exactly 500 tiles per bucket**
(5000 total), giving an even spread of concentration levels rather than a corpus dominated by
0 % / 100 %.

## Time range & change handling

Sea ice is **dynamic**, so each label is treated as **state-at-time** (spec §5): `change_time`
is **null** and `time_range` is a **tight ±3-day window (6 days total)** centered on the SAR
acquisition timestamp parsed from the filename (e.g. `20190519T194908`). This is far under the
360-day cap and short enough that the regional concentration field is roughly stable, while
Arctic S1/S2 revisit (very frequent at high latitude) still gives ample pairing opportunities.

## Metadata / value semantics

`metadata.json`: `task_type=regression`; `regression = {name: "sea_ice_concentration",
unit: "percent", dtype: "float32", value_range: [0.0, 100.0], nodata_value: -99999,
buckets: [0,10,…,100]}`; `num_samples=5000`; `sensors_relevant=[sentinel1, sentinel2]`.

## Verification (spec §9)

- 5000 `.tif` + 5000 `.json`, contiguous ids `000000`…`004999`.
- All tiles single-band **float32**, **64×64**, resolution (10, −10) north-up, **UTM** CRS
  (EPSG:326xx), nodata −99999. All valid pixel values within **[0, 100]**; 0 out-of-range or
  empty tiles.
- Each `.json` has `change_time=null` and a **6-day** `time_range` (≤1 year).
- `metadata.json` value_range [0.0, 100.0] matches the tiles.
- Centroids verified in Greenland/Arctic marine waters (66–75°N). A full Sentinel-2 overlay
  was not run (high-latitude optical scenes are frequently cloud/polar-night limited); the
  georeferencing rests on the exact per-scene GCP geolocation grid.
- Re-running is idempotent: existing `locations/{id}.tif` are skipped; the `.npz` label cache
  makes the download step offline.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ai4arctic_asip_sea_ice_dataset
```

## Caveats

- Labels are **generalized analyst-drawn ice-chart polygons** — coarse effective resolution
  upsampled to 10 m (see caveat above); not a fine per-pixel product.
- Concentration is derived only from the polygon `CT` code (deciles + range midpoints), so
  values are quantized to 0/10/…/95/100 (plus range midpoints like 50), not a continuous
  retrieval.
- Coverage is Greenland waters only (the ASID-v2 charts are DMI/Greenland); no Canadian-Arctic
  scenes despite the manifest region hint.
