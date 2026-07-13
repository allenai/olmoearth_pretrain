# OSCD (Onera Satellite Change Detection)

- **Slug**: `oscd_onera_satellite_change_detection`
- **Status**: **completed** — task_type = **classification** (dense per-pixel, binary change), **1000 samples**
- **Family / region**: change_detection / global cities (24 cities, 6 continents)
- **Source**: OSCD — Caye Daudt, Le Saux, Boulch, Gousseau, *"Urban Change Detection for
  Multispectral Earth Observation Using Convolutional Neural Networks"*, IGARSS 2018.
  Project page https://rcdaudt.github.io/oscd/ .
- **License**: change maps **CC-BY-NC-SA 4.0** (research / non-commercial — fine for
  pretraining); imagery is modified Copernicus Sentinel-2 data 2015–2018.
- **Access** (no credentials):
  - Images: IMT mirror `https://partage.imt.fr/index.php/s/gKRaWgRnLMfwMGo/download`.
  - Train labels: `https://partage.mines-telecom.fr/index.php/s/2D6n03k58ygBSpu/download`
    (SSL cert name mismatch on that host → download with `curl -k`).
  - Test labels: **HuggingFace mirror `hkristen/oscd`** — the rcdaudt test-labels mirror
    (`partage.imt.fr/.../gpStKn4Mpgfnf63`) now returns **404**, so test labels came from
    `https://hf.co/datasets/hkristen/oscd/resolve/main/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Test%20Labels.zip`.

## What the dataset is

24 registered **bitemporal Sentinel-2 image pairs** over cities worldwide (official split:
14 train + 10 test; **we use all 24** per spec §5 — all splits are fair game as pretraining
labels). Each city has a manually photointerpreted **binary urban-change mask**
(`cm/<city>-cm.tif`, values **1 = no-change, 2 = change**; the `cm.png` shows 0/255). All
acquisitions are 2015–2018, entirely in the Sentinel era (nothing filtered on the pre-2016
rule).

## Georeferencing (spec §8.2 — the crux for OSCD)

OSCD is frequently distributed as coordinate-free chips: the popular `imgs_1_rect` /
`imgs_2_rect` band TIFs (10 m-resampled) are **CRS-stripped** (crs=None; torchgeo models
OSCD as a `NonGeoDataset`), and the change-map TIFs themselves carry **no** geotransform.
**However**, the ORIGINAL per-band crops in `imgs_1/<S2product>_Bxx.tif` **retain
georeferencing** — CRS **EPSG:4326** with a geotransform matching each city's true lon/lat.
Verified for **all 24 cities** that the change map's pixel grid is identical in size to the
`imgs_1` 10 m band (B04): e.g. abudhabi cm = imgs_1 B04 = 785×799. So the change map lives
on the `imgs_1` grid, and coordinates are fully recoverable. Each city was independently
re-verified: reprojected tile centers fall inside the city's own `<city>.geojson` bbox
(pisa 43.72 °N/10.39 °E, chongqing 29.40 °N/106.27 °E, mumbai, rennes, cupertino, …).
→ **Accept.**

## Class scheme (dense per-pixel classification)

| id | name      | definition |
|----|-----------|------------|
| 0  | no-change | OSCD change map value 1 — no urban change between the two acquisitions (background class) |
| 1  | change    | OSCD change map value 2 — urban change / new construction (manual photointerpretation) |
| 255| nodata    | geometric fill outside the rotated source footprint after reprojection to UTM |

Maps directly to the manifest's two classes. `no-change` (0) is the background class; both
are retained.

## Processing (label_type = dense_raster)

- Read CRS + transform from `imgs_1/*_B04.tif` (EPSG:4326), attach to the change map, remap
  `1→0` (no-change), `2→1` (change), then **reproject to local UTM at 10 m** with
  **nearest** resampling (categorical — never bilinear). UTM zone from the city center
  lon/lat via `get_utm_ups_projection`; dst grid snapped to the 10 m grid so pixel bounds
  are exact integers.
- Cut the UTM raster into non-overlapping **full 64×64** tiles (partial right/bottom edge
  tiles dropped); drop tiles that are > 50 % nodata.
- A tile counts toward **change** if it has ≥ 4 change px (~400 m²) and toward **no-change**
  if it has ≥ 64 no-change px. **Tiles-per-class balanced** (spec §5), ≤ 1000 tiles/class,
  rarer class (`change`) filled first.
- **Result**: 2225 candidate tiles → **1000 selected** (1000 contain change, 1000 contain
  no-change; because urban change is sparse, essentially every selected tile contains both
  classes). Well under the 25k cap. Per-city change-tile candidates ranged from ~8 (bercy)
  to ~146 (beirut).
- **Time / change** (spec §5): the change is an event between the pair's two acquisition
  dates (`dates.txt`). `change_time` = the **midpoint** of date_1/date_2; `time_range` = a
  **1-year window centered** on it (±182/183 days). `change_time` set on every sample.

## Verification (§9)

- 1000 `.tif` + 1000 matching `.json`. Every `.tif`: single-band **uint8**, **UTM**
  (EPSG:326xx/327xx) at 10 m, **64×64**, values ⊆ {0, 1, 255} with 255 declared nodata.
- Every `.json`: `time_range` = 365 days, `change_time` set. `metadata.json` class ids
  {0,1} cover all non-nodata pixel values.
- **Round-trip**: 8/8 sampled written tiles exactly equal the reprojected source block
  (array + integer pixel_bounds + CRS all match).
- **Spatial**: sampled tile centers fall inside each city's own bbox geojson.

## Judgment calls / caveats

- **Multi-year change interval.** OSCD pairs span ~1–2.7 years (e.g. abudhabi 2016-01 →
  2018-03). Urban change is therefore a **diffuse multi-year growth signal**, not a
  single-date event. The mandated 1-year window can't cover the whole interval; we anchor it
  on the midpoint as a representative window (pretraining uses the sample only when the input
  window spans `change_time`). This is *coarser* than the interval, not *finer* than a year,
  so it is not ill-posed in the §5 sense — kept, and flagged here.
- **Georeferencing precision.** Coordinates come from the `imgs_1` EPSG:4326 crops (produced
  by the OSCD Medusa crop + GeFolki registration pipeline), i.e. a resampled approximation of
  the native S2 UTM grid, so expect ~10–20 m (≈1–2 px) absolute registration slop. Fine for
  10 m pretraining co-location; noted for completeness. The `_rect` products and the
  cm-only TIFs are unusable for placement (no CRS) — only `imgs_1` works.
- Used **all 24 cities** (train + test); the official split is ignored per §5. `source_id`
  records the split + city + tile row/col.
- No fabricated negatives: because change is sparse, most tiles carry both classes; there are
  no "far-from-any-city" negative scenes (assembly adds cross-dataset negatives, §5).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.oscd_onera_satellite_change_detection
```

Raw inputs on weka `raw/oscd_onera_satellite_change_detection/`: the three zips plus their
extracted `Onera Satellite Change Detection dataset - {Images,Train Labels,Test Labels}/`
trees. Script is idempotent (skips already-written `{id}.tif`).
