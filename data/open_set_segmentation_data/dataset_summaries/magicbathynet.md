# MagicBathyNet

- **Slug:** `magicbathynet`
- **Task:** regression (shallow-water bathymetry / water depth)
- **Status:** completed
- **Samples:** 2857 label patches (Agia Napa 35, Puck Lagoon 2822)
- **Source:** Zenodo record 16753753 (Agrafiotis et al., IGARSS 2024),
  <https://zenodo.org/records/16753753>. arXiv:2405.15477.
- **License:** CC-BY-NC-4.0 (dataset). Non-commercial; used here for research
  pretraining. (Manifest listed "CC-BY"; the Zenodo page states Attribution-NonCommercial.)

## What the source is

MagicBathyNet is a multimodal benchmark for shallow-water bathymetry prediction and
pixel-based seabed classification over two coastal areas: **Agia Napa** (Cyprus,
Mediterranean) and **Puck Lagoon** (Poland, Baltic). It ships co-registered 180×180 m
image patches for Sentinel-2 (**18×18 px @ 10 m**), SPOT-6 (30×30 @ 6 m) and aerial
(720×720 @ 0.25 m), plus per-patch **DSM (depth) rasters** and seabed-class annotations.

We use the dataset for **bathymetry regression** (per-pixel water depth). The other
possible target (seabed habitat classification: sand / seagrass / rock / macroalgae for
Agia Napa; sand / eelgrass for Puck) is not processed here — the task brief scopes this
dataset as the per-pixel depth regression target, and a dataset writes either a
`classes` or a `regression` block, not both.

## Access / download

Public Zenodo download, no credentials. `MagicBathyNet.zip` (5.9 GB) is fetched to
`raw/magicbathynet/`; only the Sentinel-2 depth patches (`{area}/depth/s2/*.tif`) and the
`s2_split_bathymetry.txt` split files are extracted (the aerial 720×720 patches dominate
the archive size and are not needed). The 2.4 GB `..._extension_for_Swin-BathyUNet.zip` is
not downloaded.

## Why it fits (triage)

- **Georeferenced:** yes. The S2 depth patches are proper single-band GeoTIFFs already in
  a **local UTM projection at 10 m/pixel** (Agia Napa EPSG:32636 = WGS84 UTM 36N; Puck
  Lagoon EPSG:25834 = ETRS89 UTM 34N, <1 m from WGS84 UTM 34N).
- **Observable at 10–30 m:** yes — this is exactly a Sentinel-2 shallow-water-bathymetry
  benchmark. Depth is only defined in optically-shallow, clear water (roughly 0 to −30 m),
  which is what S2 can sense; deeper/turbid areas are masked out in the reference.
- **Post-2016:** yes. Sentinel-2 acquisition = **Agia Napa 2016-01-10**, **Puck Lagoon
  2021-04-20** (both Sentinel-era). Agia Napa's aerial/LiDAR reference is 2015, but depth
  is a static seabed quantity and the co-registered S2 image (what pretraining pairs
  against) is Jan 2016, so the label is anchored to the Sentinel era.

## Label construction

- **Target:** `water_depth` = per-pixel DSM elevation relative to the sea surface, in
  metres. **Negative = below water (deeper)**; small positive values at Puck Lagoon are
  emergent/near-shore land in the DSM. dtype float32, nodata `-99999`.
- **Nodata:** source uses `0.0` as the no-reference / masked fill (all real Agia Napa
  depths are negative); mapped to `-99999`.
- **Grid:** source patches are already UTM 10 m, so the source CRS is **reused** and the
  origin is snapped to the integer 10 m pixel grid (≤ half-pixel, ≤5 m shift). **No
  resampling** — written depth values are bit-identical to the source (verified). Output
  tiles are 18×18 (≤64).
- **Samples:** the annotated bathymetry split (`s2_split_bathymetry.txt`, train+test
  union) is used — 35 (Agia Napa) + 2822 (Puck Lagoon) = **2857**, comfortably under the
  5000-sample regression cap, so **all patches are kept** with no sub-sampling and no
  bucket balancing.
- **Time range:** 1-year window anchored on each area's S2 acquisition year (Agia Napa
  `[2016-01-01, 2017-01-01)`, Puck Lagoon `[2021-01-01, 2022-01-01)`). Depth is a static
  quantity, not a dated change event, so `change_time` is null.

## Statistics

- Per-pixel depth range: **[−23.68, 11.83] m**.
- Patch-mean depth percentiles: p5 = −6.19 m, p50 = −2.93 m, p95 = −0.79 m.
- Patch-mean depth histogram (m): [−20,−15): 5, [−15,−10): 9, [−10,−5): 476, [−5,0): 2367.
  (Agia Napa contributes the deeper tail to ~−24 m; Puck Lagoon is shallower.)

## Verification

- Opened several outputs: single-band float32, UTM CRS at 10 m, 18×18, nodata −99999,
  values within the declared range; each `.tif` has a matching `.json` with a 1-year
  `time_range` and `change_time=null`.
- Output values are bit-identical to the source depth rasters (with 0→nodata).
- Spatial sanity: valid-depth pixels coincide with low-reflectance water in the source
  co-registered S2 RGB patches; tile centres fall on the Agia Napa (Cyprus) and Puck
  Lagoon (Poland) coasts.

## Caveats

- Regression target mixes two water columns (clear Mediterranean vs turbid Baltic) and,
  at Puck Lagoon, a small fraction of positive land elevations from the DSM. This is
  faithful to the source; the sign convention (negative = underwater) is documented in
  `metadata.json`.
- Agia Napa depth reference dates to 2015 aerial/LiDAR; treated as static seabed anchored
  to the Sentinel-era S2 image (2016).
- License is CC-BY-**NC** (non-commercial).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.magicbathynet
```
(Idempotent: skips already-written `locations/{id}.tif`; re-extracts from the cached zip
if needed.)
