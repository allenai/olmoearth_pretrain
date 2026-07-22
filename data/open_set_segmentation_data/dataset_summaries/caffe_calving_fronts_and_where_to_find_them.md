# CaFFe (Calving Fronts and where to Find thEm)

- **slug**: `caffe_calving_fronts_and_where_to_find_them`
- **status**: completed
- **task_type**: classification (dense multi-class segmentation)
- **num_samples**: 1619 label tiles (64×64, UTM 10 m)
- **license**: CC-BY-4.0
- **source**: Gourmelon et al. 2022, *ESSD* 14, 4287–4313. PANGAEA record 940950
  (https://doi.pangaea.de/10.1594/PANGAEA.940950).

## What the source is
CaFFe is a benchmark of 681 preprocessed, geocoded, orthorectified SAR amplitude images of
7 marine-terminating glaciers (5 on the Antarctic Peninsula: Crane, Dinsmoor-Bombardier-
Edgeworth, Mapple, Jorum, Sjögren Inlet; Jakobshavn/Sermeq Kujalleq in Greenland; Columbia
in Alaska), acquired 1995–2020 by 7 SAR missions (Sentinel-1, TerraSAR-X, TanDEM-X, ERS-1/2,
Envisat, ALOS PALSAR, RADARSAT-1) at 7–20 m native resolution. Each image has two manually
annotated (expert) labels:
- **zones** (dense multi-class): grayscale PNG, values `0 = N/A` (SAR shadow/layover / no
  information), `64 = rock`, `127 = glacier`, `254 = ocean + ice mélange`. (Mapping confirmed
  from the CaFFe repo `data_postprocessing.py`: model class 1→64, 2→127, 3→254.)
- **fronts** (binary line): PNG, `255` = the calving-front line, `0` = background.

## The georeferencing problem (and the fix)
The PANGAEA release (`data_raw.zip`) ships **plain grayscale PNGs with no embedded geo tags**,
and its `bounding_boxes/*_front_extent_coord.txt` files hold only **pixel** coordinates (an
ROI around the dynamic front). On their own the PNGs have **no recoverable geocoordinates** —
which would normally be a rejection.

The rescue: the **torchgeo/caffe** Hugging Face mirror
(https://huggingface.co/datasets/torchgeo/caffe/resolve/main/meta_data.csv) adds a
`meta_data.csv` giving, for every image, its **projected bounding box + CRS**
(`EPSG:3031` Antarctic polar-stereographic for the 5 Peninsula glaciers; `EPSG:32606` UTM 6N
for Columbia; `EPSG:32622` UTM 22N for Jakobshavn). Verified: `bbox_width / png_width`
equals the stated native resolution **exactly**, so a north-up affine (origin = top-left,
res = bbox/px) georeferences every pixel. All 681 CSV rows join 1:1 to the PANGAEA PNGs by
image base name. Spot-checked reprojected tile centers land on the correct glaciers
(Columbia → -147.2°, 61.2°; Mapple → -62.2°, -65.4°).

## Processing
- Downloaded PANGAEA `data_raw.zip` (2.86 GB) + HF `meta_data.csv`; joined by base name.
- **Time filter (spec §5 / §8):** source spans 1995–2020; kept **only year ≥ 2016**
  (Sentinel era, so labels can be co-located with S2/S1/Landsat). **52 of 681** images remain
  (Columbia 22, Jorum 15, Mapple 15; 47 Sentinel-1 @20 m + 5 TanDEM-X @7 m). 629 pre-2016
  images dropped. Jakobshavn has no ≥2016 image, so Greenland/EPSG:32622 does not appear.
- Per image: remap zones → unified class ids, reproject label to **local UTM 10 m**
  (`rasterio.warp.reproject`, **nearest** — categorical) onto the rslearn pixel grid, warp the
  front mask separately, **dilate the front to ~3 px (~30 m)** and overlay it as its own class
  on top of the zones (only where observed). Tile the reprojected label into ≤64×64 patches;
  keep tiles with ≥64 observed (non-nodata) pixels.
- **Unified class map** (spec §5 "combine multi-target sources into ONE class scheme"):
  `0 = ocean_and_ice_melange`, `1 = glacier`, `2 = rock`, `3 = calving_front`,
  `255 = nodata` (CaFFe N/A zone + unobserved after warp).
- **Balancing:** tiles-per-class balanced at 1000/class, prioritizing rare classes
  (`sampling.balance_tiles_by_class`, added to the shared module). 34,895 candidate tiles →
  **1619 selected**. Candidates sorted deterministically before the seeded selection so the
  run is reproducible/idempotent.
- **Time range:** 1-year window **centered on the acquisition date** of each source image.

## Output stats
- Selected tiles per class (a tile counts toward every class it contains):
  ocean+ice_mélange **1032**, glacier **1288**, rock **1000**, calving_front **1002**.
- Tiles per glacier: Columbia 1049, Jorum 385, Mapple 176 (approx; balancing over-samples
  Columbia because it has the most ≥2016 imagery and all classes).
- Tiles per source year: 2016: 444, 2017: 243, 2018: 350, 2019: 268, 2020: 305.

## Verification (spec §9)
- 1619 `.tif` + 1619 `.json`, all paired. Tiles are single-band `uint8`, 64×64, UTM at 10 m
  (`EPSG:32606`, `EPSG:32720`, …), nodata 255, pixel values ⊆ {0,1,2,3}.
- All sampled sample JSONs have ≤1-year `time_range`; `metadata.json` class ids cover all tif
  values.
- Spatial sanity: reprojected tile lon/lat land on the correct glaciers (see above).
- Re-running the script is idempotent (1619 selected, all skipped).

## Judgment calls / caveats
- **Georeferencing depends on the torchgeo HF `meta_data.csv`.** Without it the dataset would
  be rejected (no recoverable geocoordinates). The join and bbox→resolution consistency were
  validated for all 681 images.
- **Only 52/681 images are Sentinel-era**, so the effective dataset is small (3 glaciers) and
  Antarctic Peninsula glaciers dominate the raw pool but are down-weighted by class balancing;
  Columbia (Alaska) contributes the most tiles.
- **Calving fronts shift seasonally**, so the 1-year window is an approximation for the
  `calving_front` class specifically; the glacier/rock/ocean zones are more temporally stable.
  The window is centered on the acquisition date to minimize mismatch.
- The `ocean` class merges open ocean and **ice mélange** (per the source's zone definition),
  which is what the manifest's "ocean + ice mélange" class denotes.
- Positive-only-style semantics are respected (no fabricated negatives); N/A regions are left
  as nodata (255).

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.caffe_calving_fronts_and_where_to_find_them
```
Downloads `data_raw.zip` (PANGAEA) + `meta_data.csv` (torchgeo HF) to
`raw/caffe_calving_fronts_and_where_to_find_them/`, writes tiles to
`datasets/caffe_calving_fronts_and_where_to_find_them/locations/`.
