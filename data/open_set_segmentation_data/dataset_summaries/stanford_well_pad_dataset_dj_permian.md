# Stanford Well-Pad Dataset (DJ & Permian)

- **Slug:** `stanford_well_pad_dataset_dj_permian`
- **Status:** completed
- **Task type:** classification (object detection / polygon rasterization encoded as per-pixel classes)
- **Num samples:** 2000 (1000 well-pad positive tiles + 1000 background negative tiles)
- **Family / region:** energy / USA (Permian, Denver-Julesburg, plus additional US oil-gas chips)

## Source

- **URL:** https://github.com/stanfordmlgroup/well-pad-denver-permian
- **Paper:** "Deep learning for detecting and characterizing oil and gas well pads in
  satellite imagery" (Stanford ML Group; Nature Communications).
- **Annotation method:** manual/expert- and crowd-curated bounding boxes of oil/gas well
  pads and storage tanks over the Permian and Denver-Julesburg (DJ) basins.
- **License:** public GitHub research release ("check repo").

## Access method (label-only download)

Only the **label tables** are downloaded (imagery is supplied by pretraining):
- `data/training/datasets/well-pad_dataset.csv` — 88,044 image-chip rows; 10,432 chips
  contain ≥1 well pad (12,490 well-pad boxes total). Each row: `centroid_lat/lon`,
  `extent_image` (WKT POLYGON, the chip's lon/lat extent), `annotations_latlon` (list of
  `{"bbox": WKT POLYGON}` boxes in EPSG:4326), `split`, `basin`, `source`.
- `data/training/datasets/storage-tank_dataset.csv` — downloaded for provenance but
  **not used** (see below).

Chips are 640×640 Google-Earth-basemap images (EPSG:3857); each covers ~220 m on the
ground (~20–23 px at 10 m). Annotations cover the whole chip, so within a chip every well
pad is labeled and background pixels are true negatives. Rows with an empty annotation
list are true negatives.

## Class mapping

| id | name | notes |
|----|------|-------|
| 0 | background | Land within a fully-annotated chip with no well pad (true negative). |
| 1 | well_pad | Oil/gas well pad (cleared/graded pad w/ wellheads, tanks, access roads), typ. 30–200 m. |

- `nodata_value` = 255 (declared; not present in outputs, which contain only {0, 1}).

## Key decisions (spec §2–§5)

- **Observability / storage-tank drop.** Well-pad boxes have median max-dim ~89 m
  (5–95 pct: 28–181 m) ⇒ ~9 px at 10 m, clearly observable. Individual **storage tanks**
  in this dataset are ~4–6 m (median max-dim 4.7 m, <1 px at 10 m) ⇒ **not observable at
  10 m**; the storage-tank class is dropped and the dataset kept as a single foreground
  class (`well_pad`). Documented here per spec §4 (unresolvable-at-10 m class).
- **Recipe: polygons/boxes → polygon rasterization (spec §4).** Each well-pad box is
  rasterized (`all_touched=True`) as class 1 into the chip's own **local UTM, 10 m/pixel**
  tile (the chip extent, ~18–25 px square, well under the 64 cap); outside boxes =
  background (0). Since chips are fully annotated, background is a real negative, so we
  emit both **positive tiles** (≥1 well pad) and **background-only negative tiles**
  (detection exception, spec §5).
- **Tile size.** = chip extent in UTM pixels (~20–23 px). Verified that 100% of a chip's
  annotations fall within its extent, so no positives leak outside the tile.
- **Time range.** Well pads are persistent structures and the Google-basemap chips are
  undated mosaics; manifest range is 2016–2022. Every sample gets a **static
  representative 1-year window (2021-01-01 → 2022-01-01)**, `change_time = null` (spec §5
  static labels; post-2016). Caveat: individual pads may have been constructed at various
  dates within 2016–2022; 2021 is a representative persistence window.
- **Sampling.** Single foreground class ⇒ up to **1000** positive well-pad tiles +
  **1000** background negative tiles (well under the 25k cap), matching the
  turbine/vessel detection precedent. Selection is seeded (SEED=42) and idempotent.
  Negatives prefer in-basin (Permian/DJ) chips, then fall back to the broader
  hard-negative chips (`source=similarity/wind_turbine/...`). All source splits
  (train/valid/test) are used (splits are pretraining-agnostic, spec §5).
- **Geographic spread.** Positives are concentrated in the Permian/DJ basins but the
  training set also includes well-pad chips from other US oil/gas areas (lon −117…−100,
  lat 30…44); negatives (hard/similarity chips) span the continental US
  (lon −123…−71, lat 26…49). All labels carry exact source lon/lat, so georeferencing is
  reliable.

## Verification (spec §9)

- 2000 `.tif` + 2000 `.json`; all **single-band uint8**, **UTM (EPSG:326xx) @ 10 m**,
  sizes 18–25 px (≤64). Pixel values ∈ {0, 1}; 1000 tiles contain class 1 (= selected
  positives), all have matching sidecars.
- Every `.json` has a ≤1-year `time_range` (2021 window) and `change_time=null`;
  `metadata.json` class ids {0,1} cover all values in the tifs.
- Class balance: well_pad_positive_tiles=1000 (1217 boxes), background_negative_tiles=1000.
- Coordinate sanity: sample centroids fall in US oil/gas regions (see spread above). A
  live Sentinel-2 overlay was not run headlessly; georeferencing derives directly from the
  source's exact WGS84 lon/lat, so spatial alignment is trustworthy (basemap→S2 offset is
  ≲1 px, absorbed by the box footprint).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.stanford_well_pad_dataset_dj_permian
```

Outputs to
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/stanford_well_pad_dataset_dj_permian/`
(`metadata.json`, `locations/{id}.tif` + `.json`). Raw label CSVs are cached under
`raw/stanford_well_pad_dataset_dj_permian/`.
