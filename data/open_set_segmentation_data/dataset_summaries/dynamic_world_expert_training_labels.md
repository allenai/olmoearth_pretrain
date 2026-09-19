# Dynamic World Expert Training Labels

- **Slug:** `dynamic_world_expert_training_labels`
- **Status:** completed
- **Task type:** classification (dense land-use/land-cover segmentation)
- **Samples written:** 4,831 label patches (`locations/{id}.tif` + `.json`)
- **Label type:** `dense_raster` (reference data — human expert markup)

## Source

PANGAEA "Dynamic World training dataset for global land use and land cover categorization
of satellite imagery" (Tait, Brumby, Hyde, Mazzariello, Corcoran, 2021),
DOI [10.1594/PANGAEA.933475](https://doi.org/10.1594/PANGAEA.933475). Supplement to
Brown et al. 2022, *Nature Scientific Data* 9:251 (the Dynamic World near-real-time global
10 m LULC product). License **CC-BY-4.0** (attribution required — see `metadata.json`).

The full release has three folders (Experts, Non_expert crowd, validation holdout). This
dataset uses **only the `Experts` folder**: ~5.1 km tiles (510×510 px at 10 m) densely
labeled by a team of 25 expert human labelers recruited by National Geographic Society,
via visual interpretation of Sentinel-2 L2A true-color composites. This is the highest-
quality reference subset. The Non_expert crowd tiles and the validation holdout were
deliberately excluded.

## Access method

Direct HTTP, no credentials. Single files are served at
`https://download.pangaea.de/dataset/933475/files/{filename}` (the "download all as
ZIP/TAR" links require a PANGAEA account, but per-file access is open). The script pulls
`README.txt` and `Experts_tiles.zip` (~48 MB, 4,194 GeoTIFF tiles) into `raw/{slug}/` and
extracts. The per-tile metadata xlsx was **not** needed: each tile filename encodes lon/lat
and the Sentinel-2 acquisition date (`dw_<lon>_<lat>-<YYYYMMDD>.tif`), and the tiles carry
their own CRS/geotransform.

## Georeferencing

Every source tile is a single-band uint8 GeoTIFF already in a **local UTM projection at
10 m/pixel**, north-up, 510×510 px. No reprojection is required — windows are cropped
directly, preserving the source CRS and pixel grid exactly. Verified: back-projecting a
written window's pixel bounds to WGS84 lands inside the correct source tile footprint (see
verification below). Because the labels were drawn *on* georeferenced Sentinel-2 tiles and
we only crop (no resampling), label/imagery alignment is exact by construction.

## Class mapping

Dynamic World Tier-1 values → output class ids (uint8, nodata/ignore = 255):

| src value | output id | class name         |
|-----------|-----------|--------------------|
| 1         | 0         | Water              |
| 2         | 1         | Trees              |
| 3         | 2         | Grass              |
| 4         | 3         | Flooded vegetation |
| 5         | 4         | Crops              |
| 6         | 5         | Shrub & scrub      |
| 7         | 6         | Built area         |
| 8         | 7         | Bare ground        |
| 9         | 8         | Snow & ice         |
| 0 (unmarked / no data) | 255 | (ignore)      |
| 10 (cloud)             | 255 | (ignore)      |

Nine land-cover classes, matching the manifest. Per-class definitions (Dynamic World Tier-1
schema, Brown et al. 2022) are stored in `metadata.json` `classes[].description`.

## Processing recipe

- Each 510×510 source tile is cut into a grid of ≤64×64 windows (8×8 grid; the last row/col
  windows are 62 px, still ≤64). Full coverage, no reprojection.
- Source values 1..9 → ids 0..8; unmarked (0) and cloud (10) → 255.
- A window is kept only if **≥25%** of its pixels carry a real class (unmarked/cloud
  excluded), dropping near-empty windows (labelers leave unmarked regions).
- **Tiles-per-class balanced** selection (`sampling.select_tiles_per_class`, rarest class
  first): a window counts toward every class present in it, up to 1000 windows/class,
  25k total cap. Candidate pool was 230,785 windows.
- **Time range:** 1-year window on the tile's Sentinel-2 acquisition year (parsed from the
  filename). Dates span 2017–2019 (3,741 tiles 2019, 393 2018, 60 2017) — all Sentinel-era,
  so no pre-2016 filtering was needed. No change labels (static annual land cover).

## Sample counts per class (windows containing each class)

| id | class              | windows |
|----|--------------------|---------|
| 0  | Water              | 1,077 |
| 1  | Trees              | 1,576 |
| 2  | Grass              | 1,199 |
| 3  | Flooded vegetation | 1,111 |
| 4  | Crops              | 1,003 |
| 5  | Shrub & scrub      | 1,000 |
| 6  | Built area         | 1,186 |
| 7  | Bare ground        | 1,029 |
| 8  | Snow & ice         | 1,029 |

Total distinct windows: **4,831** (a window is multi-class, so it counts toward several
classes; balancing targets ~1000/class rarest-first, hence common classes like Trees exceed
1000 because they co-occur in windows selected for rarer classes).

## Verification (spec §9)

- 4,831 `.tif` each have a matching `.json`. All single-band uint8, UTM CRS at 10 m,
  size ≤64×64, pixel values all in {0..8, 255}. No out-of-range values.
- Every `.json` has a 1-year `time_range`, `change_time=null`, and `classes_present`
  consistent with the raster. `metadata.json` class ids cover all values in the tifs.
- Georeferencing back-projection: window centers for sampled ids fall within their source
  5.1 km tile footprint (Δ ≈ 0.01–0.06° at the tile scale; longitude offsets shrink at high
  latitude as expected). Alignment is exact by construction (crop-only, no resampling).
- Re-running is idempotent (skips existing `{id}.tif`; download/extract skip-existing).

## Caveats

- Uses the Experts subset only (not the larger Non_expert crowd set nor the validation
  holdout), by design — highest label quality.
- Labels are sparse *within* a source tile (expert markup does not cover every pixel);
  unmarked pixels are ignore (255). The ≥25%-labeled filter keeps windows meaningful; many
  windows are still partly ignore, which is correct for open-set segmentation.
- The per-tile metadata xlsx endpoint returned HTTP 503 during processing but was not
  required (filename carries lon/lat + date). If richer per-tile metadata (S2 product id,
  class percentages) is ever wanted, retry `.../files/v1_dw_tile_metadata_for_public_release.xlsx`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.dynamic_world_expert_training_labels
```
