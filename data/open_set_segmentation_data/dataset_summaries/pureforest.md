# PureForest

- **Slug:** `pureforest`
- **Task type:** classification (per-pixel tree species)
- **Status:** completed — **7,830** label tiles, 13 classes
- **Source:** [Hugging Face `IGNF/PureForest`](https://huggingface.co/datasets/IGNF/PureForest) (IGN France), paper arXiv:2404.12064
- **License:** Etalab Open Licence 2.0 (open, attribution)

## Source

PureForest is a tree-species dataset over **449 monospecific forests** in ~40 southern-French
departments. It ships 135,569 patches of **50 m × 50 m**, each a monospecific forest area
annotated with a **single tree species**. The classification has **13 semantic classes**
grouping **18 species** (with a broadleaf/needleleaf + genus hierarchy). Annotation polygons
were selected from the BD Forêt vector database and curated by IGN expert photointerpreters;
French National Forest Inventory ground truth was used to confirm stand purity.

The full release includes multi-GB VHR aerial imagery (0.2 m) and aerial Lidar zips. **We do
not download those** — only the label geometry + species are needed, which live in
`metadata/PureForest-patches.gpkg` (EPSG:2154 / Lambert-93; one 50 m square polygon per patch
with a `class_index` 0–12) and `metadata/PureForestID-dictionnary.csv` (class→species map).

## Access

`hf_download("IGNF/PureForest", <file>, raw_dir)` (public, unauthenticated). Raw metadata
lands in `raw/pureforest/metadata/`; `SOURCE.txt` records provenance.

## Label construction

Each patch is only 5×5 px at 10 m, so instead of writing thousands of tiny tiles we
**aggregate patches on a 320 m metric grid** (in Lambert-93, patches snapped by centroid).
Each occupied grid cell → one **≤32×32 UTM 10 m** tile centered on the cell center:
`rasterio.features.rasterize` burns each patch's `class_index` into its 50 m footprint
(`all_touched=True`); pixels outside any patch are **255 = nodata/ignore**. There is **no
background class** — unlabeled land is "ignore", not negative (assembly supplies negatives
from other datasets, per spec §5). 14,600 grid cells are occupied; cells are essentially
monospecific (only 6/5,530 at the 640 m scale, ~2 at 320 m, straddle two neighbouring
forests, which is a valid multi-class tile). Grid built from the WGS84-reprojected polygons
via the eurocrops `geom_to_pixels` + `rasterize_shapes` path.

**Grid-size choice (320 m / 32 px):** chosen over the 640 m / 64 px alternative because it
yields ~2.6× more tiles, denser labeled fill, and better rare-class coverage (e.g. Fir 89 vs
35 tiles). Even so, tiles are sparsely filled (median ~12 % labeled) since a monospecific
forest rarely tiles a full grid cell — this is genuine coverage, remaining pixels are ignore.

## Classes and sampling

Class ids are the dataset's native 0–12 (not re-derived). **Tiles-per-class balanced**
(`select_tiles_per_class`, per_class=1000, 25k cap). All 13 classes fit under the 254-class
uint8 cap, so none are dropped. Rare classes (Fir, Douglas, Larch, Spruce) are kept in full —
they are inherently rare; the downstream assembly step, not this script, filters classes that
end up too small.

Selected tiles per class (a tile counts toward every class it contains):

| id | class | tiles | id | class | tiles |
|----|-------|------:|----|-------|------:|
| 0 | Deciduous oak | 1000 | 7 | Black pine | 1000 |
| 1 | Evergreen oak | 1000 | 8 | Aleppo pine | 586 |
| 2 | Beech | 1000 | 9 | Fir | 89 |
| 3 | Chestnut | 451 | 10 | Spruce | 300 |
| 4 | Black locust | 376 | 11 | Larch | 268 |
| 5 | Maritime pine | 677 | 12 | Douglas | 85 |
| 6 | Scotch pine | 1000 | | **total** | **7,830** |

Per-class `description` in `metadata.json` lists the grouped Latin/English species and the
broadleaf/needleleaf + genus hierarchy.

## Time range

Tree species is a **static** label (a monospecific stand does not change species year to
year). Source acquisitions span 2018–2025, but **per-patch acquisition years are not present
in the released metadata files** (GPKG/CSV). Per spec §5 (static labels), every sample is
anchored on a single representative 1-year window in the Sentinel era: **2021-01-01 →
2022-01-01**. `change_time` is null.

## Verification

- 7,830 `.tif` each with a matching `.json`; all single-band uint8, UTM (e.g. EPSG:32631),
  10 m, 32×32, nodata 255, pixel values ∈ {0–12, 255}.
- Tile centers reproject back to southern France (lat ≈ 43–45 N, lon ≈ 1–5 E). ✓
- `metadata.json` class ids cover all values appearing in the tifs.

## Judgment calls

- Downloaded only the metadata GPKG/CSV, not the imagery/Lidar zips (labels are all we need).
- Aggregated 50 m patches on a 320 m grid into ≤32×32 tiles rather than writing 5×5 per-patch
  tiles (bigger context, fewer files, better rare-class balance).
- Fixed representative year 2021 because per-patch acquisition years are absent from metadata
  and the species label is static.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.pureforest
```
Idempotent (skips already-written `{sample_id}.tif`).
