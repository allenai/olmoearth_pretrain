# Supraglacial Lakes & Channels, West Antarctica

- **Slug:** `supraglacial_lakes_channels_west_antarctica`
- **Status:** completed
- **Task type:** classification (positive-only, two-class polygon segmentation)
- **Samples:** 1,022 label tiles (64×64 @ 10 m)
- **Family / region:** glacier / West Antarctic Ice Sheet + Antarctic Peninsula

## Source

"Supraglacial lakes and channels in West Antarctica and Antarctic Peninsula during
January 2017" — Corr, D., Leeson, A., McMillan, M., Zhang, C. & Barnes, T., *Earth System
Science Data* (2022). Zenodo record **5642755** (DOI 10.5281/zenodo.5642755), license
**CC-BY-4.0**. A continent-scale inventory of ~10,500 supraglacial lakes and channels
delineated from January-2017 Landsat 8 and Sentinel-2 imagery (semi-automated
classification followed by manual post-processing).

### Access / download

The Zenodo record is dominated by ~190 raw Landsat-8 / Sentinel-2 **scene archives**
(`LC08_*.tar.gz`, `T*_*.tar.gz`), ~500 MB–1 GB each, **~130 TB total**. Pretraining
supplies its own imagery, so none of these are downloaded (this would be the
"impractical download volume for the label signal" case). **All label geometry lives in
one 17 MB file, `WAIS_Max_Extent.zip`**, containing `WAIS_Jan_2017_Polygons.shp` (10,478
features; also provided as GeoJSON and KMZ). The script downloads only that file via
`download_zenodo(..., filenames=["WAIS_Max_Extent.zip"])`. Total raw footprint on weka:
~60 MB.

## Label mapping / class scheme

Source CRS is Antarctic Polar Stereographic (PS_WGS84, lat_of_origin −71). Per-feature
attribute `Feature_Cl` ∈ {Lake, Channel}. Source counts: **10,223 Lakes, 255 Channels**.
Other attributes (`POLY_AREA`, `Location` = Ice Shelf / Grounded Ice / Crosses GL, REMA
elevation, ice speed, shape metrics) are retained only in provenance.

Class map (uint8):

| id  | name                    | source                     |
|-----|-------------------------|----------------------------|
| 0   | `supraglacial_lake`     | `Feature_Cl == "Lake"`     |
| 1   | `supraglacial_channel`  | `Feature_Cl == "Channel"`  |
| 255 | nodata / ignore         | every non-feature pixel    |

**Positive-only foreground (spec §5).** This is a two-foreground-class dataset with no
clean background/negative class; per the orchestrator's dataset-specific directive and
spec §5, non-feature pixels (surrounding ice, firn, snow, rock, unmapped area) are left as
**nodata/ignore (255)** — no synthetic negatives are fabricated. The pretraining-assembly
step supplies negatives by sampling other datasets. (This differs deliberately from the
sibling Hi-MAG glacial-lake dataset, which used a `background=0` class; surrounding
Antarctic ice/firn/cloud is a less clean negative than High-Mountain-Asia terrain, and the
directive here is positive-only.)

## Processing recipe

Polygon rasterization into local UTM/UPS 10 m tiles (spec §4 polygons, mirroring the
Hi-MAG glacial-lake script):

1. Each feature centroid → lon/lat → `get_utm_ups_projection` (UTM north of −80°, UPS
   [EPSG:5042] south of −80°) at 10 m, snapped to a 64-px (640 m) grid. Unique grid cells
   become candidate tiles (4,333 candidates).
2. Every lake/channel polygon intersecting a tile is rasterized into a 64×64 uint8 array:
   lake→0, channel→1, fill→255. **`all_touched=True`** so the smallest lakes (min ~96 m²
   ≈ 1 px @ 10 m) and thin channels stay visible at 10 m. Lakes are painted first, then
   channels on top so the rarer channel class wins at any adjacency. 4,332 tiles contain
   feature pixels.
3. **Tiles-per-class balanced** selection (`select_tiles_per_class`, rarest class first),
   ≤ 1000 tiles/class, ≤ 25k total → **1,022 tiles** selected.

Feature sizes: median lake bbox max-dim ~60 m; only ~2.7% of lakes and ~13.7% of channels
exceed 640 m. Large features are captured as a representative central 640 m window (their
centroid tile).

### Sample counts per class (tile-appearance)

| class                   | tiles containing it |
|-------------------------|---------------------|
| supraglacial_lake (0)   | 1,000               |
| supraglacial_channel (1)| 364                 |

Total distinct tiles: 1,022 (some tiles contain both classes). **Channels are sparse (255
source features) but all channel-containing tiles are retained** per spec §5 (rare classes
kept; downstream assembly drops too-small classes if needed).

## Time range / change handling

The inventory is a single January-2017 (austral-summer melt-peak) snapshot. Supraglacial
lakes/channels are seasonal, so this is treated as a seasonal/annual label: **1-year window
anchored on 2017** (`[2017-01-01, 2018-01-01)`, spec §5). `change_time = null` — it is a
single dated inventory, **not** a pre/post change label, so the change-timing rule does not
apply.

## Verification

- 1,022 `.tif` each with a matching `.json`. Sampled tiles: single-band, uint8, UTM/UPS at
  10 m, size 64×64, nodata 255, pixel values ∈ {0, 1, 255}. All sample JSONs have a 365-day
  `time_range` and `change_time = null`. `metadata.json` class ids {0,1} cover all
  non-nodata values in the tifs.
- **Georeferencing (rigorous):** 48/48 randomly sampled labeled pixels, reverse-geocoded
  from tile CRS back to the source Polar-Stereographic CRS, land within ≤ 15 m of a source
  polygon **of the same class** — confirming the PS → UTM/UPS → pixel pipeline and class
  assignment end-to-end.
- **Imagery eyeball:** attempted an overlay on a local Sentinel-2 L1C scene from the Zenodo
  record, but that scene (T20DNH_20170103) is nearly empty (1 tiny feature) and the true-
  color window is saturated white bright ice — not a usable visual. A dedicated eyeball is
  redundant here anyway: the source polygons are, by construction, delineated directly from
  the Jan-2017 Landsat/Sentinel-2 imagery, so class↔imagery correspondence is inherited
  from the source, and the reverse-geocode check independently validates georeferencing.
- Re-running the script is idempotent (skips already-written `{sample_id}.tif`).

## Caveats

- Positive-only labels: tiles are mostly nodata (255) with lake/channel pixels; this is
  intentional (§5) and negatives come from assembly.
- Channel class is sparse (255 source features, 364 tile-appearances); may be dropped by
  downstream min-count filtering.
- Large features (>640 m) are represented by a central window, not their full footprint.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.supraglacial_lakes_channels_west_antarctica
```
