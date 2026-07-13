# Global River Gravel Bars (Carbonneau & Bizzi)

- **Slug:** `global_river_gravel_bars_carbonneau_bizzi`
- **Status:** completed
- **Task type:** classification (per-pixel, dense_raster)
- **Num samples:** 3968 label patches (64×64, single-band uint8, local UTM @ 10 m)
- **Source:** Durham University Research Online, Carbonneau & Bizzi, a global 10 m
  Sentinel-2 semantic classification for **July 2021** produced with a fully-convolutional
  network + image processing. <https://researchdata.durham.ac.uk/files/r17w62f824x>
- **License:** CC-BY-NC-4.0.

## Source & access

The product ships as a single ~7.6 GB zip (`CarbonneauResearchData.zip`) containing **469
single-band uint8 GeoTIFF tiles**, one per MGRS grid zone (6° lon × 8° lat), covering
~89% of the non-polar globe. Each tile is already in its zone's **UTM CRS at 10 m**
(source nodata = 0). No credentials required.

**Access caveat — truncated download (bounded set: 246 of 469 tiles).** The Durham server
does **not** support HTTP range requests, and the bulk download truncated at ~3.84 GB (the
incomplete `raw/.../CarbonneauResearchData.zip.tmp` is retained). We sequentially extracted
the **246 complete tiles** that were fully present in the truncated archive and processed
those. Per spec §5 (large global derived-product rasters → **bounded-tile sampling**), a
representative bounded subset is exactly what is required — global coverage is not a goal.
A quick re-download retry was not attempted at length because the server's lack of range
support means it would restart from zero with the same truncation risk; the script is
tile-count agnostic, so dropping the full 469-tile archive under `raw/.../tiles/` and
re-running would simply scan the additional zones.

The 246 extracted tiles span **UTM zones 14–60** (27 distinct zone numbers) and **MGRS
latitude bands G–X** — i.e. both hemispheres from the deep southern mid-latitudes through
the tropics to high northern latitudes, covering Europe, Africa, Asia, Australia, and the
eastern Americas. The western Americas / eastern Pacific (UTM zones 1–13) fell outside the
truncated portion and are not represented (see caveats).

## Class mapping

Native source pixel codes → our 0-based class ids (native code − 1 for the six observable
phenomena); land/cloud/data-gap → nodata/ignore (255):

| native code | meaning              | class id | class name           |
|-------------|----------------------|----------|----------------------|
| 0           | land / background    | —        | 255 (nodata/ignore)  |
| 1           | river water          | 0        | river water          |
| 2           | lake water           | 1        | lake water           |
| 3           | sediment / gravel bar| 2        | sediment/gravel bar  |
| 4           | ocean                | 3        | ocean                |
| 5           | glaciated terrain    | 4        | glaciated terrain    |
| 6           | snow                 | 5        | snow                 |
| 7           | cloud                | —        | 255 (nodata/ignore)  |
| 8           | data gap             | —        | 255 (nodata/ignore)  |

There is **no land class** in the product (code 0 is "everything else"), so land is treated
as ignore; per spec §5 the assembly step supplies negatives from other datasets — we do not
fabricate synthetic negatives. **Sediment/gravel bar (id 2)** is the key fluvial class this
product uniquely adds.

## Sampling (bounded-tile, tiles-per-class balanced)

Because tiles are already local UTM at 10 m, each **64×64** block (640 m) is cropped
**natively** from its source tile — no reprojection, so georeferencing is exact and there is
no categorical-resampling loss. Each tile is scanned in non-overlapping 64×64 blocks
(2048-row parallel chunks, `multiprocessing.Pool(64)`).

**Presence rule** (whether a class "counts" toward a block for balancing):
- **Thin fluvial classes** (river id 0, gravel bar id 2): present if **≥ 40 px** in the
  block. Rivers and bars are narrow features surrounded by land, so a fraction/homogeneity
  gate would wrongly exclude the key classes.
- **Areal classes** (lake id 1, ocean id 3, glaciated id 4, snow id 5): present if
  **≥ 15%** of the 64×64 block (confident, spatially-coherent windows for a derived
  product).

Blocks are selected **tiles-per-class balanced, rarest class first**, up to **1000 tiles
per class** (25k total cap, not reached). A tile counts toward every class present in it,
so per-class totals slightly exceed 1000 where a selected rare-class tile also contains a
common class.

Scanned **678,542 candidate blocks** (per-class candidates: river 381,918; lake 222,800;
gravel bar 65,774; ocean 71,210; glaciated 7,264; snow 0).

**Time range:** 1-year window `[2021-01-01, 2022-01-01)` for every sample; `change_time` =
null. The product is a static July-2021 snapshot, not a dated change label.

### Selected class counts (tiles-per-class; a tile counts toward every class it contains)

| class | count |
|-------|-------|
| river water (0)          | 1017 |
| lake water (1)           | 1000 |
| sediment/gravel bar (2)  | 1010 |
| ocean (3)                | 1002 |
| glaciated terrain (4)    | 1001 |
| snow (5)                 | 0    |

Total distinct samples: **3968**.

## Verification (spec §9)

- Opened sample output `.tif`s: all **single-band, uint8, 64×64, UTM at 10 m**
  (e.g. EPSG:32753/32737/32659/32648/32758), nodata **255**. Dataset-wide unique pixel
  values across all 3968 tiles = `{0, 1, 2, 3, 4, 255}`, all covered by `metadata.json`
  classes (class 5 = snow legitimately absent — see caveats).
- 3968 `.tif` ↔ 3968 `.json`; every sidecar has a ≤1-year `time_range` (365 days) and
  `change_time` = null.
- **Georeferencing round-trip (spatial sanity):** for 6 random samples, re-read the exact
  source window (`source_id` = `<tile>.tif:<col>_<row>`), applied the code→id remap, and
  compared to the written tile — **6/6 exact match**. Because blocks are cropped natively in
  the source UTM CRS at 10 m (no reprojection), label placement is exact by construction.
- Script is idempotent (skips existing `{sample_id}.tif`).

## Caveats

- **246 of 469 tiles** processed (truncated download; no range-request support). Broad but
  not fully global: **UTM zones 14–60**, bands G–X, both hemispheres — Europe, Africa, Asia,
  Australia, and the eastern Americas; the **western Americas / eastern Pacific (zones 1–13)
  are not represented**.
- **Snow (class 5) has 0 samples**: this is a Northern-Hemisphere-**summer** (July 2021)
  product, so persistent snow is very rare. Kept in the class map for completeness; per spec
  §5 downstream filtering drops too-small classes.
- **No land/background class** — land (code 0) is ignore (255); negatives come from other
  datasets at assembly time.
- CC-BY-**NC**-4.0 (non-commercial) license.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_river_gravel_bars_carbonneau_bizzi --workers 64
```

Raw tiles: `raw/global_river_gravel_bars_carbonneau_bizzi/tiles/` (246 `Class_S2_*.tif`,
~4.7 GB; plus the retained truncated `CarbonneauResearchData.zip.tmp`). Outputs:
`datasets/global_river_gravel_bars_carbonneau_bizzi/{metadata.json, locations/*.tif+.json}`.
