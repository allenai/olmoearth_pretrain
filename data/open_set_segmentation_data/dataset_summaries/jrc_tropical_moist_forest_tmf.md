# JRC Tropical Moist Forest (TMF)

- **Slug**: `jrc_tropical_moist_forest_tmf`
- **Status**: completed
- **Task type**: classification (dense_raster, derived-product map)
- **Samples**: 6000 (1000 / class, all 6 classes reached the cap)
- **Source**: EC JRC Tropical Moist Forest product — <https://forobs.jrc.ec.europa.eu/TMF/data>
- **Citation**: Vancutsem et al. 2021, *Science Advances*, doi:10.1126/sciadv.abe1603
- **License**: free with attribution ("No limitations on use").

## What the source is

The JRC TMF is a pan-tropical, 30 m, Landsat-derived product tracking the state and
disturbance history of tropical moist forests (1990–2025). We use the **AnnualChange**
collection: one raster per year giving the per-pixel forest state for that year. The
AnnualChange pixel legend maps exactly onto the manifest classes:

| src value | class id | class name          |
|-----------|----------|---------------------|
| 1         | 0        | undisturbed forest  |
| 2         | 1        | degraded forest     |
| 3         | 2        | deforested          |
| 4         | 3        | regrowth            |
| 5         | 4        | water               |
| 6         | 5        | other               |
| 0         | 255      | nodata / outside product |

The product is distributed as 10°×10° tiles in EPSG:4326. This is a huge global product,
so per the spec we do **bounded-tile sampling** — download a handful of representative
tiles for one year, not the whole product.

## Access method (download mechanism)

The `/TMF/data` page is a Vue SPA; the tile-download URL builder was recovered from its JS
bundle (`/dist/assets/index-*.js`). Direct HTTP tile URLs (no auth):

```
https://ies-ows.jrc.ec.europa.eu/iforce/tmf_v1/download.py?type=tile&dataset=AnnualChange_<year>&lat=<latLabel>&lon=<lonLabel>
```

where `<latLabel>_<lonLabel>` is the 10°×10° tile id (NW-corner label, e.g. `N0_E20`,
`S10_W60`; longitude/latitude labels are **not** zero-padded). Downloaded via
`download.download_http` (atomic, idempotent). Each tile is ~60–90 MB (uint8 BigTIFF).

## Regions / tiles sampled (year 2020)

Six tiles across the three tropical moist-forest basins:

| tile id  | region |
|----------|--------|
| S10_W60  | Amazon — S Brazil / Rondônia (heavy deforestation, degradation, regrowth) |
| S10_W70  | Amazon — W Brazil / Peru / Bolivia |
| N0_E20   | Congo Basin — DR Congo |
| N0_E10   | Congo Basin — Gabon / Cameroon |
| N0_E110  | SE Asia — Borneo (Kalimantan) |
| N0_E100  | SE Asia — Sumatra / Malay Peninsula |

Raw tiles kept at
`raw/jrc_tropical_moist_forest_tmf/JRC_TMF_AnnualChange_2020_<tile>.tif` (438 MB total).

## Sampling method

- For each source tile, scanned every non-overlapping **22×22 native-pixel** block
  (≈660 m ≈ one 64×64 @ 10 m UTM tile footprint) and computed the per-block class
  histogram (vectorised).
- A block qualifies as a **homogeneous window** if a single class is ≥ **50 %** of the
  block (that class becomes its label) and the nodata fraction is ≤ 20 %. 15.9 M candidate
  windows qualified across the 6 tiles.
- `sampling.balance_by_class` then took a seeded random subsample of ≤ **1000 per class**.
  All six classes reached 1000.
- Each selected window's centre lon/lat is reprojected from native 30 m EPSG:4326 into a
  **local UTM** projection at **10 m** using **nearest** resampling (categorical labels),
  producing a **64×64 uint8** patch. Source values 1–6 → class ids 0–5; source 0 and
  out-of-coverage → 255 (nodata).

Because the floor is a 50 % majority (not purity), some patches contain minority pixels of
other classes (their `classes_present` lists them); the tile's label is the dominant class.

## Time range / change handling

The AnnualChange raster is a clean **per-year state** map, so — as the spec directs — it is
treated as a plain **classification** label with a **1-year** time range anchored on the
chosen year (`2020-01-01 .. 2021-01-01`). `change_time` is `null` (no per-event dating).
The optional per-event `change_time` scheme was not used: the annual-state layer already
gives a well-posed yearly classification and does not need finer temporal precision.

## Output spec

Single-band uint8, local UTM, 10 m, north-up, 64×64. nodata = 255. Verified on random
samples: correct CRS (EPSG:326xx/327xx), 10 m resolution, ≤64×64, values ∈ {0..5, 255},
matching `.json` sidecars with ≤1-year `time_range`.

## Verification

- `metadata.json` class ids (0–5) cover all values appearing in the tifs; 6000 `.tif` and
  6000 `.json`, class_counts = 1000 each.
- **Spatial sanity check**: for 8 random samples, reprojected the patch centre back to
  lon/lat and read the JRC source value there — 7/8 matched the patch's dominant class
  exactly; the 1 that differed was a majority-undisturbed window whose single centre pixel
  was degraded (expected under a 50 % majority floor, not a georeferencing error).
  Sample coordinates land correctly in Borneo (~113°E), Amazon (~-54°W), Congo (~11–28°E)
  and Sumatra (~103°E).

## Caveats

- Degraded forest and regrowth are spatially diffuse transition classes; at a strict
  (≥85 %) purity floor they are scarce, so a 50 % majority floor was used to reach 1000/
  class. Their patches are therefore more mixed than the undisturbed/other patches.
- Only year 2020 and 6 tiles were sampled — a bounded, representative slice of a pan-tropical
  product, not global coverage. More years/tiles could be added by extending `TILES` / `YEAR`.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.jrc_tropical_moist_forest_tmf
```

Idempotent: re-running skips existing raw tiles and existing `locations/{id}.tif`.
(Registry not modified per task instruction.)
