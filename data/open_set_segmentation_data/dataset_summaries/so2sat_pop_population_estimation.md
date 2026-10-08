# So2Sat POP (Population Estimation)

- **Slug**: `so2sat_pop_population_estimation`
- **Status**: completed
- **Task type**: regression (population density, persons per square kilometre)
- **num_samples**: 5000 (bucket-balanced, hard cap for regression is 5000)
- **Source**: So2Sat POP Part1, Doda et al., *Sci Data* 9:715 (2022),
  doi:10.14459/2021mp1633792, TU Munich mediaTUM, CC-BY-4.0.
  Paper: https://www.nature.com/articles/s41597-022-01780-x ·
  Record: https://mediatum.ub.tum.de/1633792

## What the source is

A benchmark for population estimation over **98 European cities**. The population
reference is the EU **GEOSTAT 1 km population grid built from the 2011 census**
(EPSG:3035, ETRS89-LAEA Europe). Every 1×1 km grid cell carries an **absolute population
count** (persons living in that km²) and a **log2-binned population class**
(Class 0 = 0 persons; Class *c*≥1 means 2^(c-1) ≤ pop < 2^c, up to Class 16). The dataset
also ships, per cell, Sentinel-2 seasonal mosaics (2016), local climate zone, land-use
proportions, VIIRS nightlights, a DEM, and OSM data — but those are pretraining *inputs*
we do not need; we only extract the population **label**.

## Task decision: regression (not classification)

The source provides both a continuous count and a discrete class bin per cell. Population
is naturally a regression target and the continuous count is strictly more informative, so
we **regress the count**. Because each cell is exactly 1 km², the per-cell count *is* a
**population density in persons per km²** — a resolution-invariant intensity — so it can be
written to a tile of any size without a count/area rescale. This matches the unit/convention
of the sibling `worldpop_global_population_density`. The discrete log2 class is documented in
`metadata.json` for anyone who wants classification instead.

## Label → patch mapping

- **Coordinates (recoverable, label-only).** Populated cells use a GEOSTAT/INSPIRE LAEA grid
  id `1kmN{north_km}E{east_km}` giving the lower-left corner in EPSG:3035 (values in km).
  Cell centre = (`east_km`·1000+500, `north_km`·1000+500); reprojected 3035→WGS84 and placed
  on a **local UTM** grid. Verified: e.g. `budapest`→(18.9°E, 47.4°N), `london`→(−0.46°E,
  50.9°N), `berlin`→(13.4°E, 52.6°N) — all land in the correct cities.
- **Uninhabited filler cells skipped.** Cells *not* on the population grid carry a plain
  numeric id and POP=0 (no `1kmN...` name → no recoverable coordinates, and no population
  signal). They are dropped. Downstream assembly supplies negatives from other datasets
  (spec §5), so this is expected and correct.
- **Tiles.** 64×64 @ 10 m (~640 m) single-band **float32** GeoTIFFs, local UTM, north-up,
  nodata **−99999**. Each tile is filled *uniformly* with its cell's density (the product
  gives one value per 1 km cell; density is constant within a cell, so a centred 640 m
  sub-window loses no label information — the full 1 km footprint would be 100 px, over the
  64 px cap).

## Sampling / balancing

- 98 city CSVs → **100,144** populated, coordinate-bearing grid cells across all 98 cities
  (train+test splits both used; the source split is not filtered, per spec §5).
- Population is extremely right-skewed, so we **bucket-balance across log10(count) deciles**
  down to 5000 tiles (`sampling.bucket_balance_regression`, seed 42, deterministic).
- Value range of selected tiles: **[1.0, 48900.0] persons/km²**.
- Population-count bucket edges: [1, 10, 27, 58, 120, 250, 515, 1073, 2199, 4346, 53119].
- Selected-tile histogram (persons/km²):
  `[1,10)=500 · [10,50)=893 · [50,100)=501 · [100,500)=1085 · [500,1000)=475 ·`
  `[1000,5000)=1152 · [5000,10000)=264 · [10000,50000)=130`.
- All-cell population percentiles (persons/km²): p50=250, p90=4346, p99=15263, max=53119.

## Time-range and change handling

- **Time range = 1-year window at 2016** (`io.year_range(2016)`). The So2Sat POP
  Sentinel-2 mosaics are from 2016, so anchoring there gives the tightest label↔imagery
  alignment. `change_time = null` (not a change dataset).
- **On the 2011 census / pre-2016 rule.** The underlying population grid derives from the
  2011 census, but population is a **persistent/slowly-varying** attribute of built
  structure, and the dataset was explicitly assembled to pair with 2016 Sentinel-2 imagery.
  A post-2016 pairing window is usable (EU urban population distribution is stable across
  2011→2016+), so this is **not** a pre-2016 rejection — the label is not a dated
  pre-Sentinel observation but a static attribute observable in any Sentinel-era image. This
  is treated as a static label per spec §5. Caveat noted for downstream users.

## Access method (label-only; no imagery downloaded)

Part1 is distributed as a single **~103 GB `So2Sat_POP_Part1.zip`** on the mediaTUM
WebDAV/dataserv share (public creds `m1633792`/`m1633792`, also published for rsync — no
private credential needed). The per-city population CSVs (`{split}/{city}/{city}.csv`,
columns `GRD_ID,Class,POP`; **98 total**, a few KB–90 KB each) live inside that zip. We read
the zip's **central directory over HTTP Range requests** and extract only those 98 CSVs into
`raw/{slug}/city_csv/` — never touching the ~96 GB of imagery/aux patches. Reading the
1,117,287-entry central directory takes ~37 s / 4 range requests; extracting the 98 CSVs is
the slow part (~14 s each due to server per-request latency, ~20 min total, idempotent).

Note: mediaTUM mishandles a degenerate `bytes=0-0` probe by streaming the whole 103 GB file.
The shared `download.HttpRangeFile` was updated to **stream the size-probe response** (read
only headers, never the body), which fixes the hang and is strictly safer for all servers.

## Verification (spec §9)

- 5000 `.tif` + 5000 matching `.json`. Opened several: single-band `(1,64,64)`, float32,
  local UTM (EPSG:32630/32631/32632/32633/32634), 10 m resolution, nodata −99999, uniform
  positive values.
- Every `.json` has a 2016 calendar-year `time_range` (`change_time=null`); `metadata.json`
  regression block declares value range [1, 48900], unit persons/km², nodata −99999.
- Spatial sanity: tile centres reprojected back to WGS84 fall inside their named cities
  (Budapest/Paris/London/Berlin/Catania/Barcelona all matched known coordinates).
- Re-running the script is idempotent (deterministic seeded selection; existing tiles
  skipped; ~5 s no-op).

## Caveats

- Population is uniform per 1 km cell (single value), so tiles are constant-valued; the
  spatial detail is in the paired imagery, not the label.
- Only populated cells (POP>0 with a LAEA grid id) are placed; zero-population filler
  patches are unplaceable and omitted.
- Census epoch is 2011 while the pairing window is 2016 (persistent-label assumption).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.so2sat_pop_population_estimation --workers 64
```

Outputs:
- `/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/so2sat_pop_population_estimation/{metadata.json, registry_entry.json, locations/*.tif, locations/*.json}`
- `/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/so2sat_pop_population_estimation/{SOURCE.txt, city_csv/*.csv}`
