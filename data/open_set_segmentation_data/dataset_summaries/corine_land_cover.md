# CORINE Land Cover (CLC2018) — `corine_land_cover`

- **Status:** completed
- **Task type:** classification (dense multi-class land cover)
- **Label type:** polygons / dense_raster (accessed as the 100 m raster)
- **Num samples:** 24,373 label patches (64×64, uint8, local UTM @ 10 m)
- **Classes:** 44 (full CLC level-3 nomenclature), ids 0–43, nodata = 255

## Source

CORINE Land Cover 2018 (CLC2018), version **V2020_20u1** (product
`U2018_CLC2018_V2020_20u1`) — the EEA / Copernicus Land Monitoring Service pan-European
land-cover/land-use inventory. It is a **photointerpreted (visual)** derived map produced
from Sentinel-2 / Sentinel-1 imagery, distributed as a 100 m raster in EPSG:3035 with a
25 ha minimum mapping unit (MMU) and a hierarchical 3-level nomenclature whose level 3 has
44 thematic classes (grid codes `111`…`523`). Covers ~39 EEA countries.

- URL: https://land.copernicus.eu/en/products/corine-land-cover
- DOI (raster 100 m): 10.2909/960998c1-1870-4e82-8051-6485205ebbac
- License: Copernicus open (free access, full reuse with attribution).

## Access method (and why)

The authoritative full-coverage download from `land.copernicus.eu` is gated behind a free
EEA/Copernicus **Land Portal** login (an EU-Login account). This is a *different* system from
the Copernicus **Data Space** (Sentinel) whose credentials are in
`.env` (`COPERNICUS_USERNAME/PASSWORD`), so those creds do **not**
authorize the CLC download. The EEA `discomap` ArcGIS services only expose *styled* MapServer
renderings (RGB, not raw class codes), from which recovering the 44-class codes would be
fragile.

Instead we read the **identical** CLC2018 100 m product from **Google Earth Engine**, asset
`COPERNICUS/CORINE/V20/100m/2018`, band `landcover` = the raw 3-digit CLC grid code, using
the authorized GEE service-account key referenced by `.env`
(`/etc/credentials/gcp_credentials.json`; spec §8 authorizes GEE creds). This avoids the
credential gate entirely and yields raw class codes directly. No open no-login full-raster
mirror was needed.

## Processing recipe

CLC is a large European **derived-product map**, so per spec §§4–5 we do **bounded-tile,
homogeneous-window** sampling (no full coverage):

1. **Blocks (56 curated regions).** Fetch 56 native-100 m EPSG:3035 blocks
   (1500×1500 px ≈ 150 km each) via `ee.data.computePixels`, snapped to the CORINE LAEA grid
   (origin 900000, 5500000) so the read is pixel-aligned (no resampling in). Regions span
   every European biogeographic zone (Boreal, Alpine, Atlantic, Continental, Pannonian,
   Steppic, Mediterranean, Macaronesian, Black Sea) and are placed to include the
   geographically-restricted classes: rice (Po/Camargue/Valencia/Thessaloniki), olive groves
   (Andalusia/Puglia/Peloponnese), agro-forestry/dehesa (Extremadura/Alentejo), glaciers &
   bare rock (Alps/Iceland/Norway), intertidal flats & salt marshes (Wadden Sea), salines &
   coastal lagoons (Venice/Camargue/Aegean), estuaries (Gironde/Tagus), peat bogs
   (Ireland/Scotland/Fennoscandia/Baltics). Blocks are cached as GeoTIFFs under
   `raw/corine_land_cover/blocks/` for reproducibility/idempotency.
2. **Homogeneous-window scan.** Each block is scanned on its native 100 m grid in
   non-overlapping **6×6 (≈600 m) windows**; a window qualifies when a single CLC class
   occupies **≥ 60 %** of the window (§4 "prefer homogeneous/high-confidence windows for
   derived maps"). The window's **dominant class** is its label for balancing. Up to 200
   qualifying windows per (block, class) are kept (bounds memory). → 194,643 candidate
   windows.
3. **Tiles-per-class balancing.** `balance_by_class(..., total_cap=25000)` selects up to
   1000/class, lowered by the 25 k cap to **25000 // 44 = 568/class**, prioritizing rarer
   classes. → 24,373 selected windows.
4. **Reproject & write.** Each selected window is reprojected from EPSG:3035 100 m to a local
   **UTM** projection at **10 m** with **nearest** resampling (categorical) into a 64×64
   patch. The output tile keeps the **true CLC class of every pixel** (full multi-class
   segmentation), not just the dominant class; only genuine source nodata / unclassified
   codes (0 / 990 / 995 / 999) become 255.

## Class mapping

Output class id = index of the CLC grid code in ascending order (0 = `111` Continuous urban
fabric … 43 = `523` Sea and ocean). The full id↔code↔name↔description table (definitions
condensed from the CLC2018 nomenclature guidelines) is in
`datasets/corine_land_cover/metadata.json` (`classes[]`, each with `clc_code`).

## Time range & change handling

CLC2018 is a **static per-year land-cover state** (2018 reference year). Per §5: `time_range`
= a 1-year window on 2018 (`[2018-01-01, 2019-01-01)`), `change_time = null`. No change
labels. All labels are 2018 → well inside the Sentinel era (no pre-2016 filtering needed).

## Sample counts

24,373 total; 44/44 classes present. Cap 568/class reached by 41 classes. Only three small
artificial classes fall below cap because they rarely form homogeneous ≥600 m windows:
**Dump sites 301, Road & rail networks 367, Construction sites 441**. Per §5 all classes are
kept even where sparse; downstream assembly drops any too-small ones. Full counts in
`metadata.json` (`class_counts`).

## Caveats

- **Coarse native resolution.** CLC is 100 m with a 25 ha MMU, so a 640 m tile carries only
  ~6–7 native pixels per side — a deliberately coarse land-cover probe, not a 10 m-native
  signal. The 10 m output grid is an upsampling (nearest) of the 100 m source.
- **Homogeneity bias.** Selection favours windows where one class dominates (≥60 %), so tiles
  are cleaner/more homogeneous than a random CLC crop; minority classes still appear within
  tiles (full segmentation) but the class-balance is driven by dominant class.
- **Bounded sampling, not full coverage.** 56 regions, not all of Europe; representative
  rather than exhaustive.
- **Artificial small classes are sparse** (see above).

## Verification (§9)

- 24,373 `.tif` and 24,373 `.json` (1:1). Sampled 2,000 tiles: all single-band uint8, 64×64,
  local UTM (EPSG:326xx) at exactly 10 m; 0 bad. All pixel values ∈ {0–43, 255}; no values
  outside the metadata class map; all 44 classes observed.
- **Spatial/temporal sanity:** for 12 random samples, an independent GEE point query of
  `COPERNICUS/CORINE/V20/100m/2018` at each tile's centre lon/lat matched the tile's
  centre-pixel class **12/12** (e.g. a "Sea and ocean" tile at (-1.43, 44.95) lands in the
  Bay of Biscay; a "Peat bogs" tile at (-18.45, 63.83) lands in Iceland) — confirming the
  georeferencing is exact end-to-end. `time_range` is a 1-year 2018 window; `change_time`
  null.
- Idempotent: cached blocks under `raw/` and existing `locations/{id}.tif` are skipped on
  re-run.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.corine_land_cover --workers 64
```

Requires the GEE service-account key at `/etc/credentials/gcp_credentials.json`.
Outputs to `/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/corine_land_cover/`.
