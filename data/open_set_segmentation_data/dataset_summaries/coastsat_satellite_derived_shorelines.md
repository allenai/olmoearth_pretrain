# CoastSat Satellite-Derived Shorelines

- **Slug:** `coastsat_satellite_derived_shorelines`
- **Status:** completed
- **Task type:** classification (binary line segmentation)
- **Num samples:** 17,533 (14,533 positive shoreline tiles + 3,000 background-only negatives)
- **Classes:** `0 = background`, `1 = shoreline`
- **Reproduce:** `python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.coastsat_satellite_derived_shorelines`

## Source

CoastSat ([kvos/CoastSat](https://github.com/kvos/CoastSat)) is an open-source toolkit
that maps the instantaneous land/water boundary of sandy coastlines from 40 years of
Landsat + Sentinel-2 imagery (validated horizontal accuracy ~10–15 m). Regional
satellite-derived shoreline products are published on Zenodo (CC-BY-4.0). Two regional
releases were used:

- **Pacific Rim** — Zenodo record [15614554](https://zenodo.org/records/15614554),
  file `shorelines.geojson` (3,146 beaches; Pacific basin incl. SE Australia, NZ, Chile,
  W US, Japan).
- **US East Coast** — Zenodo record [18435286](https://zenodo.org/records/18435286),
  file `US_East_shorelines.geojson` (301 beaches; US Atlantic + Gulf coast).

The manifest lists the region as "Pacific Rim, SE Australia, Atlantic Europe". The Pacific
Rim release covers the Pacific Rim + SE Australia directly; the US East Coast release is
used as the Atlantic-side representative (no standalone Atlantic-Europe `shorelines.geojson`
was found on Zenodo at process time).

## Access method

Only the two `shorelines.geojson` files (~22 MB total) were downloaded via the Zenodo
files API. The large `shoreline_data.zip` (per-transect time series, ~0.7–1.2 GB each) was
**not** downloaded — it is not needed for the label signal. Raw files and a `SOURCE.txt`
are under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/coastsat_satellite_derived_shorelines/`.

## Label mapping / encoding

Each `shorelines.geojson` feature is one sandy-beach **reference shoreline** LineString in
WGS84 lon/lat (a median/representative position aggregated over the full ~1984–2024 record;
attributes: beach length, median orientation, median beach slope, tidal range, confidence
interval). Following the spec §4 *lines* recipe (and mirroring the `termpicks` glacier-front
script), each beach line is:

1. tiled into up to 8 window centers sampled at 600 m spacing along its length (beaches are
   km-scale: median ~1.4 km Pacific / ~10 km US East), capped so a few very long beaches do
   not dominate;
2. for each window, reprojected into the local UTM (10 m) projection, dilated ~1 px
   (`buffer` radius 1.0, `all_touched=True`) → a ~2–3 px (~20–30 m) wide mask so it is
   visible at 10 m, and rasterized into a 64×64 tile (`1 = shoreline`, `0 = background`);
3. windows clipping < 3 shoreline pixels are dropped.

3,000 **background-only negative** tiles are added, offset 3–30 km from decimated shoreline
vertices and rejected if within 1.5 km of any shoreline vertex. Background is a real class 0
here (land or open water away from the shoreline), consistent with the binary line-segmentation
precedent (`termpicks`).

## Time-range and change handling

The reference shorelines are median 1984–2024 aggregates, i.e. effectively **static**, so a
single representative 1-year Sentinel-era window (**2020**, `[2020-01-01, 2021-01-01)`) is
assigned to every sample with `change_time = null` (spec §5 static-label rule).

## Signal intentionally NOT used

The CoastSat transect product's **linear trend (m/yr)** = "erosion vs accretion" (the
second manifest class) is a **multi-decadal change rate**, not a dated change event, and is
not observable within a single 1-year pretraining window. Per spec §5 (change-timing /
observability), it cannot be expressed as a per-pixel class/regression at the pairing
timescale, so it is dropped (and the per-transect time-series archive was not downloaded).
The usable, observable signal is the shoreline (land/water boundary) position, which is
exactly what CoastSat extracts at 10–15 m from Sentinel-2/Landsat.

## Sample counts

- Positive tiles with shoreline: 14,533
- Background-only negatives: 3,000
- Total: 17,533 (well under the 25k per-dataset cap)
- Source beaches: 3,146 (Pacific Rim) + 301 (US East) = 3,447; geographic spread lon
  −160…178, lat −55…48 (Pacific) and the US Atlantic/Gulf coast.

## Verification

- 3–5 output `.tif`s: single-band, `uint8`, UTM CRS at 10 m, 64×64, values ⊆ {0, 1};
  UTM zone varies correctly by location; all 17,533 `.tif`s have a matching `.json` with a
  1-year `time_range` and `change_time = null`. `metadata.json` class ids cover all values.
- Shoreline pixel fraction per positive tile: median ~5.3% (~217 px), consistent with a
  ~3-px-wide line across a 64-px tile.
- **Spatial sanity check (Sentinel-2 overlay).** For sampled positive tiles a low-cloud 2020
  S2 scene was fetched (Element84 public STAC) and NDWI computed at the tile grid: 3 of 4
  tiles straddle land + water (tile water-fraction ~0.45) with shoreline-labeled pixels
  sitting at the water/land boundary (mean |NDWI| at shoreline pixels 0.04–0.17 vs 0.24–0.59
  whole-tile), i.e. the label overlays the real coastline. 1 of 4 (a Peru beach) showed the
  labeled line offset landward of the 2020 instantaneous shoreline — the expected caveat below.
- Re-running the script is idempotent (existing `{sample_id}.tif` are skipped).

## Caveats

- The rasterized line is a **median/reference** shoreline, so the instantaneous shoreline in
  any single image can be offset by the beach's variability (often ~10–50 m). The ~1 px
  dilation and coarse 10 m raster partly absorb this, but some tiles will have the mask a few
  pixels landward/seaward of the boundary in a given image (observed in ~1/4 of the S2 spot
  checks). This is a coarse shoreline mask, not a per-image shoreline.
- Only sandy coastlines are represented (CoastSat's scope); rocky/muddy/mangrove coasts are
  absent.
- The erosion/accretion (trend) signal is deliberately excluded (see above).
