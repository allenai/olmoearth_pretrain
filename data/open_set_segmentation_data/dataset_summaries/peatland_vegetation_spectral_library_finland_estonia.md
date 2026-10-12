# Peatland Vegetation Spectral Library (Finland & Estonia)

- **Slug:** `peatland_vegetation_spectral_library_finland_estonia`
- **Status:** completed
- **Task type:** classification (sparse points → point table, spec §2a)
- **Num samples:** 446 points
- **Family / region:** wetland / Finland (323 plots) & Estonia (123 plots)
- **License:** CC-BY-4.0

## Source

Mendeley Data `3866tj3w8v` v1, Salko, S.-S., Hovi, A., Burdun, I., Juola, J.,
Rautiainen, M. (2024), *"Geographically extensive spectral library of peatland
vegetation from 13 hemiboreal, boreal, sub-Arctic and Arctic peatland sites"*
(DOI 10.17632/3866tj3w8v.1; companion paper Ecological Informatics DOI
10.1016/j.ecoinf.2024.102772).

446 georeferenced 1 m × 1 m field plots across 13 peatland sites, measured in the
2022 (162 plots) and 2023 (284 plots) growing seasons with an ASD FieldSpec 4
spectroradiometer. Each plot carries: WGS84 lon/lat (EPSG:4326), survey date, a
Finnish peatland-classification type, per-plot plant-functional-type (PFT) fractional
cover (11 categories, %), tree basal areas, and a 350–2500 nm reflectance spectrum.

### Access method

Open HTTP via the Mendeley public API (no credential). Files listed at
`https://data.mendeley.com/public-api/datasets/3866tj3w8v/files?folder_id=root&version=1`;
downloaded with a Firefox User-Agent to
`raw/peatland_vegetation_spectral_library_finland_estonia/`:
- `Data_description.pdf` — codebook / column descriptions.
- `raw.csv` — reflectance + all plot metadata + PFT columns (used).
- `smoothed.csv` — Savitzky-Golay-smoothed spectra (not needed for labels).

The first 3 CSV rows are citation/reading info; row 4 is the column header; rows 5–450
are the 446 plots. The reflectance spectra (wl350–wl2500 columns) are not used — only the
plot location, type, date, and cover columns form the label signal.

## Label design

This is a pure sparse in-situ POINT dataset → one dataset-wide `points.geojson`
(FeatureCollection, one Point per plot), NOT per-point GeoTIFFs.

**Primary label (classification):** coarse peatland ecohydrological class mapped from the
39 detailed Finnish peatland types:

| id | class | plots | definition |
|----|-------|-------|------------|
| 0 | `bog` | 213 | Ombrotrophic (rain-fed, Sphagnum-dominated) mire: rahkaneva/rahkarame, isovarpurame, tupasvillaneva/-rame, lyhytkorsineva, kalvakkaneva; + Estonian pine-covered `Rame` and treeless `Neva`. |
| 1 | `fen` | 206 | Minerotrophic (groundwater-fed) mire: sedge fens (saraneva, rimpineva), rich fens (letto, rimpiletto), flood fens (luhta, luhtaneva), spruce/hardwood mires (korpi types); + Estonian flood-influenced `Neva_luhtainen`. |
| 2 | `palsa_mire` | 27 | Ombrotrophic peat mounds with a permafrost core (Finnish `Kumpupalsa`), sub-Arctic/Arctic; kept separate as a distinct landform. |

Rationale: the manifest's stated purpose is to add "bog-vs-fen peatland vegetation
classes." A coarse bog/fen/palsa scheme is (a) the natural, robust classification target
and (b) keeps all 446 plots contributing — the alternative of using the 39 fine Finnish
types directly would leave most classes with 3–6 samples, which downstream min-count
filtering would drop, erasing the bog/fen signal. The fine type is preserved as an
auxiliary field (below) for anyone who wants a finer label.

**Auxiliary per-point properties** (in each feature's `properties`): `country`, `site`,
`finnish_peatland_type` (raw 39-value string), `mire_structure` (structural group:
neva/rame/korpi/letto/luhta/palsa), and the cover-fraction quantities named in the
manifest — `pft_sphagnum`, `pft_graminoids`, `pft_woody_stemmed` (shrub), plus
`pft_brown_mosses`, `pft_herbaceous`, `pft_lichen`, `pft_bare_peat`, `pft_water`, and
`tree_basal_area_living` (Σ living pine+spruce+deciduous basal area, tree-cover proxy).
These are regression quantities carried alongside the classification `label` (as
coastbench/gloria do), enabling a cover-fraction regression downstream if desired.

### Bog/fen mapping caveats

Finnish sites use the full trophic Finnish peatland classification (Laine et al. 2012),
so their bog/fen assignment is well-grounded. **Estonian sites (123 plots) are typed only
by tree cover** (per the source: `neva`=treeless, `räme`=pine, `korpi`=spruce), with no
explicit trophic status. I mapped Estonian `Rame` (66) and plain `Neva` (27) → **bog**
(the Estonian study sites are dominated by ombrotrophic raised bogs) and
`Neva_luhtainen` (30) → **fen** (flood/riparian influence). These ~123 Estonian
assignments therefore carry more uncertainty than the Finnish trophic types; noted here
and in `metadata.json`.

## Time range

Quasi-static peatland vegetation → a 1-year window `[Jan 1 YYYY, Jan 1 YYYY+1)` anchored
on each plot's survey year (2022 or 2023). `change_time = null`. All plots are post-2016
(Sentinel era), so none are filtered.

## Verification

- `points.geojson`: FeatureCollection, `count=446`, 446 features, `task_type=classification`.
- All labels ∈ {0,1,2}; class counts bog 213 / fen 206 / palsa_mire 27.
- Coordinates within lon 21.05–30.69°E, lat 57.65–68.88°N (Finland & Estonia) — matches
  the source WGS84 columns exactly, so georeferencing is exact by construction (one
  coordinate string with an internal space, `68. 87788`, was cleaned).
- All `time_range`s are 1 year and post-2016; all `change_time` null.
- Spot check: feature 000000 (Halssiaapa_10a, `RiL rimpiletto` → fen) sits at
  67.368°N 26.650°E, the Halssiaapa aapa mire in Finnish Lapland — correct.
- Idempotent: re-running regenerates the same deterministic `points.geojson`/`metadata.json`.

## Reproduce

```bash
# raw files already at raw/peatland_vegetation_spectral_library_finland_estonia/ ;
# to re-download: Mendeley public API file download_urls (Firefox UA), see above.
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.peatland_vegetation_spectral_library_finland_estonia
```
