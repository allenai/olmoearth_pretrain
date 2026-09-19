# OGIM (Oil & Gas Infrastructure Mapping)

- **Slug:** `ogim_oil_gas_infrastructure_mapping`
- **Status:** completed
- **Task type:** classification (object **detection** / presence segmentation, encoded as per-pixel classes)
- **Num samples:** 5,668 GeoTIFF tiles (1,000 well + 1,000 offshore_platform + 1,000 facility + 668 refinery + 1,000 pipeline anchor tiles + 1,000 background-negative tiles)

## Source

"Oil and Gas Infrastructure Mapping (OGIM) database", **v2.5.1**, Environmental Defense
Fund (EDF) / MethaneSAT, LLC. Zenodo record 13259749
(<https://zenodo.org/records/13259749>, doi:10.5281/zenodo.13259749), license
**CC-BY-4.0**. Methods paper: Omara et al., *Earth Syst. Sci. Data* 2023
(<https://doi.org/10.5194/essd-15-3761-2023>).

A global, curated, spatially-explicit database of oil & gas infrastructure, integrated
from official government + industry + academic sources (~6.7M features). Distributed as a
single **~3 GB GeoPackage** `OGIM_v2.5.1.gpkg` (all layers EPSG:4326), plus two small PDFs
(schema + data-source references).

**Access method:** `download.download_zenodo("13259749", raw_dir)` — downloaded only the
label GeoPackage + PDFs (**no imagery**; pretraining supplies its own S2/S1/Landsat). No
credentials required. The gpkg is read with `pyogrio` (attribute-only reads for point
layers; RTree-indexed `bbox=` reads to pull pipelines per tile).

## Layers used → unified class scheme

Mixed **points + lines** combined into ONE dataset with a unified class map (spec §5
multi-modality). Point layers carry `LONGITUDE`/`LATITUDE` attributes (verified bit-identical
to geometry, no nulls); the pipeline layer is LineStrings.

| id | class name | OGIM layer(s) | source feature count |
|----|------------|---------------|----------------------|
| 0 | background | — | (tile fill + negatives) |
| 1 | well | `Oil_and_Natural_Gas_Wells` | 4,519,663 pts |
| 2 | offshore_platform | `Offshore_Platforms` | 9,788 pts |
| 3 | facility | `Natural_Gas_Compressor_Stations` + `Gathering_and_Processing` + `LNG_Facilities` | 23,191 pts |
| 4 | refinery | `Crude_Oil_Refineries` | 686 pts |
| 5 | pipeline | `Oil_Natural_Gas_Pipelines` | 1,903,711 LineStrings |
| 255 | nodata/ignore | — | detection buffer rings around imprecise point locations |

Class 3 (`facility`) merges the three facility-type layers to match the manifest's
"compressor/processing/LNG facilities" class. OGIM layers **not** used (out of scope for
the manifest class list / not point-or-line infrastructure): `Equipment_and_Components`,
`Injection_and_Disposal`, `Natural_Gas_Flaring_Detections`, `Petroleum_Terminals`,
`Stations_Other`, `Tank_Battery`, and the polygon extent layers
(`Oil_and_Natural_Gas_Basins/Fields/License_Blocks`), plus `Data_Catalog`.

## Encoding (spec §4 detection + line rasterization)

Everything is written as **single-band uint8 GeoTIFF tiles**, **64×64**, local UTM
(auto-selected per tile), **10 m/pixel**, north-up — one output modality so points and
lines share one class map. Per tile:

1. **Point features** (all four point classes) → **1 px positive** at the pixel, ringed by
   a **10 px nodata (255) buffer** (21×21 ignore region), because point coordinates are not
   pixel-exact. Parameters `positive_size=1`, `buffer_size=10`.
2. **Pipelines** → rasterized from the precise line geometry, buffered to ~**30 m** (1.5 px
   half-width, `all_touched`), class 5.
3. Every tile is labeled with **ALL OGIM features that fall inside it** (cross-class
   neighbors burned in via a KD-tree over all 4.56M point features + a per-tile pipeline
   bbox read), so a well tile that also contains a facility/pipeline is labeled correctly
   and the map is genuinely unified. Burn precedence: point buffers (255) → pipelines (5,
   precise geometry beats a fuzzy buffer) → point positive centers (win).

Verified end-to-end: all 5,668 tiles are single-band uint8, ≤64×64, EPSG:32### (UTM) at
10 m, pixel values ⊆ {0,1,2,3,4,5,255}. Georeferencing round-trip: sampled tile centers
land **1–12 m** from the true OGIM feature of the anchor class (sub-pixel), and platforms
fall on open water (Gulf of Mexico, Persian Gulf), wells/refineries on land — spatially
sensible.

## Time-range and change handling (spec §5)

Infrastructure is a **persistent structure**, not a dated change event → `change_time=null`,
no change labels. `SRC_DATE` is the *source-publication/update* date (ISO `YYYY-MM-DD`),
not an event date and coarser than the ~1–2 month change-timing bar, so each tile gets a
static **1-year window anchored on its `SRC_DATE` year, clamped to the Sentinel era**:
`year = SRC_DATE year if 2016 ≤ year ≤ 2024 else 2020`. This is valid because the structure
is persistent and observable across the Sentinel era regardless of when its source record
was published (~81% of wells are dated 2024; almost all foreground features are 2016+; only
a small pre-2016 tail is clamped to 2020, a representative recent year).

## Sampling (spec §5)

- Up to **1,000 anchor tiles per foreground class** (well/platform/facility/refinery/pipeline),
  tiles-per-class balanced. Refinery only reaches **668** (there are only 686 refineries;
  the rest collapse under grid-dedup) — kept anyway (sparse classes are fine; downstream
  filters too-small classes).
- **Grid-dedup** to one anchor per ~2 km cell (`GRID_DEG=0.02`) before random sampling, so
  dense basins (Permian, Alberta, …) don't produce thousands of near-identical overlapping
  tiles; the retained anchors are geographically spread.
- Up to **1,000 background NEGATIVE tiles**: a random real feature offset 10–50 km, required
  >~2 km from **any** point feature (KD-tree). (Negatives still run the full burn, so if a
  pipeline happens to cross one it is labeled correctly rather than mislabeled background;
  in practice they are essentially all-background.)
- Total **5,668** ≪ the 25k per-dataset cap.

**Tiles containing each class** (a tile counts toward every class present in it):
background 5,664 · well 2,404 · pipeline 2,523 · facility 1,031 · offshore_platform 986 ·
refinery 667. (Wells and pipelines appear in many other-class tiles thanks to cross-class
burn-in — evidence the unified scheme is working.)

## Caveats

- **Individual wellheads are near/below 10 m resolution** (spec §8 flags this). The `well`
  label is best read as *"a well SITE is present within this ~200 m ignore region"* — onshore
  well pads / clustered well fields do produce visible surface disturbance (cleared pads,
  tanks, access roads) at 10 m, and the thick nodata buffer already absorbs positional
  imprecision. Kept with this caveat; downstream assembly can drop the class if it proves
  unusable. Offshore platforms, facilities, refineries and pipelines are comfortably
  resolvable.
- All feature statuses are kept regardless of `FAC_STATUS`/`OGIM_STATUS` (a decommissioned
  site's surface footprint typically remains visible; assembly can filter if desired).
- The open-access OGIM v2.5.1 omits ~300 Russian compressor stations (per the source note).
- OGIM point layers `Equipment_and_Components`, `Tank_Battery`, `Petroleum_Terminals`,
  `Stations_Other`, `Injection_and_Disposal`, `Natural_Gas_Flaring_Detections` were left out
  to keep the class set aligned with the manifest; they could be added later if desired.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ogim_oil_gas_infrastructure_mapping
```

Idempotent (skips already-written `locations/{id}.tif`). Outputs on weka under
`datasets/ogim_oil_gas_infrastructure_mapping/` (`metadata.json`, `registry_entry.json`,
`locations/{000000..005667}.{tif,json}`); raw gpkg + PDFs under
`raw/ogim_oil_gas_infrastructure_mapping/`.
