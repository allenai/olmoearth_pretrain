# USGS MRDS (Mineral Resources Data System)

- **Slug:** `usgs_mrds_mineral_resources_data_system`
- **Task type:** classification (positive-only point **detection** encoding)
- **Status:** completed — 14,928 samples (13,928 positive + 1,000 background negatives)
- **Source:** USGS Mineral Resources Data System (MRDS) — a global point database of mineral
  deposits, mines, prospects and occurrences with commodity / deposit-type attributes.
  Public domain.
- **Access:** national CSV export, no credentials:
  `https://mrdata.usgs.gov/mrds/mrds-csv.zip` (project page https://mrdata.usgs.gov/mrds/).
  NOTE: the USGS CDN (Cloudflare) rejects the default urllib User-Agent with HTTP 403 — the
  download passes a browser `User-Agent` header (`download.download_http(..., headers=UA)`).

## What the source is

304,632 mineral-site records, each with lon/lat (EPSG:4326), a development status
(`dev_stat`), a primary commodity (`commod1`), deposit type, operation type, and a
data-quality `score`. Global coverage but US-dominated (89% of the observable subset is in
the United States; then Chile, Peru, Mexico, Brazil, Argentina, Canada, Bolivia).

## This is a positive-only detection dataset

An MRDS point marks the **presence** of a mineral site; absence is everywhere else. Per
spec §4 we use the tunable **detection encoding** (`sampling.encode_detection_tile`): a 1 px
positive at the site, a nodata (255) buffer ring, and background (0) fill in a context tile,
plus background-only negative tiles. The **class is the primary commodity**. This mirrors
the point half of the `usgs_usmin_mine_features` precedent.

## Observability assessment (spec §8) — the crux for MRDS

MRDS is explicitly flagged as containing many sub-pixel prospects/occurrences with
low-precision coordinates. Two mitigations were applied:

1. **Development-status filter (observability).** Kept only `dev_stat ∈ {Producer, Past
   Producer, Prospect}` — sites with a physical ground disturbance (extraction workings,
   pits, dumps) plausibly observable at 10–30 m. **Dropped** `Occurrence` (67,526; a
   documented mineral presence / rock sample with no surface expression), `Plant` (3,008;
   processing buildings, not an extraction footprint) and `Unknown` (26,809; observability
   indeterminate). After filtering: 207,289 records; 196,164 remain after dropping records
   with no primary commodity or invalid coordinates.
2. **Wide ignore ring for coordinate imprecision.** MRDS coordinates are frequently
   PLSS-section-derived, so true positional error is often 100–400 m even though lon/lat are
   stored to 5 decimals (99% of records have 4–5 decimal places, but that is nominal
   precision, not accuracy). Detection encoding therefore uses **tile_size=48,
   positive_size=1, buffer_size=12** → a 25×25 (~250 m) nodata ignore ring centered on the
   site inside a 48×48 tile, so a site that lands a couple hundred metres off is *ignored*,
   not scored as a false negative, while ample background remains in the tile. (USMIN used
   32/1/10; MRDS gets the larger tile + buffer because its coordinates are coarser.)

**Honest caveat:** even with the filter, these are **weak presence-detection targets**
("a mineral mine is present near here"), not precise footprints. Producers/past-producers
of bulk commodities (sand & gravel, stone, clay, iron, copper, uranium) are the most
genuinely resolvable (large open pits / quarries); many metal prospects are small historic
workings. Kept per the task instruction (detection encoding with a ≥10 px buffer, rather
than rejecting), and flagged here. For higher-positional-accuracy mine symbols in the
Western US, prefer `usgs_usmin_mine_features`.

## Class scheme

- id **0 = background**; id **255 = nodata/ignore** (detection buffer rings).
- Commodity classes are ids **1..115** assigned in **descending frequency** (honors the
  254-class uint8 cap — 115 ≪ 254, so no classes were dropped for the cap).
- **Primary commodity** = first comma-token of `commod1`, parentheticals stripped,
  slugified, then a small synonym/merge map applied (e.g. `Barium-Barite`→`barite`,
  `Fluorine-Fluorite`→`fluorite`, `Phosphorus-Phosphates`→`phosphate`,
  `Gypsum-Anhydrite`→`gypsum`, `Talc-Soapstone`→`talc`, `REE`→`rare_earths`,
  `PGE`→`platinum_group`, `Sand`→`sand_and_gravel`, `Coal/Lignite/Bituminous`→`coal`,
  `Halite`→`salt`, Titanium/Copper sub-forms merged).
- **Dropped non-observable fluid/energy commodities** (no surface mine footprint):
  geothermal, natural gas, petroleum, carbon dioxide, helium, water, iodine, bromine,
  chlorine, nitrogen-nitrates, oil shale, oil sands, rock asphalt.
- Per spec §5, **rare classes are kept** (10 commodity classes have a single sample, e.g.
  rhodium, gallium, germanium, thallium); downstream assembly filters classes below the
  minimum sample count.

## Sampling & time

- **Per-class cap = 208** (`POS_BUDGET=24000 // 115 classes`), so the dataset stays
  class-balanced under the 25k hard cap. 54 common commodities hit the 208 cap; the rest
  keep all their (fewer) samples. Realized: **13,928 positive tiles** + **1,000
  background-only negatives** = **14,928 total**.
- **dev_stat preference:** within each commodity, records are selected **Producer > Past
  Producer > Prospect** (random within a tier), so the strongest-signal actual mines are
  chosen first. `dev_stat` and MRDS `dep_id` are recorded in each sample's `source_id`.
- **Neighbor marking:** a global EPSG:3857 KDTree marks every other classified site within
  a tile as an additional positive (its own commodity id), so co-located sites are not
  mislabeled background.
- **Negatives:** 1,000 background-only tiles, centers offset 3–15 km from a random site and
  verified site-free within 1 km (lat clamped to −60..75).
- **Time range:** mine sites are persistent and undated in MRDS; per spec §5 (static
  labels) each sample gets a 1-year window at a representative Sentinel-era year,
  pseudo-randomly spread across **2016–2022**. `change_time` is null. (Manifest's
  `time_range [2016,2016]` is a placeholder.)

## Verification

- 14,928 `.tif` + 14,928 `.json`, fully paired. All tiles single-band uint8, local UTM at
  10 m, 48×48, nodata=255; sampled pixel values all in {0, valid commodity id, 255}. The
  all-background negative tiles are all 0. Max JSON time span 366 days (leap-year window);
  `change_time` null throughout. `metadata.json` lists 116 classes (background + 115
  commodities) covering all label values.
- **Georeferencing:** for sampled positive tiles the single positive pixel sits exactly at
  the tile center (row/col 24 of 48) and the reprojected center lon/lat round-trips to the
  source site coordinate (e.g. sample 000000 → −103.79/44.36, the Black Hills, SD gold
  district — plausible for a gold Producer). CRS/bounds/resolution validated.
- **S2 overlay not run:** a Sentinel-2 image overlay was not performed (imagery access not
  available in this environment). Because MRDS points are deliberately treated as
  imprecise weak-presence targets with a ~250 m ignore ring, a per-pixel overlay is of
  limited diagnostic value here; georeferencing was instead validated by coordinate
  round-trip as above.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_mrds_mineral_resources_data_system
```
Idempotent (skips already-written `locations/{id}.tif`).
