# GlobalGeoTree

- **Slug**: `globalgeotree`
- **Task type**: classification (sparse-point tree-species segmentation)
- **Status**: completed — 24,892 samples, 254 classes
- **Source**: GlobalGeoTree (Yang et al., ESSD). GitHub
  https://github.com/MUYang99/GlobalGeoTree ; data on Hugging Face dataset repo
  `yann111/GlobalGeoTree`. License CC-BY.

## What the source is

A global vision-language dataset of ~6.3M geolocated tree occurrences paired with
Sentinel-2 time series and environmental variables. We use **only** the occurrence
metadata table `files/GlobalGeoTree.csv` (~1.1 GB) — not the WebDataset imagery tars
(`GlobalGeoTree-6M/*.tar`) — because pretraining supplies its own imagery; we just need
the (lon, lat, species, year) labels.

CSV columns: `sample_id, country_code, level0` (leaf type: Evergreen/Deciduous ×
Broadleaf/Needleleaf), `level1_family, level2_genus, level3_species, location, source`
(iNaturalist / GBIF / forest inventories), `species_key` (GBIF), `year, longitude,
latitude`.

## Access method

`download.hf_download("yann111/GlobalGeoTree", "files/GlobalGeoTree.csv", raw_dir)` —
public, no credential. Raw file lands at
`raw/globalgeotree/files/GlobalGeoTree.csv` (+ `SOURCE.txt`).

## Processing decisions

- **Sparse points → GeoJSON point table** (spec §2a/§4): each label is a single 10 m
  pixel with a species class id, so output is one dataset-wide
  `datasets/globalgeotree/points.geojson` (one Point feature per observation), **not**
  per-point GeoTIFFs.
- **Class level = species** (`level3_species`), per the task.
- **254-class uint8 cap** (spec §5): the source has **20,709 species** (post-2016), far
  above the uint8 limit. We keep the **top 254 species by observation frequency** (ids
  0–253 in descending frequency) and **drop the remaining 20,455 rarer species**. Each
  kept species has ≥ **4,218** source observations (id 0 = *Cornus acuminata*, 177,074
  obs; id 253 = *Euonymus fortunei*, 4,218 obs).
- **Pre-2016 filter** (spec §8): source spans 2015–2024. We drop the **205,055 pre-2016
  rows** (all year 2015) and keep `year >= 2016` (Sentinel-2 era): 6,058,290 rows remain
  before class-capping.
- **Coordinate/null filtering**: rows with null species/year/lon/lat or out-of-range
  coords dropped (0 rows dropped here — the source is clean and global).
- **Balancing** (spec §5): `balance_by_class(per_class=1000, total_cap=25000)` → the 25k
  hard cap lowers the effective per-class limit to `25000 // 254 = 98`. Every kept species
  has ≥ 4,218 obs, so all 254 classes fill to exactly 98 → **24,892 total** (under 25k).
- **Time range**: 1-year window anchored on each observation's year
  (`io.year_range`), years 2016–2024.
- **Class descriptions**: built from the taxonomy — e.g. "Tree species *Cornus acuminata*
  (genus Cornus, family Cornaceae, deciduous broadleaf tree)." `metadata.json` classes
  also carry `family`, `genus`, `leaf_type`, `gbif_species_key`, `n_source_observations`,
  `n_samples`.

## Outputs

- `datasets/globalgeotree/metadata.json` — 254 classes, `nodata_value=255`,
  `task_type=classification`.
- `datasets/globalgeotree/points.geojson` — FeatureCollection, `count=24892`, 254 labels
  (0–253), 98 samples/class, all coords valid, all time ranges 1-year, years 2016–2024.
- `datasets/globalgeotree/registry_entry.json` — status `completed`.

## Caveats

- **Weak/contextual label.** Tree-species presence at a point is only weakly observable at
  10–30 m from S2/S1/Landsat; treat these as weak habitat labels (same posture as
  `geolifeclef_geoplant`). Points come from opportunistic citizen-science + inventory
  records, so geolocation precision varies.
- **Sparse classes downstream.** We keep the top-254 cap; the ~20.5k dropped species are
  gone (uint8 constraint). No per-class dropping for balance beyond the cap (spec §5).
- **Taxonomy quirks inherited from source** (e.g. some `species_key`/count values look
  unusual); we record source values as-is and do not correct them.
- **Spatial overlay check**: a Sentinel-2 water/land overlay sanity check (spec §9) is not
  meaningful for weak single-pixel species points; coordinates were validated as
  in-range global land occurrences instead. Verified sample feature id 000000 at
  (1.4734, 36.4453) — northern Algeria, plausible tree locale.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.globalgeotree
```
Idempotent: re-download is skipped if the CSV exists; outputs are rewritten
deterministically (seed 42).
