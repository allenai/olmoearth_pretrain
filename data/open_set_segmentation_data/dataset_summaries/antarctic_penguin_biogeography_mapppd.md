# Antarctic Penguin Biogeography / MAPPPD

- **Slug:** `antarctic_penguin_biogeography_mapppd`
- **Status:** completed
- **Task type:** classification (sparse species-presence points)
- **Num samples:** 318 points
- **Family / region:** wildlife / Antarctica & sub-Antarctic (south of 60°S)
- **License:** CC-BY-4.0

## Source & access

The **Antarctic Penguin Biogeography Project** "Count data" — the database behind
**MAPPPD** (Mapping Application for Penguin Populations and Projected Dynamics,
[www.penguinmap.com](https://www.penguinmap.com), Oceanites / Stony Brook IACS). Published
as a **Darwin Core Archive** on the SCAR/AADC IPT and mirrored on GBIF:

- DwC-A: `https://ipt.biodiversity.aq/archive.do?r=mapppd_count_data`
- GBIF dataset: `https://www.gbif.org/dataset/f7c30fac-cf80-471f-8343-4ec5d8594661`

No credential needed (public, CC-BY-4.0). Downloaded the ~390 KB archive directly (no
bulk imagery pull). The archive has an **Event core** (`event.txt`: one survey at a
breeding site on a date, with WGS84 `decimalLatitude/decimalLongitude`, `year`,
`coordinateUncertaintyInMeters`) and an **Occurrence extension** (`occurrence.txt`: one
penguin species per event with `organismQuantity`, `occurrenceStatus`, `vernacularName`,
`scientificName`), joined on `eventID`. 4,055 events / 4,768 occurrences total.

## Why accepted

Penguin breeding colonies leave **persistent guano stains** detectable in Landsat /
Sentinel-2 at 10–30 m (the manifest note and MAPPPD's own satellite-based emperor counts
confirm observability), so a species presence at a colony is a valid **species-presence
label** (class = penguin species). Colony locations are points → written as the sparse
point table `points.geojson` (spec §2a), not per-point GeoTIFFs.

## Class mapping (manifest order → id)

`vernacularName` / `scientificName` map directly to the six manifest classes:

| id | name | scientific | count |
|----|------|------------|-------|
| 0 | Adelie | *Pygoscelis adeliae* | 47 |
| 1 | chinstrap | *Pygoscelis antarctica* | 160 |
| 2 | gentoo | *Pygoscelis papua* | 74 |
| 3 | emperor | *Aptenodytes forsteri* | 17 |
| 4 | macaroni | *Eudyptes chrysolophus* | 16 |
| 5 | king penguin | *Aptenodytes patagonicus* | 4 |

All 318 selected points span all six classes; nodata/ignore = 255. King penguin is sparse
(4) — kept per §5 (do not drop sparse classes; downstream assembly filters too-small ones).

## Filtering, dedupe, and time handling

- **Present only:** dropped `occurrenceStatus=absent` (160 occurrences).
- **Sentinel era (2016+):** all records are dated (`year` 1892–2022, none undated), so per
  §5 / the task note we **kept only surveys dated ≥ 2016** (725 present occurrences 2016+).
  Pre-2016 surveys were filtered out (they are not the entirety of the dataset, so this is
  a filter, not a rejection).
- **Dedupe:** one point per **(colony site `locationID`, species)**, keeping the **most
  recent** 2016+ survey year → 318 distinct points across 257 sites.
- **Time range:** colonies are persistent, so each point is a **static label** with a
  **1-year window anchored on its survey year** (`io.year_range`), `change_time = null`.
  Years used fall in 2016–2022 (matching the manifest `time_range`).
- **Balancing:** `balance_by_class(per_class=1000, total_cap=25000)` — well under the cap,
  so nothing truncated; all 318 kept.

## Output

- `datasets/antarctic_penguin_biogeography_mapppd/points.geojson` — FeatureCollection,
  one `Point` per (site, species). `properties`: `id`, `label` (class id), `time_range`,
  `change_time` (null), `source_id`, plus auxiliary `coord_uncertainty_m` and `locality`.
- `datasets/antarctic_penguin_biogeography_mapppd/metadata.json` — class map + provenance.
- `raw/antarctic_penguin_biogeography_mapppd/` — the DwC-A zip + extracted `dwca/` + `SOURCE.txt`.

## Verification

- `points.geojson`: 318 features, `task_type=classification`, unique ids, all labels ∈
  class ids {0..5}, all `time_range`s ≤ 1 year, all coords within lat [−77.7, −60.6] (south
  of 60°S) and valid lon.
- Geodetic datum is `EPSG:4326` for every event; coordinates are colony gazetteer
  centroids. Spot check: sample `000000` = Cape Adare (170.20°E, −71.31°S), Adélie — a
  known very large Adélie rookery — georeferences correctly.

## Caveats

- **Coordinate uncertainty:** site coordinates are colony/gazetteer **centroids** with a
  median `coordinateUncertaintyInMeters` ≈ 1.15 km (min 83 m, max ~31 km), recorded
  per-point in `coord_uncertainty_m`. At 10 m/pixel the labeled pixel can be tens to a few
  hundred pixels off; treat these as approximate presence locations, not pixel-exact
  footprints. This is inherent to the source and acceptable for sparse presence points.
- No colony **polygons** are distributed in this DwC-A (only point sites + counts), so no
  rasterized footprints were produced — points only.
- Positive-only (presence) dataset: no synthetic negatives fabricated (§5); non-object
  pixels are left to downstream assembly, which supplies negatives from other datasets.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.antarctic_penguin_biogeography_mapppd
```

Idempotent: re-downloads only if missing (`skip_existing`) and rewrites `points.geojson` /
`metadata.json` atomically to the same result.
