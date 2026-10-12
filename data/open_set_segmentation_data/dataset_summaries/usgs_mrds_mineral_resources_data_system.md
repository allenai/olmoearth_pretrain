# USGS MRDS (Mineral Resources Data System)

- **Slug:** `usgs_mrds_mineral_resources_data_system`
- **Status:** completed · classification (presence-only points, multi-class by commodity) ·
  **14,414 points**
- **Source:** USGS Mineral Resources Data System (MRDS) — a global point database of mineral
  deposits, mines, prospects and occurrences. Public domain. <https://mrdata.usgs.gov/mrds/>
- **Annotation method:** manual compilation of mineral deposit records.

## Source & access

National CSV export, no credentials: `https://mrdata.usgs.gov/mrds/mrds-csv.zip` (the USGS CDN
rejects the default urllib UA with HTTP 403, so a browser `User-Agent` header is sent). 304,632
mineral-site records, each with lon/lat, development status (`dev_stat`), primary commodity
(`commod1`), etc. US-dominated. No imagery downloaded.

## Label type — presence-only points

**Converted from the old positive-only point-detection tile encoding** (48×48 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each kept site is one point carrying its primary-commodity class id. There is **no fabricated
GeoTIFF context, and no background / buffer / negative tiles** — this dataset carries **no
fabricated negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts

Class = primary commodity (`commod1` first token, normalized/merged), **115 classes**, ids by
descending frequency (254-class uint8 cap not hit). **14,414 points total.** Inclusion:
`dev_stat ∈ {Producer, Past Producer, Prospect}` (physical ground disturbance); dropped
Occurrence/Plant/Unknown and non-observable fluid/energy commodities (geothermal, natural gas,
petroleum, helium, CO2, water, brine halogens).

Balanced under the 25k-total cap, so the 54 common commodities cap at **217 points/class** (gold,
sand_and_gravel, stone, copper, iron, lead, silver, uranium, clay, zinc, …); rarer commodities
keep all their samples (10 classes have a single point, e.g. rhodium, gallium, germanium). Within
a commodity, records are chosen Producer > Past Producer > Prospect.

## Time handling

Persistent, undated sites → 1-year `time_range` at a representative Sentinel-era year (spread
across **2016–2022**). `change_time = null`.

## Output

- `datasets/usgs_mrds_mineral_resources_data_system/points.geojson`
- `datasets/usgs_mrds_mineral_resources_data_system/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_mrds_mineral_resources_data_system
```

Idempotent.

## Caveats

- MRDS coordinates are frequently low-precision (PLSS-section-derived; true error often
  100–400 m), so these are **weak presence targets**, not precise footprints. For
  higher-positional-accuracy mine symbols in the Western US, prefer `usgs_usmin_mine_features`.
