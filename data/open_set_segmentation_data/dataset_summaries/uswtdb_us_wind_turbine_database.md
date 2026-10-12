# USWTDB (US Wind Turbine Database)

- **Slug:** `uswtdb_us_wind_turbine_database`
- **Status:** completed · classification (presence-only points) · **1,000 points**
- **Source:** USGS / LBNL / AWEA — U.S. Wind Turbine Database, public domain (U.S. Government
  work). <https://energy.usgs.gov/uswtdb/>
- **Annotation method:** manual position verification against high-resolution aerial imagery.

## Source & access

Authoritative national inventory of onshore/offshore wind turbines in the U.S. and territories,
each position-verified against aerial imagery. Downloaded the turbine points (75,727 records) as
one JSON array from the public USGS PostgREST API `https://energy.usgs.gov/api/uswtdb/v1/turbines`
(no credentials; browser User-Agent header) via `download.download_postgrest_json()` →
`raw/{slug}/`. Fields include `case_id`, `xlong`/`ylat`, project online year `p_year`,
`t_offshore` (79 offshore). No imagery downloaded.

## Label type — presence-only points

**Converted from the old positive-only object-detection tile encoding** (64×64 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each selected turbine is one point of the single foreground class. There is **no fabricated
GeoTIFF context, and no background / buffer / negative tiles** — this dataset carries **no
fabricated negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts

Single class `0 = turbine` (onshore and offshore both used). **1,000 points** (up to 1000/class,
`balance_by_class`).

## Time handling

Presence/state, not change: `p_year` is year-granular only (not resolvable to ~1–2 months per the
change-timing rule), so the persistent post-construction state is used with `change_time = null`
and a 1-year window anchored after commissioning: `window_year = clamp(p_year + 1, 2017, 2024)`;
missing `p_year` → 2022. Keeps pre-2016 turbines (still standing post-2016) while honoring the
post-2016 rule.

## Output

- `datasets/uswtdb_us_wind_turbine_database/points.geojson`
- `datasets/uswtdb_us_wind_turbine_database/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.uswtdb_us_wind_turbine_database
```

Idempotent.

## Caveats

- Turbines are ~1 px objects at 10 m; a point marks presence, not a resolvable footprint.
  Coordinates are among the most accurate available (manual verification).
