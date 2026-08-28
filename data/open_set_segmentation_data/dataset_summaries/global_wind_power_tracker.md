# Global Wind Power Tracker (GWPT)

- **Slug:** `global_wind_power_tracker`
- **Status:** completed · classification (presence-only points) · **1,367 points**
- **Source:** Global Wind Power Tracker, Global Energy Monitor (GEM), **February 2026 release**
  (<https://globalenergymonitor.org/projects/global-wind-power-tracker>).
- **License:** CC-BY-4.0
- **Annotation method:** manual/expert curation.

## Source & access

Facility-level inventory of utility-scale (≥ 10 MW) onshore/offshore wind power project phases
worldwide. The Feb-2026 release lists 33,248 phases (each a point with `Latitude`/`Longitude`,
`Status`, `Installation Type`, `Start year`, optional `Retired year`, `Location accuracy`).
Distributed behind GEM's email-gated download form (not a credential gate); reproduced with the
`mint_submission`→`presign` HTTP flow (recipe in `raw/global_wind_power_tracker/SOURCE.txt`) →
`Global-Wind-Power-Tracker-February-2026.xlsx` (~4.9 MB). No imagery downloaded.

## Label type — presence-only points

**Converted from the old positive-only object-detection tile encoding** (32×32 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each operating farm is one point of its observable class. There is **no fabricated GeoTIFF
context, and no background / buffer / negative tiles** — this dataset carries **no fabricated
negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts

Only **operating** phases used; onshore vs offshore kept as two observable classes:

| id | name | points |
|----|------|--------|
| 0 | onshore_wind_farm | 1000 |
| 1 | offshore_wind_farm | 367 |

Up to 1000 points/class (`balance_by_class`); offshore is a rare class (~367) kept in full (spec
§5). **1,367 points total.**

## Time handling

Persistent-structure model (`change_time = null`): each farm gets a 1-year window sampled from
`[max(start_year, 2016), min(2025, retired − 1)]` (missing `start_year` → `[2016, 2025]`); phases
first operating after 2025 skipped. GWPT resolves commissioning only to a calendar year (coarser
than the ~1–2 month change-timing rule), so no dated change labels.

## Output

- `datasets/global_wind_power_tracker/points.geojson`
- `datasets/global_wind_power_tracker/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_wind_power_tracker
```

Idempotent. The `.xlsx` is fetched via the presign recipe into `raw/global_wind_power_tracker/`.

## Caveats

- GWPT points are farm/project-level (mark the farm location, not each turbine); ~30% of
  operating points are "approximate".
