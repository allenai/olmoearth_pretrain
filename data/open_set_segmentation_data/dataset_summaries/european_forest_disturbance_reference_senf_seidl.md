# European Forest Disturbance Reference (Senf & Seidl) — REJECTED

- **Slug:** `european_forest_disturbance_reference_senf_seidl`
- **Manifest name:** European Forest Disturbance Reference (Senf & Seidl)
- **Family:** tree_mortality — **change/disturbance dataset**
- **Source:** Zenodo record 3561925 (https://zenodo.org/records/3561925), CC-BY-4.0
- **Label type (manifest):** points/plots; manual TimeSync interpretation
- **Final status:** `rejected`
- **Primary rejection reason:** `change-timing: event not resolvable to within ~1-2 months`

## What the source actually is

The published record contains a **single 921 KB file, `disturbances.csv`** (19,922 rows,
one per interpreted plot across 35 European countries). Columns:

```
country, plotid, disturbance_n,
year_disturbance_1, year_disturbance_2, year_disturbance_3,
agent_disturbance_1, agent_disturbance_2, agent_disturbance_3,
severity_disturbance_1, severity_disturbance_2, severity_disturbance_3
```

Each plot records up to three disturbance events, each with a **year** (`year_disturbance_*`),
an agent (`Harvest`, `Biotic`, `Uprooting and breakage`, `Fire`, `Gravitational event`,
`Unknown canopy disturbance`) and a severity (SR / NSR). 5,639 disturbance events total;
years span **1985–2018**; only 399 events fall in the Sentinel era (>=2016). Interpretation
was done with TimeSync on annual Landsat time series.

## Why it is rejected (two independent, decisive grounds)

1. **Change-timing (§5, the decisive rule for this dataset).** Disturbance timing is
   recorded **only as a calendar year** (`year_disturbance_*`) — there is no month or day.
   TimeSync interpretation of annual Landsat composites is inherently year-resolved. Per §5,
   a change label is only usable if the event can be placed to within ~1–2 months so the
   pairing window is guaranteed to span the change; a year-resolved event cannot be, so the
   sampled imagery may not show the disturbance and the where-mask would be misaligned.
   Reject with `notes: "change-timing: event not resolvable to within ~1-2 months"`.

2. **No recoverable geocoordinates (§8).** The released CSV carries only `country` and a
   country-specific integer `plotid` — **no latitude/longitude, no CRS, no grid index**.
   There is no way to place the plots on the Sentinel-2 grid. The plot locations were not
   published with this record.

## Persistent-state recast considered and rejected

§5 permits recasting a *persistent* post-disturbance state (a completed clear-cut, a burn
scar) as presence/state classification with `change_time=null`. That escape does **not**
apply here: even setting timing aside, the total absence of coordinates makes it impossible
to place any label — presence or change — on imagery. The dataset cannot be salvaged in any
form from the released files.

## Reproduce

```bash
curl -sL "https://zenodo.org/api/records/3561925/files/disturbances.csv/content" -o disturbances.csv
head -1 disturbances.csv   # confirm columns: no lat/lon; timing is year_disturbance_*
```

No `metadata.json` / `locations/` outputs are written for a rejected dataset; only this
summary and `datasets/european_forest_disturbance_reference_senf_seidl/registry_entry.json`
(status `rejected`) are produced.
