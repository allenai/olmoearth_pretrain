# Smithsonian Global Volcanism Program (GVP)

- **Slug:** `smithsonian_global_volcanism_program`
- **Status:** completed · classification (presence-only points, multi-class by volcano type) ·
  **2,349 points**
- **Source:** Smithsonian Institution, Global Volcanism Program — *Volcanoes of the World* (VOTW).
  <https://volcano.si.edu/> · **License:** free research use (cite GVP).
- **Annotation method:** manual (expert-curated volcano census).

## Source & access

Accessed programmatically (no credentials) via the GVP OGC WFS GeoServer endpoint
`https://webservices.volcano.si.edu/geoserver/GVP-VOTW/wfs`, pulling two point layers as GeoJSON
into `raw/{slug}/`: `Smithsonian_VOTW_Holocene_Volcanoes` (1,196) and
`Smithsonian_VOTW_Pleistocene_Volcanoes` (1,451). Each feature is one point at the volcano summit
with `Primary_Volcano_Type`.

## Label type — presence-only points

**Converted from the old per-detection object-detection tile encoding** (32×32 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each summit is one point carrying its volcano-type class id. There is **no fabricated GeoTIFF
context, and no background / buffer / negative tiles** — this dataset carries **no fabricated
negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts

Class = `Primary_Volcano_Type`; `Unknown`/`None` types dropped. **19 classes**, ids assigned by
descending frequency, capped at 1000/class. **2,349 points total.**

| id | type | pts | id | type | pts |
|----|------|-----|----|------|-----|
| 0 | Stratovolcano | 1000 | 10 | Maar | 22 |
| 1 | Shield | 345 | 11 | Tuff cone | 11 |
| 2 | Volcanic field | 313 | 12 | Tuya | 11 |
| 3 | Caldera | 151 | 13 | Lava cone | 9 |
| 4 | Pyroclastic cone | 146 | 14 | Explosion crater | 7 |
| 5 | Lava dome | 120 | 15 | Crater rows | 5 |
| 6 | Fissure vent | 75 | 16 | Volcanic remnant | 3 |
| 7 | Complex | 73 | 17 | Tuff ring | 2 |
| 8 | Cone | 28 | 18 | Pyroclastic shield | 1 |
| 9 | Compound | 27 | | | |

Sparse classes kept per spec §5 (downstream assembly filters too-small classes).

## Time handling

Persistent landforms → 1-year Sentinel-era window spread across **2016–2024**;
`change_time = null`. No dated-eruption change label: GVP eruption dates are year-resolved at best
(often historical/BCE), coarser than the ~1–2 month change-label bar.

## Output

- `datasets/smithsonian_global_volcanism_program/points.geojson`
- `datasets/smithsonian_global_volcanism_program/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.smithsonian_global_volcanism_program
```

Idempotent; re-downloads the two WFS layers only if missing.

## Caveats

- A summit point is a weak proxy for the whole edifice, and volcano type is a full-edifice
  morphological property — treated as best-effort presence + type.
- Pleistocene edifices included alongside Holocene (more eroded but still large landforms).
