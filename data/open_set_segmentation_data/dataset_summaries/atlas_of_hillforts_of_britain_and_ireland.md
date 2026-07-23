# Atlas of Hillforts of Britain and Ireland

- **Slug:** `atlas_of_hillforts_of_britain_and_ireland`
- **Status:** completed
- **Task type:** classification (weak point-presence)
- **Family:** heritage · **Region:** Britain & Ireland
- **License:** open for research

## Source & access

Public "Atlas of Hillforts of Britain and Ireland" (Univ. Edinburgh / Univ. Oxford,
AHRC-funded; 4,147 Iron/Bronze Age hillfort sites, launched 2017). Website:
https://hillforts.arch.ox.ac.uk/. No credential required. Data pulled from the official
public Esri Feature Service (owner `sjoh1738_oxforduni`):

```
https://services1.arcgis.com/PTDJItLzolyiewT6/arcgis/rest/services/Atlas_of_Hillforts/FeatureServer/0
```

All 4,147 point records were downloaded via paginated `query` requests (`f=geojson`,
`outSR=4326`, `resultRecordCount=2000`) into
`raw/atlas_of_hillforts_of_britain_and_ireland/hillforts.geojson`. Every record has a
non-null WGS84 `Longitude`/`Latitude` (matching the point geometry) within the
Britain & Ireland landmass (lon -10.46..1.44, lat 49.97..60.72), so all sites are
georeferenced on the S2 grid.

## Suitability assessment (why accepted, and as what)

Hillforts are large enclosed earthwork ramparts — typically 1-20+ ha, i.e. ~100 m to
>450 m across. At Sentinel-2 / Landsat 10-30 m the individual rampart lines are subtle,
but the overall enclosure footprint and its persistent topographic / vegetation
signature over the site are plausibly detectable. This does **not** support crisp
per-pixel polygon segmentation at 10 m (the atlas provides only a representative point
per site, not rampart polygons), so it is kept as a **weak single-phenomenon presence
label at the site point** — exactly the "points → points.json" path the manifest
(`label_type: points`) and spec §2a/§4 prescribe. Not rejected: the phenomenon is
observable at 10-30 m per the manifest note, coordinates are recoverable, and the license
permits research use.

## Class mapping

Two manifest classes are recovered from the expert `Reliability of Interpretation` field:

| id | class | source value | count (all / selected) |
|----|-------|--------------|------------------------|
| 0  | hillfort          | `Confirmed`   | 3354 / 1000 |
| 1  | possible hillfort | `Unconfirmed` | 722 / 722   |

`Irreconciled issues` (71 records, conflicting source data) dropped as ambiguous. This is
a **presence-only** dataset: there is no explicit negative/background class.

## Sampling, time range

- Point-only → one dataset-wide `points.json` (spec §2a); no per-point GeoTIFFs.
- `balance_by_class(per_class=1000)` → 1000 hillfort + 722 possible hillfort = **1722**
  samples (well under the 25k cap). Confirmed class truncated from 3354 → 1000 (rare class
  `possible hillfort` kept in full).
- Persistent/static heritage sites → fixed representative 1-year window **2020-01-01 …
  2021-01-01** (Sentinel era) for every point. No change labels.

## Caveats

- Weak label: the point marks site presence, not a pixel-exact rampart mask. Small (~1 ha)
  hillforts are near the 10 m resolution limit; pretraining should treat this as a coarse
  presence signal.
- Presence-only (no negatives); the `hillfort` vs `possible hillfort` split reflects
  archaeological confidence, not a visual land-cover distinction.
- A per-pixel Sentinel-2 overlay eyeball check is not meaningful for subtle heritage
  points; sanity was instead confirmed by checking all coordinates fall on the GB&I
  landmass and match the authoritative atlas lon/lat.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.atlas_of_hillforts_of_britain_and_ireland
```

Idempotent: re-running reuses the cached `raw/.../hillforts.geojson` and rewrites the
table/metadata.
