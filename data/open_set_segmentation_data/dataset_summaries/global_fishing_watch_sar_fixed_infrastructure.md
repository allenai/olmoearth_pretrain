# Global Fishing Watch SAR Fixed Infrastructure

- **Slug:** `global_fishing_watch_sar_fixed_infrastructure`
- **Status:** completed · classification (presence-only points) · **3,000 points**
- **License:** CC-BY-NC-4.0
- **Annotation method:** manual training + deep learning (Sentinel-1 SAR).

## Source & access

Global Fishing Watch, from Paolo et al. 2024, *Nature*, "Satellite mapping reveals extensive
industrial activity at sea". Uses the paper's public figshare analysis-data repository
(<https://doi.org/10.6084/m9.figshare.24309475>), downloading only the label file
`offshore_infrastructure_v20231106.csv.zip` (11.4 MB) → `raw/{slug}/`. No imagery downloaded.
The CSV holds 1,441,242 detection-months of offshore fixed infrastructure (2017–2021) on monthly
Sentinel-1 SAR composites, with `structure_id`, `composite_date`, `lat`, `lon`, `label`.

## Label type — presence-only points

**Converted from the old positive-only object-detection tile encoding** (32×32 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each selected structure is one point of its coarse object class. There is **no fabricated GeoTIFF
context, and no background / buffer / negative tiles** — this dataset carries **no fabricated
negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts

GFW confidence tiers folded into coarse classes:

| id | name | GFW source labels |
|----|------|-------------------|
| 0 | oil | oil, probable_oil, possible_oil, lake_maracaibo |
| 1 | wind | wind, probable_wind, possible_wind |
| 2 | other | unknown (piers, bridges, power lines, aquaculture, …) |

Up to 1000 points/class (`balance_by_class`) → oil 1000, wind 1000, other 1000, **3,000 total**.
The coarse label is 100% consistent within each structure-year.

## Time handling

Persistent-structure model: a point is emitted only for a calendar year (2017–2021) in which the
structure is detected ≥ 6 months spanning both the first and last quarter, so it is present
across the whole 1-year `time_range`. `change_time = null` (detection timing is monthly on
6-month composites, coarser than the ~1–2 month change-label bar). All labels post-2016.

## Output

- `datasets/global_fishing_watch_sar_fixed_infrastructure/points.geojson`
- `datasets/global_fishing_watch_sar_fixed_infrastructure/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_fishing_watch_sar_fixed_infrastructure
```

Idempotent. Raw label CSV in `raw/global_fishing_watch_sar_fixed_infrastructure/`.

## Caveats

- Coordinates are GFW model detections (>98% classification accuracy in the paper), not in-situ
  surveys; the persistence rule filters transient noise.
- Partially overlaps `global_offshore_oil_gas_platforms` (both SAR-derived offshore oil/gas);
  downstream assembly dedups.
