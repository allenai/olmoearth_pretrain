# Global Offshore Oil & Gas Platforms (OOGPs)

- **Slug:** `global_offshore_oil_gas_platforms`
- **Status:** completed · classification (presence-only points) · **1,000 points**
- **License:** CC-BY-4.0
- **Annotation method:** derived product (Sentinel-1 SAR) + validation.

## Source & access

"The Offshore Oil and Gas Platforms (OOGPs) dataset based on satellite data spanning 2017 to
2023", Zenodo record 18350974 (<https://doi.org/10.5281/zenodo.18350974>, CC-BY-4.0). Vector
inventory of offshore oil/gas platforms across six major basins (Gulf of Mexico, Persian Gulf,
North Sea, Caspian Sea, Gulf of Guinea, Gulf of Thailand). Only the 977 KB label archive
`OOGPs_v1.0.0.zip` is downloaded (no imagery); layer `platforms` in `OOGPs_all_v1.0.0.gpkg`
(9,334 Point features, EPSG:4326) carries the per-year presence field `Year_label`.

## Label type — presence-only points

**Converted from the old positive-only object-detection tile encoding** (32×32 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each selected platform is one point of the single foreground class. There is **no fabricated
GeoTIFF context, and no background / buffer / negative tiles** — this dataset carries **no
fabricated negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts

Single class `0 = offshore_oil_gas_platform`. **1,000 points** (up to 1000/class,
`balance_by_class`). Each physical platform contributes at most one point (random year drawn
from its `Year_label`) to avoid over-representing long-lived platforms.

## Time handling

Persistent-structure model: a point is emitted only for a calendar year listed in the platform's
`Year_label`, so it is present across the whole 1-year `time_range`. `change_time = null`
(`Year_label` is year-resolved and month-precision install dates cover only ~4% of platforms —
both coarser/sparser than the ~1–2 month change-label bar). All labels post-2016 (2017–2023).

## Output

- `datasets/global_offshore_oil_gas_platforms/points.geojson`
- `datasets/global_offshore_oil_gas_platforms/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_offshore_oil_gas_platforms
```

Idempotent. Downloads the Zenodo archive to `raw/global_offshore_oil_gas_platforms/`.

## Caveats

- Partially overlaps the GFW SAR fixed-infrastructure dataset
  (`global_fishing_watch_sar_fixed_infrastructure`) — both are Sentinel-1-derived offshore
  oil/gas detections; downstream assembly handles dedup.
