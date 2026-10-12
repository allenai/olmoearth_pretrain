# HydroWASTE (Global Wastewater Treatment Plants)

- **Slug:** `hydrowaste_global_wastewater_treatment_plants`
- **Status:** completed · classification (presence-only points) · **1,000 points**
- **Family / region:** industry / Global · **License:** CC-BY-4.0
- **Annotation method:** authoritative + modeled (national/regional WWTP registries geocoded and
  completed with auxiliary data).

## Source & access

HydroWASTE v1.0 (Ehalt Macedo et al. 2022, ESSD, <https://doi.org/10.5194/essd-14-559-2022>) — a
global database of 58,502 wastewater treatment plants. Data from figshare
(<https://doi.org/10.6084/m9.figshare.14847786.v1>, one 2.4 MB zip → `HydroWASTE_v10.csv`);
openly downloadable, no credentials (`https://ndownloader.figshare.com/files/31910714`). Points
placed at the reported plant location (`LAT_WWTP`/`LON_WWTP`). No imagery pulled.

## Label type — presence-only points

**Converted from the old positive-only object-detection tile encoding** (48×48 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each selected plant is one point of the single foreground class. There is **no fabricated GeoTIFF
context, and no background / buffer / negative tiles** — this dataset carries **no fabricated
negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts

Single class `0 = wastewater_treatment_plant`. **1,000 points** (up to 1000/class,
`balance_by_class`), restricted to well-located, built plants: `QUAL_LOC ∈ {1,2}` (> 50% located
accurately) and built `STATUS` (Projected / Proposed / Under Construction / Construction
Completed excluded).

## Time handling

Persistent, undated features → 1-year `time_range` at a representative Sentinel-era year (spread
across **2016–2022**). `change_time = null`.

## Output

- `datasets/hydrowaste_global_wastewater_treatment_plants/points.geojson`
- `datasets/hydrowaste_global_wastewater_treatment_plants/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.hydrowaste_global_wastewater_treatment_plants
```

Idempotent.

## Caveats

- A point marks presence at a geocoded location (mostly ~110 m precision), not a segmented plant
  footprint; QUAL_LOC 3/4 plants excluded.
- Only 1,000 of ~52k well-located plants used (per-class cap).
