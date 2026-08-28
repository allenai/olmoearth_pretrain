# DeepOWT (Global Offshore Wind Turbines)

- **Slug:** `deepowt_global_offshore_wind_turbines`
- **Status:** completed · classification (presence-only points) · **1,615 points**
- **Family / region:** wind · global offshore · **License:** CC-BY-4.0
- **Annotation method:** derived product (deep learning on Sentinel-1 time series) + validation.

## Source & access

DeepOWT, Zhang et al., *Earth System Science Data* (ESSD), on Zenodo
(<https://doi.org/10.5281/zenodo.5933967>, CC-BY-4.0). Global inventory of offshore
wind-energy infrastructure with per-quarter deployment status (`Y2016Q3 … Y2021Q2`, status
0=open sea, 1=under construction, 2=turbine, 3=substation). Direct unauthenticated HTTP
download of `DeepOWT.geojson` (4.1 MB) → `raw/deepowt_global_offshore_wind_turbines/`. No
imagery pulled.

## Label type — presence-only points

**Converted from the old positive-only object-detection tile encoding** (32×32 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each selected offshore wind structure is one point whose class is the object type. There is
**no fabricated GeoTIFF context, and no background / buffer / negative tiles** — the open-sea
DeepOWT background (status 0) is dropped entirely. Negatives are supplied downstream by the
assembly step from other datasets, so this dataset carries **no fabricated negatives**.

## Classes / counts

Real object classes only (DeepOWT status → class id):

| id | name | points |
|----|------|--------|
| 0 | under_construction | 448 |
| 1 | offshore_turbine | 1000 |
| 2 | substation | 167 |

Up to 1000 points/class (`balance_by_class`). `under_construction` and `substation` are sparse
and kept in full (downstream assembly filters too-small classes). **1,615 points total.**

## Time handling

Persistent-structure model: a point is emitted only for a **full calendar year (2017–2020)** in
which all four quarters equal that class, so the state is persistent across the 1-year
`time_range`. `change_time = null`. Quarterly (~3-month) timing is coarser than the ~1–2 month
change-label bar, so no dated change labels.

## Output

- `datasets/deepowt_global_offshore_wind_turbines/points.geojson`
- `datasets/deepowt_global_offshore_wind_turbines/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.deepowt_global_offshore_wind_turbines
```

## Caveats

- Derived product (DL on Sentinel-1) with validation, not in-situ reference.
- Quarterly timing forces the persistent-structure model (no dated-change labels).
