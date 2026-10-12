# OlmoEarth Ethiopia maize — REJECTED (needs-credential / needs-data)

- **Slug**: `olmoearth_ethiopia_maize`
- **Manifest name**: `OlmoEarth Ethiopia maize`
- **Status**: `rejected` (`needs-credential`: raw source data not present locally)
- **Task type (intended)**: classification (binary `maize` / `non_maize`, sparse points)
- **Manifest url**: `olmoearth_projects/projects/ethiopia_maize`
- **License**: internal

## What the source is

`OlmoEarth Ethiopia maize` is an **olmoearth_projects project**, not a materialized
rslearn dataset. Its full checkout on disk
(`olmoearth_projects/olmoearth_projects/projects/ethiopia_maize`) contains
**only three files**:

- `prepare_training_points.py` — a label-prep recipe that reads raw survey files from a
  local `ethiopia_labels/` directory and writes a combined `labels_ess_rdm.geojson`
  (columns `geometry`, `year`, `maize_or_not`, `start_time`, `end_time`).
- `unified_config.yaml` — OlmoEarth finetune/inference config. Confirms the label field is
  `maize_or_not` with a 2-class segmentation legend (`maize`=0, `non_maize`=1, nodata=2),
  point annotations buffered by 31 px, Feb–Dec (11 × 30-day) growing-season temporality.
- `README.md` — reports **24,673 instances**, label balance ≈ 48.4% maize / 51.6%
  non_maize.

The recipe's inputs (`ethiopia_labels/`) are:
1. **Ethiopia Statistical Service (ESS) 2022 field-survey shapefiles** — the bulk of the
   dataset: `SelectedDistrictsForTestinginAOI/{SelectedMaize2022ESS, SelectedTeff2022ESS,
   SelectedWheat2022ESS}.shp`, `NonCropin HighMaize Production Woredas/`,
   `5CropsCleaned in High Maize Production Woredas/{MaizeHPW_FV_Edited, CheckPeas_Cleaned2022HMPZ,
   Sorghum_Cleaned2022HMPZ, Teff_Cleaned2022HMPZ, Wheat_Cleaned2022HMPZ}.shp`,
   `MaizeandNonMaizeSelectedAOI`, `maize_data_with_gps.csv`. Teff/wheat/sorghum/peas/non-crop
   are folded into `non_maize`; maize files → `maize`.
2. **WorldCereal RDM parquet files** (`rdm/`): `2018_eth_faowapor1_poly_111_dataset.parquet`,
   `2018_eth_faowapor2_poly_111_dataset.parquet`, `2020_eth_ethct2020_point_110_dataset.parquet`,
   `2020_eth_nhicropharvest_poly_100_dataset.parquet` (RDM `sampling_ewoc_code == maize` → maize,
   else non_maize; `cropland_unspecified`/`temporary_crops` dropped). Years 2018 and 2020.

## Why it was rejected

Despite `have_locally: true` in the manifest, **none of the raw source inputs are present
on disk**, and **no materialized rslearn dataset exists**. Verified by searching:

- `olmoearth_projects/.../ethiopia_maize` — only the 3 recipe files (git-tracked);
  no `ethiopia_labels/`, no output geojson.
- `/weka/dfive-default/rslearn-eai/datasets/crop/` — has `togo_2020`, `nigeria_maize`,
  `kenya_maize_cropland`, etc., but **no `ethiopia_maize`** (contrast: the sibling
  `olmoearth_togo_cropland` project *did* have a materialized dataset at `crop/togo_2020`).
- `/weka/dfive-default/rslearn-eai/datasets/olmoearth_evals/` — has `ethiopia_crops` (the
  *different* wheat/barley/maize/teff sibling), but no ethiopia maize-vs-non-maize dataset.
- `olmoearth_run_data/` — no `ethiopia_maize` project.
- Filesystem searches for `ethiopia_labels`, `labels_ess_rdm.geojson`, and the distinctive
  ESS shapefile/RDM parquet names under `/weka/dfive-default` and the local data root — no hits.

The ESS shapefiles are **internal Ethiopia government survey data** (not fetchable
unauthenticated); the WorldCereal RDM parquets require the **rdm.esa-worldcereal.org**
portal. This is therefore a `needs-credential` / needs-pre-downloaded-copy rejection per
AGENT_SUMMARY §2/§1a, not a transient failure. Nothing was written to weka `datasets/`
beyond the required `registry_entry.json` (and a provenance `raw/<slug>/SOURCE.txt`).

## Intended processing (for whoever supplies the data)

Once `ethiopia_labels/` (or a materialized rslearn dataset) is on disk, this is a
straightforward **sparse-point classification** dataset that mirrors
`olmoearth_ethiopia_crops.py` / `olmoearth_togo_cropland.py`:

- Classes: `maize`=0, `non_maize`=1 (uint8; nodata 255). ~24.7k roughly-balanced points.
- Points → single dataset-wide `points.geojson` via `io.write_points_table(slug,
  "classification", points)` (spec §2a) — **no per-point GeoTIFFs**.
- Balance to ≤1000/class with `sampling.balance_by_class` (well under the 25k cap; with only
  2 classes ≈2000 points selected).
- Time range: per-point 2022 (or the record's 2018/2020 RDM year) growing-season window,
  Feb 1 – Dec 30 (≤1 year), taken from the recipe's `start_time`/`end_time`, or
  `io.year_range(year)` as a fallback.
- Geometry: recipe geometries are WGS84 points (polygons → centroid); write `[lon, lat]`
  directly. If instead a materialized rslearn dataset is provided, read window
  `metadata.json` `options`/`bounds` like the sibling scripts.

## Reproduce

```
# 1. Place the raw survey data at:
#    olmoearth_projects/olmoearth_projects/projects/ethiopia_maize/ethiopia_labels/
#    (ESS 2022 shapefiles + rdm/*.parquet), then run the project recipe to build the geojson:
cd olmoearth_projects/olmoearth_projects/projects/ethiopia_maize
python3 prepare_training_points.py    # -> ethiopia_labels/labels_ess_rdm.geojson
# 2. Then add datasets/olmoearth_ethiopia_maize.py (mirroring olmoearth_ethiopia_crops.py,
#    reading the geojson) and run:
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_ethiopia_maize
```

## Caveats / judgment calls

- **Manifest `have_locally: true` is inaccurate** for this entry — the project ships only a
  recipe, not the data. Flag for the manifest owner.
- No fabricated negatives / no dropped classes were needed (dataset not processed).
- If only the RDM parquets become available (not the ESS shapefiles), the reconstruction
  would be a small, non-representative subset (the 24.7k-instance figure is ESS-dominated,
  2022) — prefer to wait for the full ESS drop before processing.
