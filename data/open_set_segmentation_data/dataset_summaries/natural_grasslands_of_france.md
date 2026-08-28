# Natural Grasslands of France

- **Slug:** `natural_grasslands_of_france`
- **Status:** completed
- **Task type:** classification (sparse points)
- **num_samples:** 1770

## Source

Zenodo record [10.5281/zenodo.7895449](https://doi.org/10.5281/zenodo.7895449)
(Panhelleux, Rapinel & Hubert-Moy 2023, *Data in Brief* 109348),
"Natural grasslands across mainland France: a dataset including a 10 m raster and ground
reference points". License **CC-BY-4.0**.

Downloaded via `download.download_zenodo` (unauthenticated, public):
`grassland_ground_points.geojson` (1,770 field/aerial-verified ground reference points).

## Access / processing

- The record also ships a 10 m natural-grasslands **raster** (`natural_grasslands_2020.tif`,
  ~394 MB) derived from five annual (2016-2020) 10 m land-cover maps. Per the spec's
  "prefer in-situ reference over derived-product maps", the raster is **not used**; only the
  ground reference points are processed. The raster file was not downloaded.
- Points are in **EPSG:2154 (Lambert-93)**; reprojected to **WGS84** lon/lat with pyproj.
  Resulting extent (lon -4.90..9.55, lat 41.41..50.98) matches mainland France.
- Label field is `type`: "natural grassland" (compilation of field-based vegetation maps)
  vs "artificial grassland" (EU LPIS). No pixel-level footprint -> pure sparse-point
  classification -> written as one `points.json` (spec 2a), not per-point GeoTIFFs.

## Class mapping

| id | name | count |
|----|------|-------|
| 0 | natural grassland | 882 |
| 1 | artificial grassland | 888 |

Both classes fall under the 1000/class cap, so all 1,770 points are kept (none dropped).

## Time range

Labels are seasonal/annual grassland type. The source is a 2016-2020 compilation with **no
per-point year**, so each point is assigned a representative 1-year window anchored on
**2018** (`2018-01-01 .. 2019-01-01`), within the 2016-2020 period. No change labels.

## Outputs (on weka)

`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/natural_grasslands_of_france/`
- `points.json` (1,770 rows: id, lon, lat, label, time_range, source_id=`ID=<n>`)
- `metadata.json`
- `registry_entry.json` (status=completed)

Raw: `raw/natural_grasslands_of_france/grassland_ground_points.geojson`.

## Caveats

- Natural vs artificial grassland is a subtle distinction that may not always be separable
  from 10 m S2/S1/Landsat spectral+temporal signal, but both are genuine grassland-type
  reference labels suitable for open-set pretraining.
- No per-point acquisition year in the source; the 2018 anchor is an approximation.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.natural_grasslands_of_france
```
Idempotent: Zenodo download skips if present; re-running rewrites `points.json`/`metadata.json`.
