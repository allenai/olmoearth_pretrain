# OlmoEarth Togo cropland

- **Slug**: `olmoearth_togo_cropland`
- **Task type**: classification (sparse points -> `points.json`, spec §2a)
- **Family / region**: cropland / Togo
- **License / source**: internal / `olmoearth`
- **Samples**: 1582 (non_crop 588, crop 994)

## Source

Local rslearn dataset materialized from the `togo_cropland` olmoearth_projects project
(manifest `url`: `olmoearth_projects/projects/togo_cropland`). The materialized dataset
lives on weka at:

```
/weka/dfive-default/rslearn-eai/datasets/crop/togo_2020/20260127
```

Underlying labels are field-collected crop / non-crop points for Togo from
[nasaharvest/togo-crop-mask](https://github.com/nasaharvest/togo-crop-mask)
(Zenodo record 3836629). `raw/olmoearth_togo_cropland/SOURCE.txt` records these pointers
(nothing copied, per `have_locally: true`).

## Layout inspected

The project's `projects/togo_cropland` dir only holds the window-creation scripts
(`create_windows_for_lulc.py`, `create_label_raster.py`); the actual rslearn windows are
under the materialized dataset above: `windows/togo_cropland/{name}/metadata.json`. Each
window is one label point buffered to a 32x32 window (EPSG:32631 UTM, 10 m). Relevant
fields per window `metadata.json`:

- `options.lulc_category` -> `"crop"` or `"non_crop"` (the class)
- `options.split` -> `train` / `val` / `test`
- `time_range` -> `2019-02-01 .. 2019-09-30` for every window (growing season, < 1 yr)
- `projection` + `bounds` -> point location = center of the window bounds

There is also a `cropland_label` vector layer whose feature is the full-window polygon
with the same `category`; the class was taken from `options.lulc_category` (equivalent,
cleaner).

## Processing decisions

- **Point-only dataset** -> single dataset-wide `points.json` (no per-point GeoTIFFs).
- **Location**: WGS84 lon/lat computed from the center of each window's pixel `bounds`
  under its UTM projection. Verified against the `lat_lon_...` window names (match to ~1e-5
  deg). Resulting bbox lon [-0.17, 1.76], lat [6.22, 11.16] — Togo.
- **Classes** (manifest order): `non_crop` = 0, `crop` = 1.
- **Time range**: the source window range `2019-02-01..2019-09-30` used verbatim per point
  (already < 1 year). No change labels.
- **Splits**: all three (train/val/test) used, per spec §5. Source split kept in
  `source_id` implicitly via window name.
- **Balancing**: `balance_by_class(per_class=1000)` — both classes are under 1000 and the
  total (1582) is far under the 25k cap, so all points are kept.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_togo_cropland
```

Idempotent: rewrites `points.json` / `metadata.json` / `registry_entry.json` each run.

## Caveats

- Point labels only; each is a single 10 m pixel (pretraining projects lon/lat onto the S2
  grid). The 32x32 buffer used for the finetuning project is not carried over.
- All labels are 2019 growing season; the manifest's `[2019, 2023]` range reflects intended
  applicability, but the actual field data is 2019.
