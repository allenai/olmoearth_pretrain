# WRI/DeepMind Global Drivers of Forest Loss

- **Slug:** `wri_deepmind_global_drivers_of_forest_loss`
- **Status:** completed
- **Task type:** classification (sparse points → `points.geojson`, spec §2a)
- **Num samples:** 6504
- **Source:** Zenodo record [15366671](https://zenodo.org/records/15366671) —
  "Global drivers of forest loss at 1 km resolution" (Sims et al., WRI + Google DeepMind;
  ERL). License **CC-BY-4.0**.

## Source

The Zenodo record ships a 1 km derived-product raster
(`drivers_forest_loss_1km_2001_2024_v1_2.tif`, 295 MB) **and** the photointerpreted
interpreter label points used to train/validate the CNN:

- `training_2001_2024_v1_2.geojson` — 8193 points
- `validation_2001_2022.geojson` — 3574 points

Per the manifest note ("Prefer the interpreter train/validation points over the 1 km
raster") and spec §1 (prefer reference data over derived maps), **we use the points, not
the raster.** Only the two point GeoJSONs were downloaded to
`raw/wri_deepmind_global_drivers_of_forest_loss/`; the 1 km raster was skipped.

Each point has WGS84 `Longitude`/`Latitude`, a `Driver_primary_code` (1–8), a
`Driver_primary` name, and `Confidence_primary` (High/Medium/Low) plus `Region`.

## Access method

`download.download_zenodo("15366671", raw, filenames=[...])` (public record, no
credentials). Label-only: the 1 km GeoTIFF was not pulled.

## Class mapping

The manifest's 7 classes are exactly source `Driver_primary_code` 1–7, in order. Mapped
to ids 0–6:

| id | name | source code | selected count | available |
|----|------|-------------|----------------|-----------|
| 0 | Permanent agriculture | 1 | 1000 | 3252 |
| 1 | Hard commodities | 2 | 657 | 657 |
| 2 | Shifting cultivation | 3 | 919 | 919 |
| 3 | Logging | 4 | 1000 | 2665 |
| 4 | Wildfire | 5 | 1000 | 2018 |
| 5 | Settlements & infrastructure | 6 | 936 | 936 |
| 6 | Other natural disturbances | 7 | 992 | 992 |

**Code 8 ("Noise/non-forest", 328 pts, training-only) is dropped** — it is an
annotation-quality flag (point turned out to be non-forest / mislabeled), not a semantic
driver class, and it does not appear in the validation set or the manifest's 7-class
scheme.

Both source splits are combined and used (spec §5 — all splits are fair game). Selection
is `balance_by_class(..., per_class=1000, total_cap=25000)`: 7 classes × 1000 = 7000 <
25k, so no cap reduction; four classes have <1000 points and are kept in full. Per-class
descriptions in `metadata.json` follow the WRI/Curtis-et-al. driver taxonomy as extended
by the DeepMind CNN.

Per-point `confidence`, `region`, and `driver_name` are copied into each feature's
`properties` as auxiliary fields so downstream can filter (e.g. High-confidence only) if
desired.

## Time range and change handling

The points are **not dated to a loss year** in their properties (the product attributes
tree-cover loss over the whole 2001–2024 span). Following the task instruction and spec
§5, the driver is treated as a **persistent land-use state**: each point gets a **static
1-year Sentinel-era window (2020-01-01 → 2021-01-01)** with `change_time=null`. It is
**not** encoded as a change label (spec §5 forbids change labels whose date is only
year/coarser-resolved).

**Caveat.** Permanent agriculture, hard commodities, shifting cultivation, and
settlements are genuinely persistent land uses, so a static recent window is well aligned.
Wildfire, logging, and other-natural-disturbance are more event-like: if the loss occurred
early in the record the post-disturbance state may have partially recovered by 2020, so the
2020 imagery may not clearly show the driver at those points. Since no per-point dates are
available, this is unavoidable; downstream users can restrict to persistent-driver classes
or to High-confidence points via the auxiliary properties if needed.

## Tile size

N/A — pure sparse points, written 1×1 to the dataset-wide `points.geojson` (no per-sample
GeoTIFFs, spec §2a).

## Verification (§9)

- `points.geojson`: `FeatureCollection`, `task_type=classification`, `count=6504`; every
  feature is a `Point` with valid WGS84 coords, a 1-year `time_range`, and
  `change_time=null`.
- Label ids present {0..6} exactly match `metadata.json` class ids {0..6}.
- Class-balance counts (above) all ≤1000.
- Spatial sanity: coordinates validated in-range and consistent with the recorded
  `Region`; labels are photointerpreted reference points (no per-pixel mask to overlay).
- Idempotent: re-running overwrites `points.geojson`/`metadata.json` atomically with a
  seeded, deterministic selection.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.wri_deepmind_global_drivers_of_forest_loss
```
