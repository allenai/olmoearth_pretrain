# Drought Watch (Kenya forage condition)

- **Slug**: `drought_watch_kenya_forage_condition`
- **Status**: **rejected** — `notes: "no recoverable geocoordinates: public TFRecords + extra_data.csv strip lon/lat"`
- **Family / region**: rangeland / Northern Kenya rangeland
- **Source**: ILRI / Cornell / UC San Diego, released as the Weights & Biases "Drought Watch"
  benchmark. Repo https://github.com/wandb/droughtwatch ; paper
  *Satellite-based Prediction of Forage Conditions for Livestock in Northern Kenya*
  (arXiv:2004.04081).
- **License**: open benchmark (freely downloadable — NOT a credential/access problem).
- **time_range**: 2016–2019 (post-2016 — the Sentinel-era rule is satisfied; not the reason for rejection).

## What the dataset is

~107,869 ground observations of **forage / drought condition** in Northern Kenya. Expert
pastoralists were asked, via an ODK mobile form, how many cows the forage within ~20 m of
their standing location could feed for one day — an **ordinal label 0, 1, 2, or 3+ cows**
("carrying capacity"). Each observation is paired with a **65×65-pixel, 10-band Landsat-8
patch** (30 m/pixel, ~1.95 km across) centered on the observation. Distributed as TFRecords
(`dw_train_86K_val_10K.zip`, ~4.3 GB; 86,317 train + 10,778 val + 10,774 held-out) plus an
`extra_data.csv` (112,674 ODK form rows) in the repo.

This is a genuinely attractive label signal: real in-situ, expert, per-point rangeland/forage
condition, post-2016, and observable at 10–30 m. It would map cleanly to either a 4-class
ordinal classification or a regression on carrying capacity. **The only problem is placement.**

## Why rejected — no recoverable geocoordinates (spec §8 / §2)

The observation locations were **deliberately removed** from every public artifact, so labels
cannot be placed on the Sentinel-2 / Landsat grid. Verified cheaply (schema + CSV header only,
**without** downloading the 4.3 GB archive, per spec §8):

- **TFRecords** (`train.py` feature schema): each example holds only band rasters
  `B1`…`B11` (`tf.string`) and an integer `label` (`tf.int64`). There is **no CRS, no
  geotransform, no lon/lat, no MGRS/tile id** — the patches are pure coordinate-free tensors
  (torch/tf models treat them as non-geo images).
- **`extra_data.csv`** (the ODK form export) contains the survey answers
  (`Screen1a…Screen3c`, `CarryingCapacity`, `water_points`), timestamps, `device_id`, `image`
  filename, and — tellingly — only `my_geopoint-Altitude` and `my_geopoint-Accuracy`. An ODK
  `geopoint` normally emits Latitude, Longitude, Altitude, Accuracy; **Latitude and Longitude
  were stripped** (privacy of pastoralist locations), leaving altitude/accuracy behind. Full
  column list: `SubmissionDate, meta-instanceID, start, end, device_id,
  my_geopoint-Altitude, my_geopoint-Accuracy, image, Screen1a..Screen3c, CarryingCapacity,
  water_points, Screen8, KEY`. No coordinate column anywhere.

With neither the tensors nor the CSV carrying lon/lat (and no per-sample geotransform to
recover it from), the labels cannot be co-located with pretraining imagery. This is the
spec's common, fast **"no recoverable geocoordinates"** rejection for ML-ready tensor/anonymized-CSV
releases. Not a credentials issue (data is public) and not a transient/infra failure.

## If coordinates ever become available

The dataset would be a good **point** dataset: one `points.geojson` feature per observation
(`properties.label` = ordinal cow-count class 0/1/2/3, or a regression value), a short
seasonal/1-year `time_range` anchored on each observation's `SubmissionDate` (the ODK
`start`/`end`/`SubmissionDate` timestamps are present in the CSV and give the assessment
period), no per-sample GeoTIFFs (§2a). Re-evaluate if ILRI/Cornell/UCSD publish the
un-redacted geopoints or a georeferenced version.

## Reproduce (verification of the rejection)

```
git clone --depth 1 https://github.com/wandb/droughtwatch.git
# TFRecord schema (bands + label only): see train.py `features = {...}`
grep -n FixedLenFeature train.py
# CSV columns (no lat/lon):
head -1 extra_data.csv | tr ',' '\n'
```

Nothing was written to weka `datasets/` except the per-dataset `registry_entry.json`
(status `rejected`).
