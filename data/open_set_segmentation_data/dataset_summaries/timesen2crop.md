# TimeSen2Crop — REJECTED (no recoverable geocoordinates)

- **Slug**: `timesen2crop`
- **Name**: TimeSen2Crop
- **Source**: Hugging Face `monster-monash/TimeSen2Crop` (MONSTER reformat) /
  IEEE JSTARS 2021 (Weikmann, Paris & Bruzzone); original release Zenodo record 4715631
- **URL**: https://huggingface.co/datasets/monster-monash/TimeSen2Crop
- **Family / region**: crop_type / Austria
- **Label type (manifest)**: points (pixel time series), farmer-declaration labels
- **Classes (manifest)**: 16 crop types (15 in the MONSTER version; see below)
- **Time range**: 2017–2018 agronomic year (Sep 2017 – Aug 2018)
- **License**: CC BY 4.0 (open, research)
- **Status**: **rejected**
- **Rejection reason**: `no-geocoordinates` — the pixel time series carry no lon/lat and
  no within-tile pixel index, so samples cannot be placed on the Sentinel-2 10 m grid.

## What TimeSen2Crop is

A pixel-based crop-type dataset of 1,135,511 Sentinel-2 time series covering all of
Austria over the 2017–2018 agronomic year. Each sample is a single 10 m pixel's
multispectral temporal signature (9 bands, daily-interpolated to length 365 in the
MONSTER version) with a crop-type label. Extracted from the 15 Sentinel-2 MGRS tiles that
cover Austria (plus one 2019 tile in the original). Labels come from farmer declarations.
It is built for time-series *classification*, not for georeferenced segmentation.

## Why it is rejected

The open-set-segmentation pipeline pairs each label with Sentinel-2 / Sentinel-1 /
Landsat imagery by geography and time, so every sample needs real-world coordinates (or an
S2 tile + within-tile pixel row/col from which lon/lat can be recovered). TimeSen2Crop
provides neither, in either distribution:

1. **HF MONSTER version** ships three arrays:
   - `TimeSen2Crop_X.npy` — N × 9 × 365 spectral time series,
   - `TimeSen2Crop_y.npy` — one int64 crop class id per sample (values 0–14, 15 classes;
     the "other crops" class was dropped),
   - `TimeSen2Crop_meta.npy` — a single int64 per sample in 0–14, i.e. **which of the 15
     S2 MGRS tiles** the pixel came from.

   The tile id localizes a pixel only to a ~110 × 110 km tile. There is **no within-tile
   pixel index** and no lon/lat. Verified directly: `meta.shape == (1135511,)`,
   `dtype=int64`, `unique == [0..14]` (a flat tile id, not structured coordinates).

2. **Original Zenodo release (record 4715631)** is organized hierarchically as
   `<tile>/<crop_class>/<n>.csv`, where each CSV is one pixel's temporal signature (rows =
   acquisition dates; columns = B2, B3, B4, B5, B6, B7, B8A, B11, B12 + a
   clear/cloud/shadow/snow flag), plus a per-tile `dates.csv`. Samples are **anonymized
   running-index CSVs** (`0.csv … N.csv`); the official `TimeSen2Crop_Description.pdf`
   documents no coordinate field and no pixel row/col. So the original does not recover
   coordinates either.

Because no per-pixel geolocation exists, the pixel time series cannot be reprojected to a
local-UTM 10 m pixel and placed on the S2 grid. This is the spec's stated rejection
condition for pixel-time-series points without recoverable geocoordinates.

## Label distribution (for the record)

MONSTER `y.npy` class counts (class id → count), 15 classes, 1,135,511 samples total:

```
0: 7951      1: 283263    2: 164316    3: 30678     4: 22787
5: 70884     6: 94061     7: 1472      8: 53694     9: 41901
10: 34064    11: 85353    12: 132327   13: 66448    14: 46312
```

(The MONSTER release does not ship an explicit id→name map; the source crop types are
Legumes, Grassland, Maize, Potato, Sunflower, Soy, Winter Barley, Winter Caraway, Rye,
Rapeseed, Beet, Spring Cereals, Winter Wheat, Winter Triticale, Permanent Plantation,
Other Crops — the last dropped in MONSTER.)

## Access method (for the record)

Public, no credentials needed. `download.hf_download("monster-monash/TimeSen2Crop", ...)`
fetches the metadata artifacts (README, loader, `*_meta.npy`, `*_y.npy`) to
`raw/timesen2crop/`. The multi-GB `TimeSen2Crop_X.npy` was not needed. Nothing was written
under weka `datasets/` (rejection path); the registry was left untouched.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.timesen2crop
```

Downloads the small metadata artifacts, re-verifies the absence of coordinates, and prints
the rejection. Makes no dataset outputs. Idempotent.
