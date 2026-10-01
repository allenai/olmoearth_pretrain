# GeoLifeCLEF / GeoPlant

- **Slug:** `geolifeclef_geoplant`
- **Status:** completed (classification, 50,800 samples)
- **Family / label_type:** species_occurrence / points -> 1x1 sparse point segmentation
- **Source:** GeoPlant (GeoLifeCLEF 2024, Pl@ntNet-INRIA / NeurIPS 2024 Datasets & Benchmarks)
- **License:** CC-BY

## Source and access

GeoPlant is the GeoLifeCLEF 2024 dataset of European plant-species occurrences. The label
records live in two tables: a **presence-only (PO)** table of ~5.08M opportunistic
observations (GBIF + Pl@ntNet + iNaturalist + national biodiversity agencies), and a
**presence-absence (PA)** table of ~90k standardized survey plots. We used the **PO**
table: `PresenceOnlyOccurrences/PO_metadata_train.csv`.

**Access is fully open — no Kaggle account required.** The task flagged a possible
`needs-credential: kaggle` rejection, but the Pl@ntNet Seafile mirror hosts the full
dataset publicly:
- Landing: https://github.com/plantnet/GeoPlant
- Seafile mirror: https://lab.plantnet.org/seafile/d/59325675470447b38add
- PO metadata resolves to a direct raw-download URL via the Seafile share API (the
  per-dataset script does this automatically); no auth token is sent.
- GBIF extraction DOI (2022-11-08): https://doi.org/10.15468/dl.4ysfh4

The 691 MB `PO_metadata_train.csv` is saved under `raw/geolifeclef_geoplant/` with a
`SOURCE.txt` pointer.

## PO record structure

Columns: `publisher, year, month, day, lat, lon, geoUncertaintyInM, taxonRank, date,
dayOfYear, speciesId, surveyId, region, county, district`. One row = one species observed
at one lon/lat on one date. `speciesId` is an **anonymized integer** (the source ships no
species-name lookup — see caveat). Coverage: Europe (lat 34.6-71.2, lon -10.5-34.6),
years **2017-2021** (all in the Sentinel era), 9,709 unique species. Geolocation
uncertainty is good: median 10 m, 90th pct 56 m, 99th pct 88 m.

## Suitability decision (accepted as a weak / contextual label)

Plant-species presence at a point is only **weakly** observable from 10-30 m S2/S1/Landsat,
and the taxonomy is huge (~10k species). We accepted it anyway, matching the manifest's
stated intent ("species presence points for habitat/context pretraining") and the spec's
explicit support for weak/contextual labels and huge taxonomies. The reasoning:

- Data is openly accessible (no credential gate).
- Coordinates are precise (median 10 m uncertainty) and fit the sparse-point -> 1x1 recipe
  exactly (one species label per point).
- PO chosen over PA because PA puts many species at one survey location, which cannot be
  encoded as a single-class 1x1 patch.

**Hard class-count constraint.** The label GeoTIFF is single-band **uint8** (ids 0..254,
255 = nodata), so at most ~254 distinct classes can be encoded, while the source has 9,709
species. We keep the **top 254 species by observation frequency** (each with >= 4,585
observations), assigning ids 0..253 in descending frequency order. The other **9,455 rarer
species are dropped**. This is a deliberate, documented restriction driven by the uint8
label spec; a full-taxonomy encoding is not representable here.

## Processing

- Filter: drop rows missing lat/lon/year/speciesId; drop `geoUncertaintyInM > 100 m`
  (~1% of rows; in practice 0 rows dropped here since the 691 MB file's values were all
  within range after NA handling). Both `SPECIES` and `SUBSPECIES` taxon ranks kept.
- Class set: top 254 speciesIds by count -> class ids 0..253.
- Sampling: `balance_by_class` with `per_class = 200` (seeded, shuffled). 254 x 200 =
  **50,800** samples, near the spec's ~50k total-tile cap. Every kept species had far more
  than 200 observations, so all 254 classes have exactly 200 samples (perfectly balanced).
- Tiles: **1x1** uint8, value = class id, local UTM at 10 m/pixel, nodata 255.
- Time range: **1-year** window anchored on each observation's year (2017-2021), via
  `io.year_range`.
- Provenance: `source_id = survey_{surveyId}` (the source occurrence id).

## Outputs

- `datasets/geolifeclef_geoplant/metadata.json` — 254 classes; each class entry carries
  `source_species_id` and `n_source_observations` alongside `id`/`name`
  (`species_{speciesId}`)/`n_samples`. Class descriptions are `null` (source anonymizes
  species names).
- `datasets/geolifeclef_geoplant/locations/{000000..050799}.tif` + `.json`.

## Verification

- 50,800 `.tif` and 50,800 `.json`; 1:1 pairing.
- Spot-checked tifs: single band, uint8, shape (1,1,1), UTM CRS (e.g. EPSG:32632/32631/
  32633), 10 m resolution, nodata 255, values within 0..253.
- JSON sidecars carry a 1-year `time_range` and matching `classes_present`.
- Class balance: 200 samples for every one of the 254 classes.

## Caveats

- **Weak label:** a single citizen-science plant observation is not directly inferable
  from a 10-30 m S2/S1/Landsat pixel; treat as habitat/biogeographic *context*, not a
  strong per-pixel target.
- **Anonymized classes:** speciesIds are integers with no shipped name mapping, so class
  names are `species_{id}`. The frequency-rank id ordering is stable and reproducible.
- **Truncated taxonomy:** 9,455 of 9,709 species dropped to fit the uint8 label; the kept
  254 are the most-observed species.
- Points-only positive labels; no explicit background/negative class (each 1x1 asserts
  "species X was observed in this pixel within the year").

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.geolifeclef_geoplant --workers 64
```

Idempotent: re-running skips already-written `{sample_id}.tif`. The PO CSV is downloaded
into `raw/` on first run (skipped if present).
