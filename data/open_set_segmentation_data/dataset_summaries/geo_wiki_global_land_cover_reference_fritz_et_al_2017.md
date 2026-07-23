# geo_wiki_global_land_cover_reference_fritz_et_al_2017

**Status:** completed · classification · 10,000 samples (point table)

## Source
PANGAEA doi [10.1594/PANGAEA.869680](https://doi.pangaea.de/10.1594/PANGAEA.869680) —
Fritz et al. 2017, *"A global dataset of crowdsourced land cover and land use reference
data (2011–2012)"* (Sci Data). Specifically the **Global Crowd** campaign file
`globalcrowd.csv` (open, direct download; CC-BY-3.0). No credentials required.

Each row is one crowdsourced visual interpretation of a location by one volunteer:
`lon`/`lat`, up to three land-cover classes (`LC1/LC2/LC3`, source codes 1–10) with their
fractional cover (`perc1/perc2/perc3`), plus confidence, field size, and the imagery date
used (`googleimagedate`). 151,942 crowd records over **79,848 unique locations** (mean 1.9
volunteers/location, max 82).

## Access
- `download.download_http("https://store.pangaea.de/Publications/FritzS-etal_2016/globalcrowd.csv", ...)`
  (15 MB) → `raw/{slug}/globalcrowd.csv`; column dictionary `globalcrowd_metadata.xlsx`.
- Only the label table is pulled (no imagery); `raw/{slug}/SOURCE.txt` records the DOI/URLs.

## Processing
- **Label = dominant land-cover class.** Per crowd record the dominant code is the `LC`
  with the largest `perc` among the three (fallback `LC1`). Per location we take the
  **majority vote** of per-record dominant codes across volunteers, breaking ties by
  lowest class id.
- **Location key = rounded (lon, lat)** (6 dp). `pixelID` is *not* a stable location key in
  this file — the same `pixelID` (even within one competition) appears at wildly different
  coordinates (within-id lon std > 100°), so coordinates are the only reliable geokey.
- Source codes 1–10 → class ids 0–9 (`id = code − 1`), matching the manifest class order:
  tree cover, shrub cover, herbaceous/grassland, cultivated & managed,
  mosaic cultivated/natural, flooded/wetland, urban, snow & ice, barren, open water.
  Per-class definitions from the source codebook are in `metadata.json` `classes[].description`.
- Sparse point segmentation → **GeoJSON point table** (`points.geojson`, spec §2a), not
  per-point GeoTIFFs. Each feature: `{lon, lat, label=class_id, time_range, source_id,
  n_votes}` (`n_votes` = crowd votes at that location, kept for provenance).
- Balanced to **≤1000 per class** via `balance_by_class` (seeded); all 10 classes have
  ≥1,350 candidate locations, so every class reaches exactly 1,000 → 10,000 samples
  (well under the 25k cap).

## Time range (caveat)
The interpreted imagery pre-dates the Sentinel era: `googleimagedate` is populated for
~41% of records and clusters in **2003–2012** (median ≈ 2010), with essentially none
2016+. Land cover is a comparatively **static** class, so per spec §5 (static labels) every
point is assigned a **representative Sentinel-era 1-year window (2016-01-01 → 2017-01-01)**,
`change_time=null`. **Caveat:** because interpretation used ~2003–2012 imagery, a minority
of points may reflect land cover that has since changed (e.g. cropland↔natural, urban
expansion); the majority (tree/barren/water/snow) is stable across the gap. This dataset
was *not* rejected on the pre-2016 rule because the class is static and reusable at a
representative Sentinel window (spec §8).

## Output
- `datasets/{slug}/points.geojson` — 10,000 point features (FeatureCollection).
- `datasets/{slug}/metadata.json` — class map (with source definitions) + per-class counts.
- `raw/{slug}/globalcrowd.csv`, `globalcrowd_metadata.xlsx`, `SOURCE.txt`.

## Class counts (selected / candidate)
tree cover 1000/21,959 · shrub cover 1000/10,284 · herbaceous/grassland 1000/10,047 ·
cultivated & managed 1000/15,131 · mosaic cultivated/natural 1000/5,722 ·
flooded/wetland 1000/1,350 · urban 1000/3,036 · snow & ice 1000/1,930 · barren 1000/8,952 ·
open water 1000/1,437.

## Verification
- `points.geojson` is a valid FeatureCollection (`count=10000`, `task_type=classification`);
  all labels ∈ [0,9], all coords in valid WGS84 range, ids unique, `time_range` = 1 year.
- `metadata.json` class ids cover all label values present.
- Spatial plausibility (lightweight, no imagery loaded): per-class |latitude| is physically
  sensible — snow & ice mean |lat| 60° (median 67°) and wetland 55° (62°) skew boreal, while
  tropical classes (shrub/mosaic) sit at ~23°. Labels are not scrambled. A full S2 overlay
  eyeball was not run (headless; no imagery access wired here) — flagged for downstream.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.geo_wiki_global_land_cover_reference_fritz_et_al_2017
```
Idempotent (re-downloads only if missing; rewrites `points.geojson`/`metadata.json`).
Use `--skip-download` if `raw/{slug}/globalcrowd.csv` is already present.

## Notes
- 1×1 point labels carry no spatial context by design; paired with S2/S1/Landsat at
  pretraining time by lon/lat + time overlap.
- Crowdsourced (non-expert) interpretation; majority-vote aggregation mitigates individual
  error. `confidence_LC` and `n_votes` are available in the raw file for optional downstream
  confidence weighting.
- License CC-BY-3.0 (cite Fritz et al. 2017).
