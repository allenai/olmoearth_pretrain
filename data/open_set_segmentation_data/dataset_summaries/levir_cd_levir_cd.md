# LEVIR-CD / LEVIR-CD+ (`levir_cd_levir_cd`) — REJECTED

- **Source:** GitHub / Remote Sensing — https://justchenhao.github.io/LEVIR/ (manifest url https://chenhao.in/LEVIR/)
- **Family:** change_detection · **label_type:** dense_raster · **have_locally:** false
- **Final status:** `rejected`
- **Primary reason (notes):** `change-timing: event not resolvable to within ~1-2 months (bitemporal pairs 5-14 years apart, no dated change); also no recoverable per-patch geocoordinates (PNG Google Earth tiles)`

## What the dataset is
LEVIR-CD is a binary building-change-detection benchmark: 637 very-high-resolution
(0.5 m/pixel) Google Earth bitemporal image **pairs**, each 1024×1024 px, with manually
annotated building-change masks (1 = change, 0 = no-change; 31,333 change-building
instances). LEVIR-CD+ extends it. Imagery is from 20 regions in several Texas cities
(Austin, Lakeway, Bee Cave, Buda, Kyle, Manor, Pflugerville, Dripping Springs).

## Why rejected (triage, no download performed)
Two independent, decisive blockers — both checked cheaply from the datasheet/releases per
SOP §8 (no multi-GB archive was downloaded):

1. **Change timing not resolvable (§5 hard rule — primary reason).** Each pair's two
   acquisitions span **5 to 14 years apart**, with capture dates ranging **2002–2018**.
   The building change is only known to have occurred *somewhere within* that multi-year
   gap; there is no per-sample dated event. Per §5, a change label is only usable if the
   change date is known to within ~1–2 months so the event can be placed confidently
   inside the pretraining pairing window. This is exactly the "multi-year pre/post
   comparison" case §5 rejects (same ground as `oscd` and
   `olmoearth_land_cover_change`). It cannot be recast as a persistent presence/state
   label because there is no single dated post-change state to anchor a 1-year window on.

2. **No recoverable geocoordinates (§8).** The dataset is distributed as PNG image
   patches (train/val/test `*.png`) exported from Google Earth. Releases (project page,
   Kaggle, HuggingFace `blanchon/LEVIR_CDPlus`) carry only coarse region names, not
   per-patch lon/lat or a CRS, so labels cannot be placed on the S2 grid. A region/city
   id without within-tile pixel geolocation is insufficient (§8).

3. **Secondary:** the phenomenon (individual buildings — villas, small garages,
   warehouses — at 0.5 m VHR) is largely unresolvable at 10 m S2/S1/Landsat; and imagery
   is partly pre-2016.

Any one of (1) or (2) is sufficient for rejection; together they make the dataset
unusable for this pipeline as distributed.

## Reproduce
Triage only — no data written to weka `datasets/` beyond `registry_entry.json`. To
re-triage: read the datasheet at https://justchenhao.github.io/LEVIR/ and confirm the
5–14 year bitemporal gap and PNG (non-georeferenced) distribution before any download.
