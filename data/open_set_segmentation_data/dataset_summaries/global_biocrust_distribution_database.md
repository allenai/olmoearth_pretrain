# Global Biocrust Distribution Database — REJECTED

- **Slug:** `global_biocrust_distribution_database`
- **Manifest name:** Global Biocrust Distribution Database
- **Source:** SOIL (Copernicus) — Wang et al. (2024), "Advancing studies on global biocrust distribution", SOIL 10, 763–2024, <https://soil.copernicus.org/articles/10/763/2024/>
- **License:** CC-BY-4.0
- **Manifest label_type:** points (sparse biocrust presence/type/cover, ~3848 dryland entries)
- **Status:** rejected
- **Reason:** `needs-georeferencing` — no recoverable geocoordinates (spec §8)

## What the source actually is

The manifest describes a *global georeferenced* biocrust occurrence/cover point database
with dominant-group composition (classes: biocrust presence, cyanobacteria, lichen, moss,
mixed). The only publicly accessible copy of the database is the article **supplement ZIP**
(<https://soil.copernicus.org/articles/10/763/2024/soil-10-763-2024-supplement.zip>, 405 KB),
which contains `biocrust_database.xlsx` and a title-page PDF.

I downloaded and parsed the xlsx (to `raw/global_biocrust_distribution_database/`). It has
**3848 data rows** and exactly **4 columns**:

| column | content |
|---|---|
| `ID` | 1–3848 |
| `biocrust_cover` | percent cover, range 0–140 (values >100 = summed multi-type cover) |
| `ai` | aridity index |
| `arid_gradient` | categorical: arid (1622), semiarid (1519), dry subhumid (344), hyperarid (178), humid (167), non arid (18) |

## Why rejected

1. **No coordinates.** There is no latitude/longitude (nor any x/y, geohash, or place
   field) in the released table. The data availability statement calls the point-level
   database "unpublished data (Ning Chen et al.)"; only this aggregated cover-vs-aridity
   table is distributed. Without lon/lat the records cannot be placed on the Sentinel-2
   grid — the standard "no recoverable geocoordinates" rejection (spec §8). A brief search
   for a georeferenced mirror (Zenodo/PANGAEA/Dryad/figshare) found none; the point
   coordinates are simply not public.
2. **No biocrust-type composition either.** The manifest's cyanobacteria/lichen/moss/mixed
   class breakdown is not present in the released table (only aggregate cover).
3. **Suitability caveat (secondary).** Even with coordinates, biocrust is only weakly
   observable at 10 m from S2/S1/Landsat — it would at best be a weak presence/cover label
   in dryland soils. This is moot given (1), but noted per task instructions.

The decisive blocker is (1): the georeferenced points needed for this pipeline are not in
the accessible release.

## To revive later

Obtain the georeferenced point table (per-record lon/lat + biocrust cover/type) directly
from the authors (Ning Chen et al.) or a future data release. With coordinates this would
become a sparse point dataset: a `biocrust presence` classification (or `biocrust_cover`
regression) written to `points.json` (spec §2a), 1-year windows in the 2016–2022 range.

## Outputs written

- `raw/global_biocrust_distribution_database/soil-10-763-2024-supplement.zip` (+ `SOURCE.txt`)
- `datasets/global_biocrust_distribution_database/registry_entry.json` (status: rejected)
- this summary
- No `datasets/.../locations/`, `points.json`, or `metadata.json` (rejected).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_biocrust_distribution_database
```
