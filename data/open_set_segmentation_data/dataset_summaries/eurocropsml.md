# EuroCropsML

- **Slug:** `eurocropsml`
- **Status:** completed
- **Task type:** classification (crop type) — **sparse points** (spec §2a/§4)
- **Samples:** 13,678 points across 176 HCAT crop classes
- **Source:** EuroCropsML, Zenodo record [15095445](https://zenodo.org/records/15095445)
  (v13, 2025-03-31; concept DOI 10.5281/zenodo.10629609),
  [github.com/dida-do/eurocropsml](https://github.com/dida-do/eurocropsml)
- **License:** CC-BY-4.0

## What the source is

EuroCropsML is the ML-ready benchmark derived from **EuroCrops v9**. It provides
**706,683 agricultural parcels** across **Estonia (175,906), Latvia (431,143), and
Portugal (99,634)** for the reference year **2021**, each labeled with a harmonized
**HCAT** (Hierarchical Crop and Agriculture Taxonomy) 10-digit `EC_hcat_c` code. Labels
come from farmers' CAP/LPIS self-declarations; the crop taxonomy is HCAT3. Each parcel is
pre-processed into a per-parcel `.npz` (Sentinel-2 median time series + metadata,
including the parcel **centroid `[lon, lat]`** in WGS84).

## Product choice: points, not rasterized polygons

The spec offers two EuroCrops(ML) products: (a) EuroCrops parcel polygons rasterized to a
dense crop-type class map, and (b) the EuroCropsML ML-ready points. **The sibling
`eurocrops` dataset already produced product (a)** (18,590 rasterized-parcel dense tiles
over 8 countries), and this registry entry's `label_type` is `points`. To avoid
duplicating `eurocrops` and to match the ML-ready package, EuroCropsML is emitted as
**product (b): one WGS84 point per parcel at its centroid**, per spec §2a (sparse points →
one dataset-wide `points.geojson`, no per-sample GeoTIFFs). EuroCropsML distributes only
per-parcel median time series + centroids (the polygons live in the much larger
`raw_data.zip`), so the centroid + HCAT code + year 2021 is the natural signal. Crop
parcels are clearly observable at 10 m Sentinel-2, so a 1×1 point label at the parcel
centroid pairs cleanly with imagery.

## Access / download

Downloaded only `preprocess.zip` (~1.47 GB) to `raw/eurocropsml/`. Each parcel's HCAT code
and NUTS3 region are encoded in the member filename
`preprocess/{NUTS3}_{parcelid}_{EC_hcat_c}.npz`, so the **full class distribution is read
from the zip's central directory with no extraction**. Only the sampled subset of `.npz`
files is read (from the local zip, 64-way parallel) to pull each parcel's centroid
(`center` array = `[lon, lat]` WGS84). No imagery is needed — pretraining supplies its own.

## Class mapping (HCAT collapse)

Classes are the distinct `EC_hcat_c` codes present in the parcels. EuroCropsML has exactly
**176 distinct HCAT codes**, all resolvable via the repo's HCAT3 mapping
(`data/eurocrops_hcat3_mapping.json`) — well under the 254-class uint8 cap, so **all 176
are kept (0 dropped)**. Class ids are assigned `0..175` in **descending global HCAT-code
frequency**; names/descriptions come from the HCAT3 mapping. No coarser collapse was
needed since 176 < 254; the HCAT3 leaf names are already the harmonized crop groups (e.g.
`pasture_meadow_grassland_grass`, `winter_common_soft_wheat`, `oats`, `olive_plantations`).

Top classes (by global frequency → low ids): `pasture_meadow_grassland_grass` (id 0),
`winter_common_soft_wheat` (1), `oats` (2), `not_known_and_other` (3),
`spring_common_soft_wheat` (4), `spring_barley` (5), … The dominant meadow/grassland class
(~45% of all parcels) is capped like every other class.

## Sampling

Class-balanced (`balance_by_class`) with the 25k per-dataset cap. With 176 classes the
effective per-class limit is `min(1000, 25000 // 176) = 142`. The class distribution is
very skewed, so the total lands at **13,678** points: 73 classes reach the 142 cap, the
rest contribute as many parcels as exist (down to 13 single-parcel classes). Rare classes
are all retained (downstream assembly filters classes below its own minimum). Selection is
seeded/deterministic.

## Time range & change handling

`time_range` = 1-year window on the reference year **2021** (`[2021-01-01, 2022-01-01)`)
for every point — EuroCropsML is entirely year 2021 (post-2016 ✓). `change_time = null`:
crop type is a static seasonal label, not a change event.

## Verification (spec §9)

- `points.geojson`: `FeatureCollection`, `task_type=classification`, `count=13678`, one
  `Point` per parcel; per-feature `properties` = `id`, `label`, `time_range`,
  `change_time=null`, `source_id` (`{NUTS3}/{parcel_id}`).
- All labels are valid class ids `0..175`; `metadata.json` `classes` cover all values.
- Centroids all land inside the expected countries: lon −9.41…28.16, lat 37.37…59.58
  (Portugal ~−9°/37–42°N; Estonia/Latvia ~21–28°E/56–60°N); 0 points outside the EU bbox.
- Spatial sanity: centroids are authoritative (precomputed by EuroCropsML from the
  EuroCrops parcel polygons), so georeferencing is trusted; a point-only dataset has no
  raster to overlay.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eurocropsml
```

Idempotent: skips re-downloading `preprocess.zip` if present; selection is seeded so the
same `points.geojson` is reproduced.

## Caveats

- `not_known_and_other` (HCAT 3399000000) is a real EuroCrops catch-all class, kept as a
  normal class id; downstream filtering may drop it if undesired.
- Points are parcel **centroids**; a large parcel's centroid may sit slightly off-field for
  irregular shapes, but EuroCrops parcels are homogeneous single-crop declarations, so the
  10 m pixel at the centroid is reliably the declared crop.
