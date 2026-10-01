# WorldCereal Reference Data Module (RDM)

- **Slug**: `worldcereal_reference_data_module_rdm`
- **Status**: completed — classification, 14,830 samples (6,081 points + 8,749 polygon tiles)
- **Source**: ESA WorldCereal Reference Data Module, public REST API
  (`https://ewoc-rdm-api.iiasa.ac.at`; portal `https://rdm.esa-worldcereal.org/`)
- **License**: mixed (public reference collections; many CC-BY-4.0)

## Source & access

The RDM is a global online repository of harmonized, curated **in-situ crop-type and
land-cover reference** datasets with a unified WorldCereal legend. It exposes an
OGC-style REST API that requires **no authentication for the public collections**
(login with Copernicus Data Space credentials only unlocks private/uploaded data, which
we do not use):

- Collections list: `GET /collections?MaxResultCount=100&SkipCount={n}` — 260 public
  collections (130 Point, 130 Polygon), ~84M features total.
- Features: `GET /collections/{collectionId}/items?MaxResultCount={n}&SkipCount={n}`
  returns GeoJSON. Each feature carries `ewoc_code` (10-digit harmonized legend code),
  `irrigation_status`, a single `valid_time` date, and land-cover / crop-type quality
  scores.

Note the endpoints paginate with ABP-style `MaxResultCount`/`SkipCount`, **not**
`PageNumber`/`PageSize` (those are silently ignored and re-return the first page). Large
`MaxResultCount` under high concurrency triggers server 500s, so we page in chunks of 500
with retries/back-off and limit download concurrency to 16 workers.

## Class scheme (harmonized)

Each `ewoc_code` is mapped to a unified class using the official WorldCereal class-mapping
tables (`class_mappings.json` from the `worldcereal-classification` repo):

1. **CROPTYPE24** gives a concrete crop type when the code is a recognised crop
   (maize, wheat, rice, barley, soy, sunflower, rapeseed, fibre_crops, sugar_cane,
   potatoes, sorghum, millet, oats, rye, triticale, beet, cassava, groundnuts,
   dry_pulses_legumes, grass_fodder_crops, other_oilseed, tobacco, vegetables).
2. Otherwise **LANDCOVER10** gives the broad land-cover class (temporary_crops,
   temporary_grasses, permanent_crops, grasslands, trees, shrubland, built_up, water,
   wetlands, bare_sparsely_vegetated).
3. Codes mapping to `ignore` / `no_crop` in both tables (e.g. `1000000000`
   "cropland_unspecified") are **dropped**.

673 legend codes map to a usable class; **32 classes** are actually present in the sampled
data. Well under the 254-class uint8 cap. The `irrigation_status` (irrigated vs rainfed)
attribute is preserved in the point rows / sample provenance but is **not** used as the
primary class (it would multiply the class count).

## Sampling

84M features far exceed the 25k cap, so we draw a **bounded sample of up to 2,000 features
per collection** (for geographic + class diversity), map each to a class, then run
`balance_by_class(per_class=1000, total_cap=25000)`. With 32 classes the effective
per-class limit is `25000 // 32 = 781`; 11 common classes hit the 781 cap, the rest are
data-limited (down to 32 for tobacco, 34 wetlands, 46 rye). Total 14,830 samples.

## Geometry handling

- **Point collections** → sparse point segmentation → one dataset-wide `points.json`
  (spec §2a). One row per point: `{lon, lat, label=class_id, time_range, source_id}`.
  6,081 points.
- **Polygon collections** (field parcels, LPIS, survey polygons) → rasterized into
  `locations/{id}.tif` (spec §2/§4): a ≤64×64 UTM 10 m tile sized to the parcel footprint
  (+2 px pad, capped at 64), centered on the polygon's representative point. Polygon
  interior = `class_id`, outside-polygon = **255 (nodata)** since only the labeled
  parcel's class is known (`all_touched=True` so small parcels still register). 8,749
  tiles. Sizes range 4×4 to 64×64.

Both share ONE unified class map in `metadata.json`.

## Time range

Crop / land-cover labels are seasonal/annual, so each sample gets a **1-year window
anchored on the year of its `valid_time`** (clamped to the Sentinel era, ≥2016). Source
years span 2017–2024.

## Outputs

- `datasets/worldcereal_reference_data_module_rdm/metadata.json` — 32 classes (with
  descriptions), class_counts, n_points/n_polygons.
- `.../points.json` — 6,081 point rows.
- `.../locations/{000000..008748}.tif` + `.json` — 8,749 rasterized polygon tiles.
- `raw/worldcereal_reference_data_module_rdm/` — `collections_index.json`,
  `class_mappings.json`, `items/{collectionId}.geojson` (raw fetched features),
  `SOURCE.txt`.

## Verification

- Sampled tifs: single-band uint8, UTM CRS at 10 m, north-up, sizes 4–64 (≤64), values are
  class ids 0–31 plus 255 nodata, no out-of-range values; every `.tif` has a matching
  `.json` with a 1-year `time_range` and pixel_bounds matching the raster.
- Spatial sanity: e.g. point `p000000` (USDA-CDL) at lon −96.30, lat 44.75 labeled maize
  falls in the US Corn Belt (South Dakota) — sensible.
- Idempotent: re-running skips already-downloaded collections and already-written tiles.

## Caveats

- Per-collection sampling takes the first ≤2,000 features of each collection, which can be
  spatially clustered within very large collections (e.g. national LPIS); acceptable for a
  diverse label bank but not a globally uniform sample.
- ~90% of fetched features map to `ignore`/unspecified cropland and are dropped; the kept
  32 classes reflect where the harmonized legend resolves a specific crop or land cover.
- Rare classes are data-limited (< 781); truncation logged above.
- There is a paired derived product already on disk (`OlmoEarth WorldCereal cropland`);
  this RDM set is the richer upstream in-situ reference and is complementary.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.worldcereal_reference_data_module_rdm
```
