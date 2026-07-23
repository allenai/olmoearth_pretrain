# RPG France (Registre Parcellaire Graphique)

- **Slug**: `rpg_france_registre_parcellaire_graphique`
- **Task type**: classification (per-pixel crop type)
- **Status**: completed
- **Samples**: 19,798 label patches across 238 classes
- **Source**: IGN France / ASP, distributed via the Géoplateforme (`data.geopf.fr/telechargement`)
- **License**: Licence Ouverte / Etalab (open, attribution)

## What the source is

The RPG is the anonymized French national LPIS: every declared agricultural parcel from
the CAP (Common Agricultural Policy) farmer declarations, updated annually. From the 2015
edition on, the "RPG 2.0" product carries the crop type down to the individual **parcelle**
polygon. Each parcel has:
- `CODE_CULTU` — 3-letter detailed crop code (~300 codes nationally; 238 present in our
  sampled regions), e.g. `BTH` = Blé tendre d'hiver, `MIS` = Maïs, `VRC` = Vigne raisins de
  cuve. **Used as the class label.**
- `CODE_GROUP` — numeric crop-group code (RPG 28-group scheme), attached as the class
  description.

RPG is the largest single-country LPIS and is the annual, national analogue of the
EuroCrops snapshots — this dataset is processed exactly like `eurocrops.py`.

## Access method

Data is distributed per administrative region as `.7z` archives on the Géoplateforme:
`https://data.geopf.fr/telechargement/download/RPG/RPG_2-0__SHP_LAMB93_{REGION}_{YEAR}-01-01/…7z.001`.
The download host rejects urllib's default User-Agent (HTTP 403); a browser UA header is
sent. Archives are extracted with `py7zr` to `PARCELLES_GRAPHIQUES.shp` (Lambert-93,
EPSG:2154). The `CODE_CULTU`→French-libellé nomenclature comes from IGN/ASP as mirrored by
`etalab/api-rpg` (`codes/CULTURE.csv`).

## Bounded sampling (this is a large national dataset)

RPG is ~9.5M parcels/year nationally. We download a **bounded, geographically diverse
subset of 8 metropolitan administrative regions** for a single recent snapshot year (2022,
within the manifest's 2016–2024 range), covering all French agroclimatic zones and every
major crop:

| Region | Name | Crop emphasis | Parcels |
|--------|------|---------------|---------|
| R24 | Centre-Val de Loire | Beauce cereals, rapeseed | 579,086 |
| R32 | Hauts-de-France | sugar beet, potato, wheat, flax | 572,158 |
| R44 | Grand Est | Champagne/Alsace vineyards, sugar beet | 861,969 |
| R53 | Bretagne | maize, grassland, vegetables | 853,439 |
| R75 | Nouvelle-Aquitaine | maize, sunflower, vineyard | 1,606,111 |
| R76 | Occitanie | durum wheat, vineyard, orchards | 1,638,413 |
| R84 | Auvergne-Rhône-Alpes | grassland, orchards, maize | 1,366,524 |
| R93 | PACA | vineyard, orchards, rice (Camargue) | 318,721 |

~7.8M candidate parcels; verified samples span lon −4.4…8.0, lat 43.0…50.8.

## Label construction

Each selected parcel polygon is reprojected to its local UTM zone (EPSG:326xx, France spans
UTM 30N/31N/32N) and rasterized at 10 m into a `≤64×64` single-band **uint8** tile via
`rasterio.features.rasterize` (shared `rasterize.py`): the parcel's class id is burned
inside the polygon (`all_touched=True`), everything outside is **255 = nodata/ignore**.
There is no true background class — unlabeled land outside declared parcels is "ignore",
not "not-crop" (spec §5; assembly step supplies negatives from other datasets). Tiles are
centered on the parcel centroid; parcels larger than 640 m are cropped to a centered 64×64
window.

## Classes

- Class label = `CODE_CULTU`. 238 distinct codes appear in the sampled regions — under the
  254-class uint8 cap, so **no codes were dropped** (`dropped_code_cultu: []`).
- Class ids assigned 0..237 in **descending global frequency**. Names = French libellé from
  the RPG culture nomenclature; description = `CODE_CULTU` + RPG crop group (id + name).
- Nodata / ignore value = 255.

## Sampling / balancing

Tiles-per-class balanced with the 25k per-dataset cap (`balance_by_class`, `per_class=1000`,
`total_cap=25000`). With 238 classes the effective per-class limit is `25000 // 238 = 105`.
Common crops (wheat, maize, grassland, vineyard, sunflower, colza, …) all reach 105; rare
codes are kept in full (2 classes have exactly 1 sample). Per spec §5, sparse classes are
**not** dropped here — downstream assembly filters classes below its minimum. Final total
19,798 (25 parcels produced empty rasters — sub-10 m slivers — and were skipped).

## Time range

Seasonal/annual crop labels → 1-year window anchored on the 2022 snapshot
(`[2022-01-01, 2023-01-01)`). No change labels.

## Verification (spec §9)

- Sampled tifs: single band, uint8, local UTM (EPSG:32631/32632), 10 m resolution, size
  ≤64, pixel values are valid class ids + 255 nodata.
- 19,798 `.tif` each with a matching `.json` (1:1); every `time_range` is a 1-year window;
  `metadata.json` class ids (0–237) cover all values in the tifs.
- Spatial sanity: 200 random tile centers all fall inside metropolitan France, spanning all
  8 regions. Full Sentinel-2 overlay was not rendered, but georeferencing is inherited from
  the validated EuroCrops rasterization path (exact IGN vector geometries reprojected via
  rslearn `Projection`), and the correct France UTM zones + in-country coordinates confirm
  placement.
- Idempotent: `_write_tile` skips existing `{sample_id}.tif`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.rpg_france_registre_parcellaire_graphique
```

## Judgment calls / caveats

- Used detailed `CODE_CULTU` (~300-code nomenclature) rather than the coarse `CODE_GROUP`
  (~28 groups), for richest crop-type semantics — mirrors EuroCrops using HCAT leaf codes.
- One snapshot year (2022) across 8 diverse regions, not all years 2016–2024 nor all 13
  metropolitan regions, to honor the bounded-sampling / 25k-cap guidance while covering
  every crop class. Overseas départements (Guadeloupe, Martinique, Guyane, Réunion, Mayotte)
  were excluded from this subset.
- `CODE_CULTU` → name nomenclature is sourced from the community `etalab/api-rpg` mirror of
  the IGN/ASP culture list; codes absent from it (none occurred) would fall back to the raw
  code string.
