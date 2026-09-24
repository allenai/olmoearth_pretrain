# Spanish National Forest Inventory (IFN) — dominant tree species

- **Slug:** `spanish_national_forest_inventory_ifn`
- **Task:** classification (per-pixel dominant tree species)
- **Samples:** 8,650 GeoTIFF tiles (64×64, uint8, local UTM @ 10 m)
- **Classes:** 70 dominant tree species (ids 0–69 by descending frequency; 255 = nodata)
- **Status:** completed

## Product chosen and why

The manifest entry ("Spanish National Forest Inventory (IFN)", family `tree_species`,
`label_type: points/polygons`) covers two very different IFN products. I evaluated both:

- **(a) IFN field plots (points)** — from the NFI Downloader
  (`https://descargaifn.gsic.uva.es/`). Each permanent plot records dominant species and
  stand measurements in the field, but the **public plot coordinates are deliberately
  degraded (rounded to ~1 km)** to protect the permanent-plot network. At ~1 km precision a
  plot is **not observable at 10 m** — this is the same fatal observability problem as FIA's
  ~1-mile fuzzing (spec §2). Rejected for species labelling.
- **(b) Mapa Forestal de España 1:25.000 (MFE25) polygons** — the MFE is the official
  forest cartography of Spain and is literally the **cartographic base of the 4th National
  Forest Inventory (IFN4)**. Each polygon (tesela) is photointerpreted at 1:25,000 with
  field checking and carries up to three tree species with occupancy percentages. These
  polygons **are observable at 10 m** and rasterize to a dominant-species class map. **This
  is the product used** (exactly the MFE-polygon path the task spec recommends).

## Access

The per-province MFE25 shapefile downloads on `mapama.gob.es`
(`descargafichero.aspx?f=mfe_*.zip`) are **gated behind a Google reCAPTCHA** and are not
scriptable. The **identical** MFE25/IFN4 data is served — with no credential and no captcha
— by MITECO's public **OGC API - Features** endpoint:

- Endpoint: `https://wmts.mapama.gob.es/sig-api/ogc/features/v1/collections/biodiversidad:MFE/items`
- Collection: `biodiversidad:MFE` — *"LC.Mapa Forestal de España 1:25.000 (MFE25), Base
  Cartográfica del Cuarto Inventario Forestal Nacional (IFN4)"*
- CQL2 filter: `especie1<>'sin datos' AND superficie_ha>40 AND o1>=70`
- Paging: `startIndex` + `sortby=-superficie_ha`, 1,000 features/page, geometry as WGS84
  (CRS84). ~1.5 GB of feature pages cached under `raw/{slug}/pages/`.
- License: Spanish government open data (MITECO).

No `.env` credential was needed.

## Method (GLiM-style homogeneous tiles)

1. **Candidate polygons (server-side):** large (`superficie_ha > 40` ha ≈ big enough to
   contain a 640 m tile), single-dominant-species (`o1 >= 70`, i.e. species-1 occupies
   ≥70 % of the canopy) forest teselas with a real `especie1`. The filter matched **81,716**
   polygons nationwide, spanning Spain's Atlantic, Mediterranean, Alpine/Pyrenean and
   Macaronesian (Canary Is.) biogeographic regions.
2. **Tile per polygon:** each candidate seeds one 64×64 (640 m) tile in local UTM at 10 m,
   centered on the polygon's interior *representative point* (guaranteed inside). The seed
   polygon is rasterized with its dominant-species class id; pixels **outside the seed
   polygon are 255 (nodata/ignore)**, not a fabricated background class (positive-only
   foreground mask, spec §5). Tiles are kept only if the seed species covers
   **≥ 0.5** of the tile → **68,559** homogeneous candidates.
3. **Classes:** distinct `especie1` among the homogeneous candidates → **70 species**
   (ids 0–69 by descending frequency; well under the 254 uint8 cap, so **0 species
   dropped**). Whitespace in source names stripped.
4. **Balancing:** `balance_by_class` by dominant species, `per_class=1000` capped by the
   25,000 total → effective 25000//70 = **357/class**. The 19 most common species reach 357
   each; rarer species contribute all they have (down to single-tile classes, which are
   kept per spec §5 — downstream assembly filters too-small classes). **Total 8,650 tiles.**

## Time range / change

Forest type / dominant species is a **static, persistent** label; the MFE25/IFN4 mapping
spans multiple years (~2007–2018). Per spec §5 (static labels) each sample uses a
representative Sentinel-era 1-year window **2018-01-01 → 2019-01-01** (within the manifest's
2016–2019 range). `change_time` is null.

## Class distribution (selected tiles, top classes)

`Quercus ilex, Pinus halepensis, Pinus sylvestris, Pinus pinaster, Quercus pyrenaica,
Pinus nigra, Eucalyptus globulus, Pinus pinea, Quercus suber, Fagus sylvatica, Quercus
faginea, Juniperus thurifera, Pinus radiata, Olea europaea, Castanea sativa, Pinus
uncinata, Pinus canariensis, Eucalyptus camaldulensis, Quercus robur` — 357 each; then a
long tail (Q. humilis 351 … down to single-sample species such as *Laurisilva*, *Robinia
pseudoacacia*, *Acer campestre*). Full per-class counts are in `metadata.json`
(`selected_tiles_per_class` / `written_tiles_per_class`).

## Verification (spec §9)

- 8,650 `.tif` each with a matching `.json`; all single-band `uint8`, 64×64, UTM
  (EPSG:32628–32631 for Spain) at 10 m; pixel values are valid class ids (0–69) with
  255 nodata; 8,352 tiles carry an ignore border where the seed tesela edge crosses the
  tile.
- Every `.json` has a 365-day `time_range` and `change_time: null`.
- 40/40 sampled tile centroids fall inside Spain; centroid span lon [-17.97, 4.05],
  lat [27.72, 43.67] (mainland + Canary Islands), consistent with national coverage.
- Georeferencing is exact (rslearn `GeotiffRasterFormat` encode; polygon reprojected
  CRS84→UTM via rslearn `STGeometry`). A full Sentinel-2 image overlay was not rendered,
  but coordinate/projection consistency and in-country centroids were confirmed.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spanish_national_forest_inventory_ifn
```

Idempotent: cached API pages under `raw/{slug}/pages/` and existing
`locations/{id}.tif` are skipped on re-run.

## Caveats

- Labels are photointerpreted map polygons (a derived reference product), not per-pixel
  field truth; the positive-only, homogeneous-tile sampling keeps only interiors of large
  single-species stands, so tiles are high-confidence but under-represent mixed/edge forest.
- Occupancy filter `o1>=70` and coverage `≥0.5` bias toward pure stands; naturally mixed
  forests (`especie2`/`especie3` present) are intentionally excluded from the label set.
- Some `especie1` values are genus/group labels (`Prunus spp.`, `Otras frondosas`,
  `Mezcla de coníferas`, `Laurisilva`) rather than single species; kept as-is.
