# AI4Boundaries

- **Registry slug:** `ai4boundaries`
- **Status:** completed
- **Task type:** classification (dense 3-class field-boundary segmentation)
- **Samples:** 1005 label patches (`locations/{id}.tif` + `.json`)
- **Source:** European Commission, Joint Research Centre — *AI4Boundaries* (d'Andrimont et
  al., *Earth Syst. Sci. Data* 15, 317–329, 2023). JRC Open Data Catalogue
  ([landing page](https://data.jrc.ec.europa.eu/dataset/0e79ce5d-e4c8-4721-8773-59a4acf2c9c9)).
- **License:** CC BY 4.0 / EC reuse notice (public, no credential required).

## What the source is

AI4Boundaries is an AI-ready benchmark for agricultural **field-boundary delineation**,
derived from openly-released **GSAA** (Geospatial Aid Application) parcel declarations for
**2019** across 7 EU regions: Austria (AT), Catalonia/Spain (ES), France (FR), Luxembourg
(LU), Netherlands (NL), Slovenia (SI), Sweden (SE). It ships two paired image/label sets:
a **10 m Sentinel-2** monthly-composite set (256×256 patches) and a **1 m aerial
orthophoto** set (512×512). We use **only the 10 m Sentinel-2 label masks** — pretraining
supplies its own imagery, and field extent/boundaries are observable at 10 m S2 (this is
exactly what the S2 half of the benchmark was built to detect). The 1 m aerial masks and
all imagery (`.nc` S2 series, orthophotos) are **not** downloaded.

Each S2 label mask (`sentinel2/masks/{NUTS0}/{id}_S2label_10m_256.tif`) is a 256×256, 10 m,
**EPSG:3035 (ETRS89-LAEA)** 4-band `float32` GeoTIFF:
1. field-**extent** mask (1 = field, 0 = non-field)
2. field-**boundary** mask (1 = boundary pixel)
3. distance-to-boundary (unused)
4. field enumeration / instance id (unused)

## Access / download

Downloaded only the label bundle from the JRC FTP
(`https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/`):
- `sentinel2/masks.zip` — 832 MB, 7598 S2 label masks (the split-assigned subset; the 233
  Sweden/France NA-split samples are excluded by the bundle).
- `sentinel2/ai4boundaries_ftp_urls_sentinel2_split.csv` — file→split table (train/val/test).

Extracted to `raw/ai4boundaries/masks/{train,val,test}/`. No imagery pulled. See
`raw/ai4boundaries/SOURCE.txt`.

## Label / class mapping

A 3-class dense segmentation built from bands 1–2 (boundary takes priority over interior):

| id | name           | definition |
|----|----------------|------------|
| 0  | background     | non-field / not a declared 2019 GSAA parcel |
| 1  | field interior | inside a parcel (extent==1 and boundary==0) |
| 2  | field boundary | GSAA-derived boundary pixel (boundary==1) — the core signal |

No nodata used inside the tiles (all pixels observed); `nodata_value` in `metadata.json` is
255 (unused). **Caveat (from the source paper):** GSAA parcels can be missing, so
background (0) mixes true non-field land with un-declared/missing fields. AI4Boundaries is a
masked-learning benchmark — models are meant to learn the extent/boundary of *included*
fields, not to treat background as a clean negative.

## Processing (spec §4 dense_raster, VHR-style reprojection)

1. Derive the 3-class array in EPSG:3035 from bands 1–2.
2. Reproject to a **local UTM zone at 10 m** with **nearest** resampling (preserves the
   1-px boundary lines; never bilinear). Reprojected grid is ~256–274 px depending on
   latitude/UTM offset.
3. Tile into non-overlapping **64×64** windows (spec cap). Keep only windows with ≥1 field
   pixel (class 1 or 2); reprojection slivers <32 px on either axis are dropped.
4. **Tiles-per-class balanced** selection (`sampling.select_tiles_per_class`): ≤1000 tiles
   per class, rarest-class-first, ≤25 000 total. All three source splits used.

Because the 3 classes co-occur in nearly every field tile, class-balancing to 1000/class
yields ~1005 tiles (of 97 468 field-containing candidates) — this is the spec-driven cap,
not a data limitation.

- **Time range:** 1-year 2019 window `[2019-01-01, 2020-01-01)` (S2 composites are Mar–Aug
  2019; post-2016 Sentinel era). Not a change dataset (`change_time=null`).
- **GeoTIFF:** single-band uint8, local UTM, 10 m, ≤64×64.

## Sample counts

- Total patches: **1005**.
- Per-class tile counts (a tile counts toward every class it contains): background 1000,
  field interior 1005, field boundary 1001.
- By source split: train 715, val 146, test 144.
- Countries represented across candidates: AT, ES, FR, LU, NL, SE, SI.

## Verification

- 1005 `.tif` + 1005 `.json`; every tif single-band uint8, UTM (EPSG:326xx), 10 m, 64×64;
  pixel values ∈ {0,1,2} matching `metadata.json`; all time ranges = 2019 (≤1 yr).
- Spatial sanity: one sample per country reprojected back to WGS84 — every tile centroid
  falls inside the expected country (AT/ES/FR/LU/NL/SE/SI), confirming the LAEA→UTM
  reprojection and georeferencing are correct.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ai4boundaries
```

Idempotent: existing `locations/{id}.tif` are skipped; the mask scan is cached at
`raw/ai4boundaries/scan_cache.pkl` (delete to force a re-scan).

## Caveats

- Background (0) is not a pure negative (missing GSAA parcels) — see label mapping above.
- ~5% of source cells have no declared parcels (all-background); these contribute no tiles.
- Very small parcels (<0.5 ha) are near the limit of S2 10 m resolution, so some fine
  boundaries may be under-represented in the raster labels.
