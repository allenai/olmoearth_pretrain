# Circumpolar Arctic Vegetation Map (CAVM)

- **Slug:** `circumpolar_arctic_vegetation_map_cavm`
- **Status:** completed
- **Task type:** classification (dense_raster → bounded-tile sampling)
- **Num samples:** 18,000 (18 classes × 1000)

## Source

Raynolds et al. (2019), *A raster version of the Circumpolar Arctic Vegetation Map (CAVM)*,
Remote Sensing of Environment. Distributed on Mendeley Data
(doi:[10.17632/c4xj5rv6kv.2](https://doi.org/10.17632/c4xj5rv6kv.2)), license **CC-BY-4.0**.

A single file, `Raster CAVM GIS data.zip` (2.4 MB compressed → 118 MB), contains
`raster_cavm_v1.tif`: a single-band **int8** grid at **1000 m** native resolution in a
north-polar **Sphere Lambert Azimuthal Equal Area** projection (lat_center=90, sphere
radius 6370997), 10798×10798, nodata=127. Each pixel encodes one Arctic tundra vegetation
unit from expert photointerpretation (grouping >400 plant communities into 16 vegetation
types), plus glacier / water / non-arctic codes. A legend CSV with short + long class
descriptions is included and was used to populate `metadata.json` descriptions.

### Access
Mendeley's download URL rejects the default urllib User-Agent (HTTP 403); `download_http`
was extended with an optional `headers` argument and the script sends
`User-Agent: Mozilla/5.0`. No credentials needed.

## Class scheme (18 classes, ids 0–17)

16 vegetation types kept in legend order as ids 0–15; then glacier (id 16); then water
(id 17). Source-code → class-id mapping:

| id | name | CAVM code(s) |
|----|------|--------------|
| 0 | Cryptogam, herb barren | B1 (1) |
| 1 | Cryptogam, barren complex | B2a (2) |
| 2 | Non-carbonate mountain complex | B3 (3) |
| 3 | Carbonate mountain complex | B4 (4) |
| 4 | Cryptogam, barren, dwarf-shrub complex | B2b (5) |
| 5 | Graminoid, forb, cryptogam tundra | G1 (21) |
| 6 | Graminoid, prostrate dwarf-shrub, forb, moss tundra | G2 (22) |
| 7 | Non-tussock sedge, dwarf-shrub, moss tundra | G3 (23) |
| 8 | Tussock-sedge, dwarf-shrub, moss tundra | G4 (24) |
| 9 | Prostrate dwarf-shrub, herb, lichen tundra | P1 (31) |
| 10 | Prostrate/hemi-prostrate dwarf-shrub, lichen tundra | P2 (32) |
| 11 | Erect dwarf-shrub, moss tundra | S1 (33) |
| 12 | Low-shrub, moss tundra | S2 (34) |
| 13 | Sedge/grass, moss wetland complex | W1 (41) |
| 14 | Sedge, moss, dwarf-shrub wetland complex | W2 (42) |
| 15 | Sedge, moss, low-shrub wetland complex | W3 (43) |
| 16 | glacier | GL (93) |
| 17 | water | FW (91) + SW (92) |

- **NA (99, non-arctic / outside tundra)** and raster **nodata (127)** → 255 (ignore); not
  a class and never sampled.
- **Judgment call — water:** fresh water (FW, lakes/rivers) and sea water (SW, ocean) were
  **merged** into a single `water` class (id 17) to match the manifest's stated class scheme
  ("16 Arctic vegetation types", "glacier", "water"). They could be kept separate; merged
  here for fidelity to the manifest.

Full source pixel counts (per legend, before sampling): every class has ≥18k source cells
(rarest: W1=18,095; SW=42.8M), so all 18 classes reached the full 1000-sample cap.

## Processing

Global derived-product map → **bounded-tile sampling** (spec §5), mirroring the GHS-SMOD
recipe:
1. Read the full 1 km raster; for each class, take all matching pixels and randomly sample
   up to **1000** (seed 42), circumpolar. Class-balanced.
2. Convert each sampled cell center (Sphere-LAEA metres) → lon/lat.
3. Around each cell cut a **64×64** tile in a **local UTM/UPS** projection at **10 m**
   (`get_utm_ups_projection` handles >84°N polar cells automatically), reprojected from the
   1 km source with **nearest** resampling (categorical labels).
4. Map source codes → class ids; NA/127 → 255. Write single-band **uint8** GeoTIFF
   (nodata=255) + sidecar JSON.

Because a 64×64 @10 m tile (640 m) is smaller than one native 1 km cell, each tile is
essentially the **homogeneous** CAVM class at that location. This heavy 1 km → 10 m
upsampling is intentional and documented (the CAVM class is defined on the 1 km grid),
identical in spirit to the GHS-SMOD 1 km product handling.

## Time range
The CAVM vegetation label is quasi-static (expert map; raster v1 built 2018; manifest range
2016–2019). Assigned a representative **1-year** Sentinel-era window anchored on **2018**
(`[2018-01-01, 2019-01-01)`). No change labels.

## Outputs
- `datasets/circumpolar_arctic_vegetation_map_cavm/metadata.json`
- `datasets/circumpolar_arctic_vegetation_map_cavm/locations/{000000..017999}.tif` (+ `.json`)
- Each tile: 64×64, uint8, local UTM/UPS at 10 m, nodata 255.

## Verification
- 18,000 tifs + 18,000 matching jsons. Sampled tifs confirmed single-band uint8, UTM CRS at
  10 m, 64×64, nodata 255, valid class ids. All 18 class ids (0–17) present across the set;
  46 distinct UTM zones observed.
- Every checked json has a ≤1-year `time_range` and `change_time=null`.
- **Round-trip spatial check:** for 60 random tiles, the source-raster vegetation code at the
  tile's center lon/lat was mapped through the class table and compared to the tile's dominant
  label — **0 mismatches, 0 all-nodata tiles**. Confirms geolocation + class mapping are
  correct. (Max sampled latitude 81.4°N in the spot check, within the UTM range; the code
  falls back to UPS for any >84°N cells.)

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.circumpolar_arctic_vegetation_map_cavm --workers 64
```
Idempotent: existing `{id}.tif` are skipped; the raw download/extract is skipped if
`raw/.../raster_cavm_v1.tif` is already present.

## Caveats
- Homogeneous-tile / heavy upsampling as noted above — the label is a 1 km class projected
  onto a 640 m S2-scale tile.
- Fresh + sea water merged into one `water` class (judgment call, see above).
- Non-arctic (NA) areas are ignore, not a class; the dataset is positive-only for tundra
  vegetation types — downstream assembly supplies negatives from other datasets (spec §5).
```
