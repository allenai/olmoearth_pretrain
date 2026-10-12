# GRanD (Global Reservoir and Dam Database)

- **Slug:** `grand_global_reservoir_and_dam_database`
- **Status:** completed — **classification**, **7,424 samples**
- **Family:** dams | **Region:** Global | **label_type (manifest):** points + polygons

## Source & access

The manifest dataset is **GRanD v1.3** (Lehner et al. 2011; v1.3 = 7,320 large dams plus
reservoir polygons, expert-curated). The *standalone* GRanD product has been **discontinued
and fully integrated into the Global Dam Watch (GDW) consensus database v1.0** (Nature
Scientific Data 2024, doi:10.1038/s41597-024-03752-9), which is now the canonical,
freely-downloadable form. The official standalone GRanD download is gated (SEDAC/Earthdata
login; Global Dam Watch download is behind a Google Form), but GDW v1.0 is on figshare with
no credentials required:

- **Download:** figshare record **25988293**, file `GDW_v1_0_shp.zip` (70 MB),
  `https://ndownloader.figshare.com/files/47913754` → `https://doi.org/10.6084/m9.figshare.25988293`
- **License:** CC-BY-4.0
- Two ESRI shapefile layers, EPSG:4326: `GDW_barriers_v1_0.shp` (41,145 points),
  `GDW_reservoirs_v1_0.shp` (35,295 polygons). Both carry `GDW_ID`, `ORIG_SRC`, `GRAND_ID`,
  `YEAR_DAM`, dam coords, etc.

Raw stored at `raw/grand_global_reservoir_and_dam_database/` (`SOURCE.txt` + extracted shp).

## Judgment call: scoped to GRanD provenance (not full GDW)

GDW integrates several source databases (`ORIG_SRC`): GOODD (23,633), **GRanD (7,424)**,
GROD (6,060), GOODD-NID (2,298), JRC-GSW, FHReD, … The manifest catalogs **GOODD as a
SEPARATE dataset**, so to avoid duplicating GOODD's ~24k points into this one, I **scoped
this dataset to GRanD-sourced records only**: barriers with `ORIG_SRC=='GRanD'` / `GRAND_ID>0`
(**7,424 dam points**) and reservoirs with `GRAND_ID>0` (**7,378 polygons**). This matches
GRanD v1.3's documented ~7,320 records (minor +100 from GDW harmonization). All 7,378
reservoirs have a matching GRanD barrier; 46 barriers are dam-only (run-of-river, etc.).

## Unified class scheme (mixed-modality, spec §5)

Reservoirs are polygons (segmentation) and dams are positive-only points (detection),
combined into ONE scheme:

| id | name | meaning |
|----|------|---------|
| 0 | background | land / other surface inside the tile |
| 1 | reservoir | inside a GRanD reservoir polygon (impounded water body, visible at 10–30 m) |
| 2 | dam | GRanD barrier point — detection positive (1 px) |
| 255 | nodata/ignore | 10 px buffer ring around each dam positive |

## Encoding

- One **64×64 (640 m) local-UTM tile at 10 m/pixel** per GRanD dam point, **centered on the
  dam** (the dam sits at the reservoir margin/outlet, so a dam-centered window captures
  reservoir water + surrounding land + the dam structure — a natural mix of all three classes).
- All GRanD reservoir polygons intersecting the tile are rasterized to class **1**
  (`all_touched=True`; ~5% of polygons have invalid rings → repaired with `shapely.make_valid`).
- Every GRanD dam point in the tile is stamped as a class-**2** positive (1 px) ringed by a
  **10 px nodata (255) buffer** (dam coords aren't pixel-exact; buffer avoids penalizing a
  few-pixel offset). Applied on top of the reservoir raster.
- In-tile **land (class 0) supplies spatially-meaningful dam negatives**, so no separate
  negative tiles are fabricated (spec §5 detection exception; reservoirs are positive-only
  otherwise).
- All ~7,424 GRanD records are used (well under the 25k per-dataset cap) — no subsampling.

## Time range

Dams/reservoirs are **persistent structures** visible throughout the Sentinel era; most
GRanD dams predate 2016 (`YEAR_DAM` mostly unknown/−99 or pre-2016). Per spec §5 (static
labels) each tile gets a **1-year window with start year sampled uniformly in 2016–2020**
(counts: 2016=1536, 2017=1546, 2018=1496, 2019=1451, 2020=1395). The handful of dams built
≥2016 are treated as persistent rather than change labels (yearly precision makes a change
label ill-posed) — no `change_time` set. Noted as a minor caveat.

## Sample counts / class balance

- **Total tiles:** 7,424 (7,424 tif + 7,424 json).
- **Tiles containing each class:** background 7,419; reservoir 7,383; dam 7,424 (every tile
  has its dam).
- Reservoirs occupy ~46% of valid pixels per tile on average (median reservoir ≈ 0.4 km² ≈
  one tile footprint).

## Verification

- Opened multiple tifs: single-band **uint8**, **UTM @ 10 m**, size **64×64**, nodata **255**,
  values ⊆ {0,1,2,255}; every tif has a matching json with a ≤1-year `time_range`; metadata
  class ids cover all raster values.
- **Geometric sanity check:** 149/149 sampled dam points (class 2) sit adjacent (≤12 px) to
  reservoir polygon pixels — confirms consistent georeferencing between the point and polygon
  layers and that dams land on reservoir margins.
- A full Sentinel-2 pixel overlay was not run; labels are authoritative GIS vectors written
  via rslearn's exact encoder and internal consistency is strong. Any future S2 eyeball
  should confirm class-1 sits on open water.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.grand_global_reservoir_and_dam_database --workers 64
```
Idempotent (skips existing `locations/{id}.tif`). Runtime ≈ 2 min on 64 workers.

## Caveats

- Uses GDW v1.0 (the current form of discontinued GRanD) scoped to GRanD provenance; a few
  records differ from the original v1.3 due to GDW harmonization.
- Dam construction year is unknown/pre-2016 for most records; time windows are representative
  Sentinel-era windows, not exact acquisition times.
- The manifest note "GOODD adds ~38k dam points" is intentionally NOT merged here — GOODD is
  a separate manifest/registry dataset.
