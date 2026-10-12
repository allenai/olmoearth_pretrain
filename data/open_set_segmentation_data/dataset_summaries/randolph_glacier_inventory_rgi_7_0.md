# Randolph Glacier Inventory (RGI 7.0)

- **Slug**: `randolph_glacier_inventory_rgi_7_0`
- **Status**: completed
- **Task type**: classification (binary per-pixel segmentation: glacier vs background)
- **Num samples**: 1000 tiles (64×64, UTM, 10 m)
- **Source**: NSIDC nsidc-0770 v7 — RGI 7.0 Consortium 2023, coordinated with GLIMS
  (doi:10.5067/f6jmovy5navz), CC-BY-4.0.

## Source & access

RGI 7.0 offers four products (G glacier, C glacier-complex, I intersects, L centerlines).
We use the **glacier product (G)**: ~274,531 individual, manually delineated glacier
outline polygons, distributed as one zipped ESRI shapefile per first-order region (19
regions), each in EPSG:4326.

Downloaded over HTTP from the NSIDC DAAC:
`https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G/RGI2000-v7.0-G-<region>.zip`.
The host requires NASA Earthdata / URS OAuth login; credentials come from
`.env` (`NASA_EARTHDATA_USERNAME`/`PASSWORD`) written to `~/.netrc`
and used via `download.download_earthdata` (requests + netrc, follows the OAuth redirect).
All 19 regional zips (~486 MB total) are downloaded and extracted to
`raw/randolph_glacier_inventory_rgi_7_0/<region>/`.

## Class scheme

Binary segmentation:

| id | name | definition |
|----|------|------------|
| 0 | background | non-glacier terrain (bedrock, snow-free ground, seasonal snow, water, vegetation) outside every RGI outline |
| 1 | glacier | land ice inside an RGI 7.0 glacier outline (RGI2000 nominal-2000 extent) |

The glacier outline is a true boundary against *observable* non-glacier terrain, so this
is a genuine two-class segmentation rather than a positive-only presence mask. Each tile
is rasterized with **all** glacier polygons intersecting its footprint (via a spatial
index), so adjacent glaciers in a dense tile are correctly labeled — not just the glacier
the tile is centered on. `nodata=255` is declared in metadata but no pixel uses it (every
pixel is 0 or 1).

**Why not terminus type**: the manifest notes "glacier (with terminus-type attributes)",
but in RGI 7.0 the `term_type` attribute is "not assigned" (code 9) for 99.4% of glaciers
(only 1,561 of 274,531 carry a real terminus code, almost all marine-terminating=1), so it
cannot support a class scheme. `term_type` is recorded per sample in `source_id` for
provenance instead.

## Sampling (bounded regional, spec §5)

RGI is global (~274k glaciers), so a bounded set is sampled. Glaciers **≥ 0.1 km²** (drops
sub-resolution slivers and improves temporal stability) are sampled **round-robin across
all 19 regions** for geographic diversity, up to the **1000-per-class** cap. Result: 1000
glacier-centered tiles, ~52–53 per region across every region (Alaska → Subantarctic/
Antarctic islands). Each glacier's centroid (`cenlon`/`cenlat`) is the tile center; the
64×64 UTM 10 m window spans 640 m.

- Glacier (class 1) present in 1000/1000 tiles; background (class 0) present in 825/1000.
- Glacier pixel-fraction per tile: min 0.16, median 0.73, mean 0.71; 175 tiles are fully
  glacier (large ice fields), the rest show a boundary against background.
- Scan pool after the 0.1 km² floor: 170,512 glaciers.

## Time range (spec §5, static/persistent label)

RGI 7.0 is the **nominal-2000 inventory**: outline source dates are ~99.9% pre-2016 (mean
year 2001, only 238 glaciers dated ≥2016). Glacier extent is a slowly changing, persistent
feature, so — per the task ("static extent → representative Sentinel-era 1-year window") —
every sample is assigned a uniform **1-year window in 2020** (`[2020-01-01, 2021-01-01)`).
The original outline source date (RGI2000 acquisition year) is recorded per sample in
`source_id` (e.g. `...@src_date=2003-08-13T00:00:00;term_type=9`).

**Caveat**: glaciers — especially small ones — have retreated somewhat since ~2000, so a
2020 Sentinel-2 image may show a modestly smaller glacier than the RGI2000 outline. The
≥0.1 km² area floor limits (but does not eliminate) this mismatch. This is inherent to
pairing a year-2000 inventory with Sentinel-era imagery and is accepted per the task's
static-label handling.

## Verification (§9)

- All 1000 `.tif` are single-band uint8, exactly 64×64, projected UTM at 10 m/pixel; pixel
  values are only {0,1}; every `.tif` has a matching `.json` with a 1-year `time_range`.
- `metadata.json` class ids {0,1} cover all values in the tifs.
- Georeferencing sanity: all 200 sampled tiles have glacier at the center block (tiles are
  centered on RGI centroids), confirming correct 4326→UTM reprojection and pixel bounds.
  (A full Sentinel-2 overlay was not run; georeferencing derives directly from RGI's own
  exact coordinates via the validated reprojection path.)
- Idempotent: re-running skips existing `{sample_id}.tif`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.randolph_glacier_inventory_rgi_7_0
```
Requires `~/.netrc` with `machine urs.earthdata.nasa.gov` credentials (from
`.env`).
