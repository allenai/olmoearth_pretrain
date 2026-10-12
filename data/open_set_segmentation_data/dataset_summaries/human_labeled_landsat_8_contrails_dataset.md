# Human-Labeled Landsat-8 Contrails Dataset — COMPLETED

- **Slug**: `human_labeled_landsat_8_contrails_dataset`
- **Name**: Human-Labeled Landsat-8 Contrails Dataset
- **Source**: Google Research — McCloskey et al., "A human-labeled Landsat-8 contrails
  dataset", ICML Climate Change AI workshop 2021
  (https://research.google/pubs/a-human-labeled-landsat-contrails-dataset/).
- **Data**: `gs://landsat_contrails_dataset/2023_01_20_1674247800/` (public, no credentials).
- **License**: CC BY 4.0 (bucket `data/LICENSE`). Underlying Landsat-8 courtesy USGS
  (unrestricted); de-identified flight context licensed from FlightAware (unused here).
- **Family / region**: atmosphere / global (daytime), 2017–2020.
- **Label type**: dense_raster (contrail annotation polygons) → binary segmentation.
- **Status**: **completed** — `task_type=classification`, `num_samples=1000`.

## What the source is

Several thousand Landsat-8 scenes with pixel-level **manual contrail annotations**. The
release is 100 JSON-lines shards (~43 GiB); each line is one Landsat-8 scene:

```
{"filename": "LC08_L1TP_036037_20180406_..._B10.TIF",
 "polygons": [[[x, y], ...], ...],          # human contrail polygons
 "advected_flight_waypoints": {...},          # flight context (ignored)
 "advected_flight_density": [[...]]}          # flight context (ignored)
```

Scanning all shards: **11,107 scenes total, 4,417 contrail-positive (39.8%), 94,332
contrail polygons**. Dates: 2017×1, 2018×8,130, 2019×794, 2020×2,182 — **entirely
post-2016**, so no pre-2016 filtering was needed.

The `polygons` are vertex lists in the pixel grid of the **10×-downsampled** Landsat-8
thermal band the labelers viewed (the released notebook builds the false-color image via
`gdal ReadAsArray(buf = shape/10)` of the 30 m band, so **1 downsampled pixel ≈ 300 m**).
The false color the labelers saw is (11 µm − 12 µm brightness-temperature difference,
1.37 µm cirrus reflectance, 12 µm brightness temperature).

## Access / download

Public GCS bucket, listed and pulled without credentials:
`gsutil -m cp "gs://landsat_contrails_dataset/2023_01_20_1674247800/data/landsat_contrails.json-*-of-00100"`
→ `raw/{slug}/shards/` (43.3 GiB, 100 shards). Only `filename` + `polygons` are used (parsed
from each line's prefix, before the large flight arrays). The dataset itself carries **no
lon/lat**; georeferencing is recovered per scene from the Landsat-8 L1 **MTL** metadata on
the public bucket `gs://gcp-public-data-landsat` (Collection-1 still hosted; every MTL
fetched HTTP 200). MTLs are cached under `raw/{slug}/mtl/`. Also fetched: `demo_shard.json`,
`LICENSE`, the acknowledgements, and the code zip (`raw/{slug}/`).

## Georeferencing (the key step)

Per scene, the MTL gives `UTM_ZONE`, hemisphere (from `CORNER_UL_LAT_PRODUCT` sign),
`CORNER_UL_PROJECTION_X/Y_PRODUCT`, `THERMAL_SAMPLES/LINES`, `GRID_CELL_SIZE_THERMAL` (30 m)
and the acquisition timestamp. Each polygon vertex `(x_ds, y_ds)` maps:

```
E = UL_E + x_ds * (SAMPLES/ds_w) * 30      # scene UTM easting
N = UL_N - y_ds * (LINES /ds_h) * 30      # scene UTM northing   (ds_w=int(SAMPLES/10),…)
```

then scene-UTM → WGS84 lon/lat, and finally into a **local UTM 10 m** tile via the shared
`geom_to_pixels` / `io.utm_projection_for_lonlat` path (same as the polygon datasets, cf.
`cal_fire_frap_fire_perimeters.py`). All 4,417 positive scenes georeferenced (0 dropped:
all MTLs present and UTM). A couple of near-polar scenes yield UPS-South (EPSG:5042) tiles —
the sanctioned `get_utm_ups_projection` behavior at the poles.

## Label / class mapping

Binary contrail segmentation (single manifest class `contrail`), with a **real background
class** because each scene was exhaustively annotated (so out-of-polygon pixels are genuine
non-contrail context — as in `cabuar_california_burned_areas`, not fabricated negatives):

| id | name          | meaning |
|----|---------------|---------|
| 0  | `no_contrail` | observed Landsat pixel with no contrail annotation |
| 1  | `contrail`    | inside a human contrail polygon |

`dtype=uint8`, `nodata=255` (reserved/unused). 64×64 @ 10 m tiles (640 m), local UTM/UPS.

## Time-range and change handling

A contrail is a **specific-image** feature valid only at the exact Landsat overpass (spec
§5 specific-image rule), **not** a seasonal/annual label and **not** a change event.
`time_range` = a **1-hour window centered on the scene acquisition time**
(`DATE_ACQUIRED` + `SCENE_CENTER_TIME` from the MTL); `change_time` is **null**. Pretraining
will therefore pair each mask only with imagery from that overpass hour — in practice the
Landsat-8 scene itself (or a coincident acquisition). All samples are 2017–2020 (post-2016).

## Tiling and sampling

Contrails span whole 185 km scenes, but a label tile is capped at 64×64 @ 10 m (640 m).
Each tile is **anchored on a point on a contrail polygon's boundary** (not the interior
centroid) so the small tile straddles a contrail edge and contains both classes; all of the
scene's contrail polygons are rasterized into the tile so neighbouring contrails are labeled
too. To maximize spatial/temporal diversity of this global dataset, selection is
**round-robin across scenes** (one tile per scene per round) up to `TARGET_SAMPLES=1000`
(the per-class cap for the single `contrail` class, spec §5). Result: **1000 tiles from
1000 distinct scenes** (each the scene's largest contrail).

- tiles with background present (mixed 0/1): **847**
- all-contrail tiles (wide contrails filling the 640 m tile): **153**
- samples per year: 2018×858, 2019×39, 2020×103.

Contrail is present in every one of the 1000 tiles (pixel split ≈ 50/50 in the 847 mixed
tiles by boundary-anchoring construction).

## Verification (spec §9)

- 5 opened tifs + a 200-tif sweep: all single-band `uint8`, 64×64, 10 m, UTM (a few UPS at
  the poles), pixel values ⊆ {0, 1}, `nodata=255`. All 1000 tifs have a matching JSON.
- Every JSON `time_range` span = exactly 1.0 h; `change_time` null; metadata class ids
  {0,1} cover all values in the tifs.
- **Spatial sanity (thermal, not S2 — contrails are transient/invisible in S2 daytime RGB):**
  warped the parent Landsat-8 B10 (11 µm) and B11 (12 µm) bands onto the tile grid and
  compared the 11−12 µm brightness-temperature difference (BTD) inside vs outside the mask.
  Contrail pixels show the expected **lower/colder BTD** (e.g. 000000: 952 vs 1070 background;
  000050: 81 vs 491), confirming the labels sit on real thermal contrail signatures and the
  georeferencing is correct. (A crude uncalibrated-DN proxy, so faint contrails show weak
  contrast.)
- Re-running is idempotent: selection is deterministic (seeded), existing `locations/*.tif`
  are skipped (second run wrote 0 new tiles, 1000 on disk).

## Caveats

- **Coarse label boundaries.** Annotations were drawn at ~300 m (10×-downsampled 30 m)
  resolution, so contrail mask edges are only accurate to ~±300 m when upsampled to 10 m.
  The where-mask is valid; the exact boundary is approximate.
- **Best sensor is Landsat/thermal.** Contrails are resolvable in Landsat-8 thermal/cirrus
  bands (how they were labeled) and are usually invisible in Sentinel-2 daytime RGB;
  `sensors_relevant` = `[landsat, sentinel2]` with Landsat preferred.
- **~15% all-contrail tiles** (wide spread contrails) have no background pixel — kept (still
  valid presence masks), analogous to `cabuar` fire-only tiles.
- Only the largest contrail of each of 1000 scenes is used (per-class cap); the remaining
  3,417 positive scenes and additional per-scene polygons are unused but available if a
  higher cap is ever wanted (raise `TARGET_SAMPLES` / `N_PER_SCENE`).

## Reproduce

```
# (one-time) gsutil -m cp gs://landsat_contrails_dataset/2023_01_20_1674247800/data/landsat_contrails.json-*-of-00100 \
#   /weka/.../raw/human_labeled_landsat_8_contrails_dataset/shards/
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.human_labeled_landsat_8_contrails_dataset
```

Outputs: `datasets/human_labeled_landsat_8_contrails_dataset/{metadata.json,
registry_entry.json, locations/{000000..000999}.tif+.json}` on weka.
