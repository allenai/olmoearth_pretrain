# Pacific Walrus Coastal Haulouts

- **Slug:** `pacific_walrus_coastal_haulouts`
- **Status:** completed
- **Task type:** classification (single foreground class, positive-only)
- **Samples written:** 887 label tiles (from 631 dated herd-outline images)
- **Family / region:** wildlife / Chukchi Sea (Alaska + Chukotka, Russia)
- **License:** CC0 1.0 (public domain)

## Source

USGS Alaska Science Center data release **"Pacific Walrus Coastal Haulout Occurrences
Interpreted from Satellite Imagery"** (Fischbach & Douglas 2022, ver 6.0 December 2025),
doi:[10.5066/P9CSM0KN](https://doi.org/10.5066/P9CSM0KN), distributed openly on ScienceBase
(parent item `6441f010d34ee8d4ade7edcc`). No account/credential required (CC0).

Trained interpreters delineated the **extent of walrus herds** (*Odobenus rosmarus
divergens*) resting on shore at eight known Chukchi Sea coastal haulout sites, autumn
2017–2025, using a mix of optical and SAR satellite imagery: Sentinel-2, Sentinel-1,
PlanetScope, Maxar/DigitalGlobe, TerraSAR-X, RADARSAT-2, Umbra, Capella, Iceye.

### Access method / layout

The release is organised as per-site, per-year ZIP packages (8 sites × 2017–2025),
each child ScienceBase item. Each ZIP contains:
- `walrus_dailySatelliteHauloutOutlines/shape/*.shp` — **one shapefile per satellite
  image** in which a walrus herd was apparent; each holds 1–8 herd sub-group polygons,
  geocoded in the site's **local UTM** CRS. This is the label layer we use.
- `walrus_dailySatelliteMaps/*.jpg` — rendered maps (imagery + outline); not needed.
- `README_MethodsAndImageProcessing.pdf`.

Shapefile filenames encode acquisition time + mission:
`[YYYYMMDD]T[hhmm|hhmmss]Z_[mission]` (e.g. `20221008T234631Z_S2`). A top-level
`walrus_hauloutAreaEstimates_chukchi.csv` lists every image examined and the summed herd
area (downloaded for provenance only).

The download step enumerates child items via the ScienceBase JSON API at runtime
(reproducible, no hard-coded disk-hash URLs), downloads each ZIP (~1.4 GB total) and
extracts only the thin outline shapefiles. Full ZIPs are retained under
`raw/pacific_walrus_coastal_haulouts/zips/`; extracted shapefiles under `.../outlines/`.

## Label / class mapping

Single-class, **positive-only** segmentation (spec §5 — herds were outlined where
walruses were apparent; absence of an outline is *not* a verified negative, so we do
**not** fabricate negatives — the assembly step supplies them from other datasets):

| id | name | meaning |
|----|------|---------|
| 0 | walrus haulout / herd extent | interior of a digitised herd-extent polygon |
| 255 | nodata / ignore | everything else |

Herd sub-polygons within one image are unioned into a single foreground mask. Rasterised
with `all_touched=True` so small/thin herds stay visible at 10 m.

## Time-range handling

Each outline was interpreted from **one dated satellite image**, and walruses are mobile,
so the herd extent is only valid at that acquisition instant (a year-long window would
pair imagery that shows no walruses). Following spec §5 (specific-image / dated-detection
labels; same convention as the Sentinel-2 vessels dataset), each sample's `time_range` is
a **~1-hour window (±30 min) centered on the image acquisition datetime** parsed from the
shapefile filename. `change_time` is null (this is dated *presence*, not a change event).
All 887 samples have a 3600 s time range (≤ 1 year). All labels are post-2016 (2017–2025),
so no pre-Sentinel filtering was needed.

Of 887 tiles, 435 (49%) derive from Sentinel-1/Sentinel-2 images (directly pairable with
pretraining's own S1/S2 within the 1-hour window); the rest derive from commercial
optical/SAR missions whose labels remain correct but may not find a matching pretraining
image in-window.

## Tiling

Local UTM projection at 10 m/pixel, chosen per-image from the herd-union centroid lon/lat
(`get_utm_ups_projection`). Herds fitting in a 64×64 tile (640 m) → one centered tile;
large/elongated haulouts (many are long thin beach strips, up to ~2.5 km) are gridded into
non-overlapping 64×64 windows, up to 20 kept per image. Selection is round-robin across
images (every image contributes ≥1 tile) capped at 25,000 (never reached). All tiles are
single-band uint8, 64×64, north-up UTM at 10 m.

## Sample counts

- **By site:** PointLay 500, CapeSerdtseKamen 191, Vankarem 109, CapeIkigur 72,
  CapeBlossom 15. (The other three sites — IcyCape, Somnitel'naya Spit, Chegitun River
  Mouth — had no walrus outlines in the available years.)
- **By year:** 2017:92, 2018:113, 2019:93, 2020:98, 2021:40, 2022:87, 2023:221,
  2024:82, 2025:61.
- **By source mission:** S1 247, PS 194, S2 188, US(Umbra) 154, RS 43, CS(Capella) 29,
  TS 20, DG(Maxar) 9, IE(Iceye) 3.
- **Class balance:** class 0 present in all 887 tiles (single positive class); pixel
  values across all tiles are exactly {0, 255}.

## Verification (spec §9)

- Opened output tifs: all single-band, uint8, 64×64, UTM-N CRS (EPSG:32601/32602/32603)
  at 10 m, values ∈ {0, 255} (class 0 + nodata 255). ✔
- Every `.tif` has a matching `.json`; all `time_range`s are 3600 s (≤ 1 year);
  `change_time` null everywhere; `metadata.json` class ids cover all values present. ✔
- **Geographic plausibility:** tile-center lon/lat match documented haulout coordinates —
  Vankarem ~0.3 km, PointLay ~2.9 km, CapeSerdtseKamen ~9.5 km. ✔
- Re-running the script is idempotent (0 tiles rewritten; existing `.tif` skipped). ✔

## Caveats

- **Georeferencing:** USGS notes that cross-mission image georeferencing can shift
  outlines by tens to >100 m from the true coastline (uncorrected in the source). Labels
  are placed by each shapefile's own georeferencing; expect minor coastline offsets.
- **`CapeBlossom_*` package coordinates:** its shapefiles are geocoded on the western
  Chukotka mainland coast (~66.9 N, 171.7 W, EPSG:32602), **not** at the Wrangel-Island
  "Cape Blossom" coordinate (70.78 N, 178.77 E) given in the site metadata — a source-side
  filing quirk. Tiles are placed at the shapefiles' actual (verified) georeferencing, so
  labels are correctly located regardless of the site name in `source_id`.
- **Source-side missing files (retry candidates):** 9 per-site/year ZIPs return HTTP 404
  and `CapeSerdtseKamen_2025.zip` is 0 bytes on ScienceBase. Of these, only
  **Vankarem_2023** (~4 herd images) and **CapeSerdtseKamen_2025** (~12 herd images)
  carry labels; the other 7 (IcyCape 2017/2018/2022, Somnitel'naya Spit 2023, Cape Inkigur
  2022, Chegitun 2017/2020/2024) had no walruses (empty outline folders). ~16 herd images
  (~2%) are therefore temporarily missing — re-running the script once the source restores
  these files will add them (idempotent).
- **`ESRI Shapefile.shp` artifacts:** 7 shapefiles were exported with the driver name as
  the filename and carry no parseable acquisition timestamp; they are skipped to preserve
  per-image time integrity.
- **Positive-only:** no negatives are emitted; sparse-negative supply is handled at
  assembly time (spec §5).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.pacific_walrus_coastal_haulouts --workers 64
```

Outputs:
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/pacific_walrus_coastal_haulouts/`
(`metadata.json`, `locations/{000000..}.tif` + `.json`, `registry_entry.json`); raw source
under `.../raw/pacific_walrus_coastal_haulouts/`.
