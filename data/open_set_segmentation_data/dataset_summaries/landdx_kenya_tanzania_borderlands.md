# landDX (Kenya-Tanzania Borderlands)

- **Slug**: `landdx_kenya_tanzania_borderlands`
- **Status**: completed
- **Task type**: classification (per-pixel segmentation), **positive-only**
- **Samples**: 1,438 label tiles (64×64, UTM, 10 m/pixel)
- **Classes (unified scheme)**: `0 = livestock_enclosure` (boma/kraal/enkang), `1 = agricultural_land`
- **Class tile counts**: boma-containing 937, agriculture-containing 1,093 (a tile can contain both)

## Source

Landscape Dynamics (landDX) database — an open-access spatial-temporal database for the
Kenya-Tanzania borderlands (Tyrrell et al. 2022, *Scientific Data* 9:8,
doi:10.1038/s41597-021-01100-9). Manual VHR digitization (Google Earth / Bing, ~0.5 m; a
few areas 30 m Landsat) of anthropogenic structures over ~31,000 km² of southern Kenya
(Kajiado + Narok counties, extending toward Amboseli/Tsavo) by SORALO, Kenya Wildlife
Trust (KWT), Aarhus University, and the Mara Elephant Project (MEP).

- **License**: CC-BY-4.0 (open access).
- **Access method**: static release from the Oxford University Research Archive
  (data DOI 10.5287/bodleian:qqv4EdRnQ), file
  `active_public_uncategorized_shpfiles.zip` (74 MB), downloaded via direct HTTP from
  `https://ora.ox.ac.uk/objects/uuid:a733ec4f-20e3-4989-acba-5f85cfd6d0eb/files/ddv13zt283`.
  No credentials required. Raw files under
  `raw/landdx_kenya_tanzania_borderlands/active_shp/` (+ `SOURCE.txt`).

The release contains four shapefiles (WGS84, EPSG:4326):

| shapefile | geom | n | contents |
|---|---|---|---|
| `landDx_polygons` | Polygon | 57,192 | Settlement_Boma 37,040 + Agriculture 20,152 |
| `landDx_polylines` | LineString | 96,879 | Fence_* 94,546 + Road_* 2,324 |
| `landDx_points` | Point | 31,024 | boma centroids (redundant) |
| `landDx_polygons_centroids` | Point | 57,080 | polygon centroids (redundant) |

## Decisions

### Multi-modality → one unified dataset (SOP §5)
The source is polygons (bomas + agriculture) + lines (fencing/roads). Combined into ONE
dataset with a 2-class scheme built from the **polygons** only.

### Fencing dropped (line-observability judgment, SOP §4)
Fencing (94,546 `Fence_*` polylines) was **dropped**. Rationale: fences are **thin line
features** (brush fences a few metres wide; wire/electric fences invisible even in VHR —
mapped only via land-use edges), and the source carries a **~39.7 m Google-Earth
positional RMSE** (Tyrrell et al. 2022, citing Potere 2008). A ~40 m location error on a
sub-10 m-wide line means a dilated 10 m mask would frequently not overlie the real
feature, so fencing is **not reliably observable/alignable at 10–30 m** from
Sentinel/Landsat. Roads (2,324 `Road_*`) are out of the manifest's 3-class scope and also
thin — dropped. Boma points / polygon centroids are redundant with the boma polygons
(which give the true footprint) and were not used.

### Kept classes (observable at 10 m)
- **livestock_enclosure** (Settlement_Boma polygons): 30–150 m cleared enclosures,
  equiv-side median ~25 m (~2–3 px), p95 ~65 m (~6–7 px), with a distinctive bare-earth /
  manure spectral signature → discernible at 10–30 m (per the manifest note).
- **agricultural_land** (Agriculture polygons): field-scale, equiv-side median ~102 m →
  clearly observable.

### Tiling (SOP §4 "polygons … sampled sub-windows")
The study area is partitioned onto a **640 m grid in World Mollweide (ESRI:54009)**; each
occupied cell → one **64×64 UTM 10 m** tile centered on the cell center, into which every
boma/agriculture polygon overlapping the cell is rasterized (`all_touched=True`,
agriculture first then bomas on top so bomas win). 22,251 cells were occupied.
**Positive-only** (SOP §5): non-labeled pixels are nodata/ignore (255); no synthetic
background class — the assembly step supplies negatives from other datasets.

### Time range (SOP §5)
Each feature carries `collect_da` (digitized imagery / ground date). Persistent-ish land
features → `change_time = null`, static 1-year window:
- Dated features in [2016, 2022] → 1-year window on their year.
- Dated features **before 2016** (pre-Sentinel imagery, 2003–2015) → **dropped**
  (7,574 bomas + 4,770 agriculture) per the mixed-dataset triage rule.
- **Undated** features (KWT imagery ≤2017; some SORALO with no GE date stamp, ≤2020) →
  kept with a **2017** window (undated is not known-pre-2016; SORALO weighted-mean date is
  2016-09).
- Per-cell window anchored on the **modal effective year** of the cell's features.

Anchor-year distribution of written tiles: 2016:34, 2017:749, 2018:216, 2019:407, 2020:32.

### Sampling
Tiles-per-class balanced (`sampling.balance_tiles_by_class`, per_class=1000): 1,523
candidate cells selected; 1,438 written (85 rasterized empty — a polygon assigned by bbox
overlap only touched a cell corner and clipped out of the centered tile). Well under the
25k per-dataset cap.

## Output

- `datasets/landdx_kenya_tanzania_borderlands/metadata.json`
- `datasets/landdx_kenya_tanzania_borderlands/locations/{000000..}.tif` (+ `.json`)
- Each `.tif`: single-band uint8, local UTM (EPSG:32736/32737), 10 m, 64×64, nodata=255,
  values ∈ {0, 1, 255}.

## Verification (SOP §9)

- Opened multiple tifs: all single-band uint8, UTM 10 m, 64×64, nodata 255; pixel values
  only {0, 1, 255}. Global pixel counts: boma 42,681; agriculture 1,281,005; nodata
  4,566,362 (bomas small → few pixels, expected).
- Every `.tif` has a matching `.json` with a 365-day `time_range`, `change_time=null`,
  and valid `classes_present`; 0 orphans either way. `metadata.json` class ids (0,1) cover
  all values in the tifs.
- **Spatial sanity**: tile centers land precisely in the Kajiado/Narok pastoral region
  (lon 34.8–37.5, lat −1.3 to −2.9). A full Sentinel-2 pixel overlay was **not** performed
  (no lightweight S2-access utility exists in the shared module and it would require
  external imagery access); georeferencing is written by rslearn's exact `GeotiffRasterFormat`
  encoder and the region is confirmed correct.

## Caveats

- Manual VHR digitization with **~40 m positional error**; small bomas (~25 m median
  equiv-side, 2–3 px) sit near the 10 m limit and may be offset, though larger bomas and
  field-scale agriculture tolerate it.
- **Boma occupancy is seasonal** — a boma mapped in one year's imagery may be absent in an
  adjacent year; the per-feature collect date is used to reduce (not eliminate) mismatch.
- Undated features assigned a 2017 window may be off by a few years from their true imagery
  date; bomas/agriculture are relatively persistent, limiting the impact.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.landdx_kenya_tanzania_borderlands
```

Idempotent (skips already-written tiles). `--probe` scans/reports without writing;
`--per-class N` sets the tiles-per-class target (default 1000).
