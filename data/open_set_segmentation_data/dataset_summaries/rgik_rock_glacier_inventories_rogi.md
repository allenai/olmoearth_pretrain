# RGIK Rock Glacier Inventories (RoGI)

- **Slug:** `rgik_rock_glacier_inventories_rogi`
- **Task type:** classification (rock-glacier activity)
- **Source:** Zenodo record [14501398](https://doi.org/10.5281/zenodo.14501398) (v2.0 =
  15467203), Rouyet et al., "Rock Glacier Inventories (RoGI) in 12 areas worldwide using a
  multi-operator consensus-based procedure" (ESA CCI Permafrost / ESSD).
- **License:** CC-BY-4.0 (open, no credential needed).
- **Family / region:** permafrost; 12 areas worldwide (Alps, Andes, Alaska/Brooks Range,
  Central Asia, Scandinavia, Greenland, New Zealand, Svalbard, Carpathians).

## Access

Single ~2.4 MB archive `Rouyet-et-al_RoGI_Zenodo_v2.0.zip`, downloaded via
`download.download_http` and unzipped to `raw/{slug}/extracted/`. It contains one
all-areas GeoPackage with layers:

- `..._AOI_...` — 12 area-of-interest polygons (not used).
- `..._GO_...` — **603 geomorphological-outline MultiPolygons** (rock-glacier landform
  footprints). Attributes include `PolyUID`, `PrimaryID`, `OutType` (Extended | Restricted).
- `..._MA_...` — 575 InSAR "moving area" MultiPolygons with `VelClass` (**not used** — see
  below).
- `..._PM_...` — **631 primary-marker Points** carrying the consensus activity
  classification (`ActiCl`), joined to GO outlines via `PrimaryID`.

## Label / class mapping

The activity class lives on the primary-marker points (`ActiCl`) and is joined onto each
GO outline polygon by `PrimaryID` (all 603 outlines match a marker). The RGIK "uncertain"
qualifier is folded into the base activity class; pure `Uncertain`/null markers are dropped.

| id | name | source `ActiCl` | tiles |
|----|------|-----------------|-------|
| 0 | active | Active, Active uncertain | 261 |
| 1 | transitional | Transitional | 163 |
| 2 | relict | Relict, Relict uncertain | 171 |

Dropped: outlines whose marker `ActiCl` is `Uncertain` (6) or null (2).

Each **GO outline polygon** is rasterized (`rasterize.rasterize_shapes`, `all_touched`) into
a **64×64 UTM 10 m** tile centered on the polygon's representative point: pixels inside the
outline = activity class id, everything outside = **255 (nodata)**. Every tile is thus a
single-class positive mask (tiles-per-class balanced, one class per tile), mirroring the
`global_debris_covered_glaciers_herreid_pellicciotti` recipe. Outside is nodata (not a
background class) because the surrounding terrain is unlabeled, not "a rock glacier of some
other activity".

**Both delineations per landform are used** as separate tiles: the Extended outline (full
landform incl. rooting zone / talus) and, where present, the Restricted outline (main body).
They share a location and class but are distinct source delineations; using both (595 total
vs 336 rock glaciers) maximizes labels for this small, rare-class inventory. ~21% of outlines
exceed 640 m and are clipped to the window (homogeneous interior tile); smaller ones show
their shape against nodata (valid-pixel fraction ranges ~0.02–0.75).

**Moving-area (MA) layer not used:** moving areas are an InSAR kinematic sub-delineation
(velocity classes) that overlaps active rock glaciers and is orthogonal to the per-landform
activity class — mixing it into the class map would create spatial label conflicts. The
kinematic signal is already reflected in the consensus activity class. AOI polygons unused.

## Time range

Rock glaciers are slow landforms; the multi-operator consensus (esp. kinematic attribution)
draws on InSAR over ~2018–2021 (manifest `time_range`). Every sample gets a uniform 1-year
window **2019-01-01 → 2020-01-01** within that observation period. No change labels
(`change_time` null).

## Sample counts

- **595 samples** total: active 261, transitional 163, relict 171.
- Well under the 1000/class and 25k/dataset caps (no truncation).
- Geographic sanity: 60/60 randomly checked tile centers fall inside the 12 RoGI AOIs.

## Verification

- 595 `.tif` + 595 matching `.json`; each tif single-band uint8, UTM CRS at 10 m, 64×64,
  nodata 255. Global unique pixel values = {0, 1, 2, 255} — all valid class ids.
- `metadata.json` class ids {0,1,2} cover all non-nodata values present.
- Time ranges are 1-year. Idempotent (existing `{id}.tif` skipped on rerun).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.rgik_rock_glacier_inventories_rogi
```
(Download + unzip of the Zenodo archive into `raw/{slug}/extracted/` is done by the script /
a one-time `unzip`; re-runs skip existing outputs.)

## Caveats

- Activity is a landform-level attribute mapped uniformly over the whole outline (no
  intra-landform activity gradient).
- Extended/Restricted tiles for the same landform are near-duplicate locations (documented
  augmentation), not independent sites.
- Small relict/transitional rock glaciers may be hard to resolve at 10 m; retained because
  the footprint spans multiple pixels and the geomorphic setting (talus slopes, cirques) is
  visible to S2/S1/Landsat.
