# Cam-ForestNet (Congo Basin drivers)

- **Slug:** `cam_forestnet_congo_basin_drivers`
- **Task:** classification (per-pixel; polygon-rasterized change labels)
- **Samples:** 3108 (all events kept; 0 dropped)
- **Source:** Zenodo record [8325259](https://zenodo.org/records/8325259) — "Labelled
  dataset to classify direct deforestation drivers in Cameroon" (de Bus et al. 2023,
  *Scientific Data*), the Cameroon / Congo-Basin analogue of ForestNet.
- **License:** CC-BY-4.0 (open, no credential needed).

## Source

Each example is a Global-Forest-Change (GFC) forest-loss patch in Cameroon, labelled by an
expert (multi-dataset overlay + manual verification) with the **direct deforestation
driver** that caused the loss. We used the `my_examples_landsat_final_detailed.zip`
release (Landsat-8, detailed 15-class scheme) plus `labels.zip`
(`Landsat final versions/detailed/all.csv`). Each event folder is named `{lon}_{lat}` and
contains `forest_loss_region.pkl` — a shapely `Polygon` in **EPSG:4326 (lon/lat)**
delimiting the GFC loss region. The CSV supplies `label` (driver), `latitude`,
`longitude`, `year` (GFC loss year), and `example_path` (which maps 1:1 to the pkl
folder — all 3108 rows matched, no duplicates).

Access notes: the examples zip uses a compression method Python's `zipfile` cannot decode,
so extraction shells out to system `unzip` (only the `forest_loss_region.pkl` files are
extracted; the RGB/aux/NCEP layers are not needed). We chose the **detailed** (15-class)
Landsat release over the 4-group scheme and over the PlanetScope release (Planet imagery
is under the NICFI license and is not needed — we only use the driver labels + GFC
polygons).

## Label / class mapping

Output labels are single-band uint8, local UTM, 10 m/pixel, 64×64 (640 m). One tile per
event, centred on the loss-polygon centroid. The forest-loss polygon is rasterized
(`all_touched=True`) with its **driver class id**; everything outside the polygon is
**background (0)**. Sub-pixel polygons (~3.8% of events, <10 m) fall back to labelling the
single centre pixel. Polygons larger than 640 m (~2.5% of events; one degenerate ~111 km
outlier) are clipped to the central 64×64 window.

Class ids: `0 = background` (forest / other land cover surrounding the loss patch), then
the 15 detailed drivers `1..15` assigned by descending event frequency:

| id | class | events |
|----|-------|--------|
| 0 | background | (pixel-only; no event is labelled background) |
| 1 | selective_logging | 546 |
| 2 | timber_plantation | 493 |
| 3 | small_scale_maize_plantation | 385 |
| 4 | small_scale_oil_palm_plantation | 271 |
| 5 | mining | 215 |
| 6 | oil_palm_plantation | 192 |
| 7 | wildfire | 152 |
| 8 | small_scale_other_plantation | 147 |
| 9 | rubber_plantation | 135 |
| 10 | hunting | 132 |
| 11 | other_large_scale_plantations | 127 |
| 12 | other | 100 |
| 13 | grassland_shrubland | 97 |
| 14 | fruit_plantation | 63 |
| 15 | infrastructure | 53 |

`background` appears as pixels in essentially every tile (the still-forest surroundings of
each loss patch) but no event is *labelled* background, so its event count is 0. Some
large-polygon tiles cover the full 64×64 window and contain only their driver class (no
background pixels).

Max per-class count (546) is below the 1000/class target and the total (3108) is far below
the 25k cap, so **all events are kept — no balancing or truncation**, and no classes were
dropped (15 ≤ 254-class uint8 limit).

## Time range & change handling

These are pre/post forest-loss **events**, encoded under the **pre/post change scheme**
(spec §5). GFC loss is only **year-resolved**, so each sample carries two independent
six-month windows (each ≤ 183 days) with `time_range` = **null**:

- `pre_time_range` = **summer of (loss_year − 1)**.
- `post_time_range` = **summer of (loss_year + 1)**.
- so the **entire ambiguous loss year sits in the gap** between the two windows.
- `change_time` = **1 July of the loss year** (reference only).

**Previously rejected; now resolved by pre/post windows.** Because GFC loss is only
year-resolved the event is not resolvable to within ~1–2 months, which is why this dataset
was originally **rejected** on change-timing grounds. Under the pre/post scheme the coarse
year-level timing is bracketed in the gap between the far-apart pre/post windows, so the
dataset is **completed / usable**.

Events span **2015–2020** (year counts: 2015=349, 2016=425, 2017=607, 2018=249, 2019=481,
2020=997). Every `post_time_range` is therefore ≥ 2016 (Sentinel era); the year-1
`pre_time_range` for 2015 events falls in the Landsat-8 era, which is acceptable. **0 events
dropped.**

## Verification

- 3108 `.tif` + 3108 `.json`; every tif single-band uint8, UTM (e.g. EPSG:32632/32633) at
  10 m, 64×64, nodata=255, values are valid class ids.
- All 3108 sample JSONs have `time_range` = **null** with `pre_time_range` and
  `post_time_range` each ≤ 183 days and `change_time` (1 July of the loss year) set between
  them (0 bad entries).
- Georeferencing sanity: tile-center lon/lat reprojected back to WGS84 matches the source
  CSV event coordinates to ~4 decimals and lands inside the Cameroon bounding box for all
  spot-checked samples.
- Idempotent: re-running skips existing `{sample_id}.tif`.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cam_forestnet_congo_basin_drivers
```

Raw source (downloaded + extracted) lives at
`raw/cam_forestnet_congo_basin_drivers/` on weka
(`my_examples_landsat_final_detailed.zip`, `labels.zip`, extracted pkls under
`extracted/`, CSV under `extracted_labels/`).

## Caveats

- `background` is the forest surrounding each loss patch (not a driver); background pixels
  dominate most tiles. Large-polygon tiles may be entirely one driver class.
- Loss-polygon geometry is from GFC (30 m), rasterized to 10 m; footprints are approximate.
- Full S2 overlay eyeballing was not run; georeferencing was verified via coordinate
  round-trip against the source CSV instead.
