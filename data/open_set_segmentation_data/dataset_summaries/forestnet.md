# ForestNet

- **Slug:** `forestnet`
- **Status:** completed — classification, 2,757 samples
- **Source:** Stanford ML Group, "ForestNet: Classifying Drivers of Deforestation in
  Indonesia using Deep Learning on Satellite Imagery" (Irvin & Sheng et al., NeurIPS 2020
  Tackling Climate Change with ML workshop). <https://stanfordmlgroup.github.io/projects/forestnet/>
- **License:** CC-BY-4.0.
- **Region:** Indonesia. **Family:** deforestation. **label_type:** polygons.

## What the source is

2,757 primary-forest-loss events in Indonesia. Global Forest Change (GFC) annual
forest-loss maps (2001–2016, 30 m) were used to obtain each loss event as a polygon with
an associated **loss year**; an expert interpreter then annotated each event with the
**direct deforestation driver** using high-resolution Google Earth imagery. Each event is
distributed as a 332×332 px Landsat-8 image (visible bands pan-sharpened to 15 m/px)
centred on the loss region, with:

- `forest_loss_region.pkl` — a shapely (Multi)Polygon delimiting the GFC loss region, in
  the **332×332 image-pixel grid** (NOT lon/lat); image-centre pixel (166, 166)
  corresponds to the CSV `(latitude, longitude)`.
- `train.csv` / `val.csv` / `test.csv` — one row per event: fine `label`, coarse
  `merged_label`, image-centre `latitude`/`longitude`, GFC `year`, `example_path`.

## Access / download

Single ~3.4 GB zip: `http://download.cs.stanford.edu/deep/ForestNetDataset.zip` (public,
no credentials). Only labels are extracted — the CSVs and `examples/*/forest_loss_region.pkl`;
the Landsat imagery and auxiliary layers are **not** extracted (pretraining supplies its
own imagery). All three splits (train/val/test) are used as pretraining labels.

## Task decision — presence/state classification (NOT a change label)

This is a deforestation-driver dataset; the driver date is only the **GFC annual loss
year**, which is coarser than the spec's ~1–2 month change-timing requirement. Rather than
reject it or force it into a dated change scheme, we take the spec-§5 "persistent
post-change state → presence/state classification" path:

- The mapped driver (oil-palm / timber / smallholder agriculture / grassland / …) is a
  **persistent post-deforestation land-use state** that stays visible for years after the
  clearing.
- `change_time = null`; each sample gets a **static 1-year window**
  `year_range(max(loss_year + 1, 2016))` = **2016 or 2017**, i.e. a Sentinel-2-era window
  that observes the persistent land-use state. This also satisfies the post-2016 rule: the
  labelled *state* (not the historical 2001–2016 loss event) is what the imagery window
  sees, so the events are not rejected as pre-2016.
- **Caveat:** for older loss events (pre-2012) the driver is *assumed* to persist to
  2016/2017. This is reliable for stable land uses (plantations, smallholder agriculture)
  but grassland/shrubland and secondary-forest states may have transitioned in the
  interim. Noted here rather than dropped.

## Label encoding

- One single-band **uint8** GeoTIFF per event, local UTM, **10 m/px**, north-up.
- The loss polygon (15 m image-pixel grid) is affine-transformed to the 10 m UTM grid
  (scale ×1.5 = 15 m/10 m, translated so image-centre px (166,166) → the event's UTM
  centre pixel), then rasterized (`all_touched=True`) with its fine driver class id.
- Pixels **outside** the loss polygon are **255 = nodata/ignore** (no fabricated
  background/negative class — the land use outside the mapped loss region is unknown;
  spec §5 for positive-only datasets). Downstream assembly supplies negatives from other
  datasets.
- Tile size = footprint + 10 px context ring, clamped to **32–64 px**; polygons larger
  than 64 px are clipped to the central 64×64 window; sub-pixel polygons fall back to the
  centre pixel. (Result: 1,963 tiles at 64×64, the rest 32–62 px.)

## Classes (12 fine drivers, ids by descending frequency)

Fine driver classes are used (richer / Indonesia-specific vs the Amazon
`olmoearth_forest_loss_driver`); each records its ForestNet coarse group in its
`metadata.json` description.

| id | name | coarse group | count |
|----|------|--------------|-------|
| 0 | oil_palm_plantation | Plantation | 599 |
| 1 | small_scale_agriculture | Smallholder agriculture | 576 |
| 2 | timber_plantation | Plantation | 387 |
| 3 | grassland_shrubland | Grassland shrubland | 265 |
| 4 | small_scale_mixed_plantation | Smallholder agriculture | 192 |
| 5 | other_large_scale_plantations | Plantation | 187 |
| 6 | small_scale_oil_palm_plantation | Smallholder agriculture | 144 |
| 7 | secondary_forest | Other | 119 |
| 8 | other | Other | 93 |
| 9 | mining | Other | 90 |
| 10 | logging | Other | 60 |
| 11 | fish_pond | Other | 45 |

Max per-class 599 < 1000 and total 2,757 « 25,000, so **all events are kept** — no
class balancing or truncation. Nodata value = 255.

## Verification

- 2,757 `.tif` + 2,757 `.json`; all single-band uint8, UTM CRS at 10 m, size ≤ 64,
  values ∈ {0–11, 255}. All `time_range`s are exactly 1 year; all `change_time` are null.
- **Georeferencing:** tile centres land within 5–18 m of the CSV image-centre coordinates
  (correct UTM zones N & S), confirming the pixel→UTM affine.
- **Spatial sanity (Sentinel-2 overlay):** for sample `000027` (oil_palm, 2015 loss →
  2016 window) the label mask precisely covers a cleared/converted (brown) block aligned
  to the forest/clearing boundary against adjacent intact forest (green), confirming both
  the ×1.5 scale and placement. Other spot checks were consistent (some limited by cloud
  cover / partial S2 scene footprints).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.forestnet --workers 64
```

Idempotent: re-running skips already-written `locations/{id}.tif`. Raw download +
extracted labels live at
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/forestnet/`.

## Notes / caveats

- Related but distinct: `cam_forestnet_congo_basin_drivers` (Cameroon analogue, kept as
  dated change labels) and `olmoearth_forest_loss_driver` (Amazon, GLAD alerts with
  precise dates → change labels). ForestNet is Indonesia-specific and, lacking sub-annual
  dates, is recast to presence/state here.
- The polygon is in the 15 m Landsat visible-band image grid; resolution taken from the
  paper/site ("332×332 px … 15 m per-pixel"). The paper's "≈5 km²" area statement is
  inconsistent with 332×15 m and was not used.
