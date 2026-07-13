# xBD / xView2 (building damage)

- **Slug**: `xbd_xview2_building_damage`
- **Registry status**: `completed`
- **Task type**: classification (change / building-damage segmentation)
- **Samples written**: 1262 tiles (`locations/{000000..001261}.tif` + `.json`)
- **Source**: xView2 / xBD Building Damage Assessment Dataset (Gupta et al. 2019), built on
  Maxar Open Data VHR imagery. Official portal <https://xview2.org/dataset> (free signup).
- **License**: CC-BY-NC-SA-4.0.

## Source and access

xBD provides ~850k manually annotated, expert-reviewed building polygons over pre/post-disaster
VHR image pairs across 6 natural-disaster types worldwide, each building tagged with a 4-level
damage subtype. The official download is behind a free-signup portal, so we used a public
Hugging Face mirror
(`WayBob/Disaster_Recognition_RemoteSense_EN_CN_JA`) that bundles the labels with imagery.

We downloaded the tier1 `xview2_train.tar.gz` (8.4 GB) + `xview2_test.tar.gz` (2.8 GB) archives
to `raw/xbd_xview2_building_damage/` and extracted **only** the post-disaster label JSONs
(`*/labels/*_post_disaster.json`, 3732 files, a few tens of MB). VHR image pixels are **not**
used — pretraining supplies its own S2/S1/Landsat imagery. The 18 GB `tier3` archive was **not**
downloaded: tier1 already covers all 6 disaster types post-2016 and yields ample class-balanced
tiles, and tier3's extra events are mostly the pre-2016 tornadoes (joplin 2011, moore 2013,
tuscaloosa 2011, pinery 2015) that the post-2016 filter would drop anyway (spec §8 download
economy).

**Georeferencing** comes directly from the label JSON: each building feature carries a WGS84
`lng_lat` WKT polygon and a `subtype`; per-image `metadata.capture_date` (day-resolved) gives the
event time, and `metadata.disaster`/`disaster_type` the event identity. No image headers were
needed. Verified: sample centroids land in the correct AOIs worldwide (Guatemala/Fuego, Haiti,
Houston, Florida panhandle, Mexico City, Arkansas, Santa Rosa CA, SoCal, Palu Sulawesi).

## Label encoding

**Damage-class scheme (2-class collapse for 10 m observability).** Buildings are ~0.4–1.4 m
native GSD (sub-pixel at 10 m), so a single building's *damage level* is not discriminable at
10 m from Sentinel-scale imagery. Per spec §4 / the manifest note we collapse the native 4-level
scale to a 2-class building-presence×damage scheme that becomes observable when damage clusters
(razed neighborhoods, tsunami-swept / burned zones):

| id | class | xBD subtypes |
|----|-------|--------------|
| 0 | `intact_building` | no-damage + minor-damage |
| 1 | `damaged_building` | major-damage + destroyed |

`un-classified` buildings (damage not assessable) are dropped (not painted). The finer 4-level
counts are retained in `metadata.json`. Native subtype totals over the extracted post-disaster
labels: no-damage 158,853; minor-damage 19,778; major-damage 18,011; destroyed 17,002;
un-classified 4,005.

**Rasterization.** Per post-disaster image, all building polygons are reprojected to a local UTM
projection at 10 m/pixel and rasterized with `all_touched=True` into ≤64×64 patches (each image
is ~50–140 m→px, i.e. one 64×64 tile). Per pixel the most-severe class wins (intact painted
first, damaged last). Non-building ground stays nodata (255) — **positive-only** (spec §5);
downstream assembly supplies negatives from other datasets (no fabricated background class).

## Change label (spec §5)

This is a disaster **before→after** damage dataset. xBD gives, per post-disaster image, a
satellite `capture_date` resolvable to the day, tasked within days of the event. We set
`change_time` = that per-image post-disaster capture date and `time_range` = a 360-day window
**centered** on it (spans the event; ≤360-day cap). The `damaged_building` mask is the
"where the change occurred" signal. This easily meets the §5 timing-precision requirement
(event known to ≪ 1–2 months). No persistent-state recast is used — damage is captured as a
dated change event.

**Post-2016 filter.** Images are filtered to `capture_date` year ≥ 2016. All tier1 disasters are
2016–2019 (hurricane-matthew is Oct 2016), so **none are dropped** by this filter; the only
pre-2016 events (tornadoes) live in the un-downloaded tier3 archive.

## Sample counts

- Selected 1262 tiles (of 2980 candidates) via tiles-per-class balancing (≤1000/class, 25k cap).
- Tiles-per-class: `intact_building` 1000, `damaged_building` 1086.
- Tiles-per-disaster-type: flooding 415, wind 397, fire 346, tsunami 79, earthquake 23, volcano 2.
- Tiles-per-disaster: hurricane-michael 227, socal-fire 202, hurricane-harvey 177,
  hurricane-matthew 170, santa-rosa-wildfire 144, hurricane-florence 138, midwest-flooding 100,
  palu-tsunami 79, mexico-earthquake 23, guatemala-volcano 2.

Disaster dates (post capture, `change_time`): guatemala-volcano 2018-06-22, hurricane-florence
2018-09-18/20, hurricane-harvey 2017-08-31, hurricane-matthew 2016-10-01/11, hurricane-michael
2018-10-13, mexico-earthquake 2017-09-20, midwest-flooding 2019-05-30/31, palu-tsunami
2018-10-01, santa-rosa-wildfire 2017-10-11, socal-fire 2018-11-14.

## Caveats

- Individual buildings are sub-pixel at 10 m; only clustered damage (whole burned/razed/swept
  areas) is genuinely observable — hence the 2-class collapse. The finer 4-level scale is
  preserved only in metadata.
- Positive-only: only building pixels are labeled; intact/damaged distinction plus the dated
  change window is the training signal.
- guatemala-volcano contributes few tiles (small labeled extent); downstream assembly may drop
  the sparsest per-disaster contributions, which is fine.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.xbd_xview2_building_damage
# --probe to scan/report without writing; idempotent (skips existing tiles).
```
Outputs on weka: `datasets/xbd_xview2_building_damage/{metadata.json, locations/*.tif|*.json,
registry_entry.json}`; raw labels under `raw/xbd_xview2_building_damage/labels/`.
