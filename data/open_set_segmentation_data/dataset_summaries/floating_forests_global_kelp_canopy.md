# Floating Forests Global Kelp Canopy -- REJECTED (temporal)

- **Slug**: `floating_forests_global_kelp_canopy`
- **Source**: Zooniverse "Floating Forests" citizen-science project, served openly (no
  login, CC-BY-4.0) via the IMAS geoserver.
  - Metadata record: https://metadata.imas.utas.edu.au/geonetwork/srv/metadata/554ef3f6-4f05-4e40-bbf5-1e6dd31d920c
  - WFS: `https://geoserver.imas.utas.edu.au/geoserver/imas/wfs` , layer `imas:TRB_FloatingForests` (GeoJSON, EPSG:4326)
- **Label type**: polygons (consensus outlines of surface giant-kelp *Macrocystis
  pyrifera* canopy, drawn by volunteers on 30 m Landsat scenes).
- **Access**: fully open, no account required. **Not a credential rejection.**

## Decision: REJECT

The openly-downloadable IMAS layer at the manifest URL is **California only** and all
**15276 features** are derived from Landsat scenes acquired **entirely pre-2016**:

    scene_timestamp year distribution -> 1999: 2416, 2000: 2757, 2001: 3215, 2002: 3110, 2013: 3778
    latest acquisition year = 2013; features in the Sentinel era (>=2016) = 0

The AGENT_SUMMARY spec (Section 8, triage) lists *"temporal coverage entirely pre-2016
(outside Sentinel era) with no usable window"* as an explicit rejection reason.

**Why there is no usable window:** surface kelp canopy is among the most temporally
dynamic marine habitats -- strong seasonal cycles (summer/autumn peak, winter storm loss)
and dramatic interannual collapse/recovery (e.g. the 2014-2016 NE Pacific marine heatwave
removed >90% of northern-California canopy). A canopy polygon mapped from a 1999-2013
Landsat scene does **not** indicate kelp presence at that location in 2016+, so the labels
cannot be relocated onto a Sentinel-era imagery window. The labels are also intrinsically
specific-image (per-scene, per-date) labels: each is tied to one Landsat acquisition and
could only be paired with imagery from that same pre-Sentinel-2 date.

## Data characteristics (for reference, had it been in the Sentinel era)

- 15,276 MultiPolygon features across 413 Landsat scenes.
- Per-scene nested polygons at multiple `threshold` levels (minimum number of volunteers
  who classified a pixel as kelp) -- a consensus/confidence axis; a single mid threshold
  per scene would be chosen to avoid nested duplicates.
- Fields: `global_fid, threshold, zooniverse_id, scene, classification_count, image_url,
  tile corner lon/lat, scene_timestamp, created_at, geom`.
- Binary target would have been: 0 = background, 1 = kelp_canopy; polygons rasterized to
  <=64x64 UTM 10 m tiles, plus background-only negative tiles.

## Note on the manifest entry

The manifest lists `time_range: [2016, 2019]` and `region: California, Tasmania,
Falklands, others`. The specific openly-available IMAS layer at the manifest URL does not
match this: it is California-only, 1999-2013 (30 m Landsat). The broader global Floating
Forests product (Tasmania, Falklands) is likewise Landsat-based (30 m, Landsat 5/7/8/9)
and is distributed via kelpwatch.org, not this record.

## QUESTION FOR USER

If OlmoEarth pretraining pairs labels with **same-date Landsat** imagery (Landsat 5/7 for
1999-2002, Landsat 8 for 2013) rather than strictly Sentinel-era imagery, this dataset is
recoverable: ~413 scenes could be processed (pick one consensus `threshold` per scene,
rasterize kelp=1 into <=64x64 UTM 10 m tiles with per-scene specific-date time ranges,
plus background negatives). Please confirm whether pre-2016 Landsat-dated labels are in
scope, and/or point to a Sentinel-era (2016+) Floating Forests / kelp-canopy release if
you prefer one; I can then process it.

## Reproduce

    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.floating_forests_global_kelp_canopy
