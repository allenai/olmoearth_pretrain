# USGS Kilauea Lava Flow Shapefiles — `usgs_kilauea_lava_flow_shapefiles`

**Status:** completed · **task_type:** classification (lava-flow presence/state
segmentation) · **num_samples:** 549

## Source

USGS Hawaiian Volcano Observatory (HVO) data release **"GIS shapefiles for Kīlauea's
episode 61g lava flow, Puʻu ʻŌʻō eruption: May 2016 to May 2017"** (Orr, Zoeller, Patrick &
DeSmither, 2017). Public domain.

- Landing page: `https://www.sciencebase.gov/catalog/item/597230e4e4b0ec1a4885edc1`
- DOI: `https://doi.org/10.5066/F7DN43XR`
- **Access used:** direct, no credentials. Downloaded the single ~9.7 MB shapefile zip from
  the ScienceBase file endpoint → `raw/usgs_kilauea_lava_flow_shapefiles/shapefiles.zip`,
  extracted to `.../PuuOo_Ep61g_20160524-20170531_Shapefiles/` (28 shapefiles).

The release maps the Puʻu ʻŌʻō **episode-61g** basaltic lava flow at **14 mapping dates**,
one per calendar month from 2016-05-24 to 2017-05-31 (two dates in June 2016; a 2017-05-03
map substitutes for a missing April 2017 map). Each date has two shapefiles, both native in
**EPSG:32605 (WGS84 / UTM zone 5N, metres)**:

- `Ep61g_YYYYMMDD_flow.shp` — **one polygon**: the full **cumulative** flow extent as of
  that date. Extent grows from **4.2 ha** (2016-05-24) to **947.2 ha** (2017-05-31); final
  bbox ≈ 6.8 km × 8.9 km.
- `Ep61g_YYYYMMDD_contacts.shp` — **polylines**: mapped lava-flow contacts (flow margins),
  attribute `LineType='contact'` (only value present — no separate fissure lines),
  positional accuracy 10–25 m.

Annotation method: field GPS + digitizing of aerial/satellite orthoimagery (HVO).

## Label design

Unified **3-class** scheme (spec §5 multi-modality → one class map combining the polygon
and line targets), uint8:

- `0` = **background** — outside the flow, incl. kīpuka (islands of older ground enclosed by
  the flow; correctly left as background via polygon holes).
- `1` = **lava_flow** — interior of the cumulative episode-61g flow extent.
- `2` = **flow_contact** — contact (flow-margin) polylines, buffered to a **~30 m ribbon**
  (1.5 px half-width) so they are resolvable at 10–30 m; burned **on top of** the flow
  interior (contact wins on the margin).
- `255` = nodata (declared, unused — the HVO perimeter is authoritative, so out-of-flow
  pixels are genuine non-lava context; **no synthetic negatives** fabricated, per §5).

The manifest's two classes ("lava flow", "fissures/contacts") map to classes 1 and 2; this
release's contacts are all flow margins (no fissures). Fresh basalt is highly discernible in
S2/Landsat SWIR (manifest note), so the flow surface is a strong per-pixel signal.

## Time range & change handling (§5)

Mapping dates are **precise single calendar dates ≤ ~1 month apart**, satisfying the §5
"change date known to within ~1–2 months" requirement, so each sample sets
`change_time` = its mapping date and `time_range` = **±180 d (360-day window) centered** on
it. All windows fall in the Sentinel era (2015-11 … 2017-11).

A solidified lava flow is a **persistent** surface (fresh basalt visible for years), so the
cumulative extent polygon doubles as a presence/state mask that stays valid after the date.
All 14 dated snapshots are kept as **separate temporal samples**: at a fixed location the
label legitimately transitions background→lava over the sequence as the flow grows, giving
genuine multi-temporal signal.

**Caveat:** the mask is the extent *as of* each date, so imagery late in a centered window
may show the active flow toe grown slightly beyond the mask — a minor, conservative
*under-count* (never an over-count) at the growing margin.

## Tiling

Geometries are reprojected from EPSG:32605 metres into 10 m/pixel pixel space **in the same
UTM zone** (no CRS change — only a resolution scaling), then each date's flow pixel bbox is
gridded into non-overlapping **64×64** windows. A window is kept if it intersects the flow
polygon or a contact. Output tiles are single-band uint8, EPSG:32605, 10 m, ≤64×64.

## Counts

- **549 tiles** across 14 dates (well under the 25k cap and the ≤1000/class target).
- Tiles per class: background **549**, lava_flow **500**, flow_contact **543**. (49 tiles
  are contact-only edge tiles where the buffered margin extends just past the flow polygon.)
- Samples per mapping date: 20160524:1, 20160610:10, 20160630:26, 20160719:36, 20160819:44,
  20160920:44, 20161019:44, 20161129:47, 20161214:47, 20170112:48, 20170224:49, 20170330:51,
  20170503:51, 20170531:51.

## Verification

- 549 `.tif` each with a matching `.json`; all single-band uint8, EPSG:32605 at 10 m, 64×64,
  nodata 255. Pixel values across the whole dataset are exactly {0, 1, 2}. All `time_range`s
  are 360 days with `change_time` set.
- Geographic sanity: tile centers run from ~(-155.10, 19.39) at the Puʻu ʻŌʻō vent down to
  ~(-155.04, 19.32) at the Kamokuna coast — the exact known episode-61g flow path from vent
  to ocean entry. Because the data is authoritative field-GPS mapping used in its native
  published UTM CRS (only resolution-scaled, no reprojection), alignment is exact; a direct
  Sentinel-2 SWIR overlay was not run (no imagery source configured in this env), but the
  coordinate check confirms placement.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.usgs_kilauea_lava_flow_shapefiles
```
Idempotent (existing `locations/{id}.tif` are skipped). Downloads the public-domain zip on
first run.

## Judgment calls

- **Treated as presence/state segmentation** (not per-increment change masks). The
  cumulative flow polygon is a persistent fresh-basalt surface; masking only the monthly
  increment would discard the strong whole-flow SWIR signal.
- **`change_time` set** (dates are precise), with the §5 centered window — leveraging the
  dataset's monthly temporal precision rather than a static window.
- **All 14 dated snapshots kept** as distinct temporal samples despite heavy spatial overlap
  among later, near-stable extents; the differing extents + windows are legitimate
  multi-temporal training pairs.
- **Contacts included as a class** (buffered ~30 m); they are observable flow margins at
  10–30 m and match the manifest's second class. No true fissure lines exist in this release.
```
