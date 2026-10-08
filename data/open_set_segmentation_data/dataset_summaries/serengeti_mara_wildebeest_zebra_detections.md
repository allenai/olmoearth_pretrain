# Serengeti-Mara Wildebeest/Zebra Detections — REJECTED

- **Slug**: `serengeti_mara_wildebeest_zebra_detections`
- **Manifest name**: Serengeti-Mara Wildebeest/Zebra Detections
- **Source**: Wu et al., *Nature Communications* 2023, "Deep learning enables
  satellite-based monitoring of large populations of terrestrial mammals across
  heterogeneous landscapes" — code + released detections on Zenodo
  (`zijing-w/Wildebeest-UNet`, DOI 10.5281/zenodo.7810487).
- **Family / label_type**: wildlife / points (+ some sample masks)
- **License**: Zenodo open (released point/mask data). The underlying Maxar VHR imagery
  is *not* redistributable (NextView EULA), but only the label points are needed here.
- **Final status**: **rejected** — reason: **phenomenon not observable at 10-30 m, and no
  aggregate/mask representation salvages it** (§8). Not a credential or geocoordinate
  problem (data is open and fully georeferenced).
- **task_type**: n/a (rejected); would have been detection/points if accepted.
- **num_samples**: 0.

## What the source actually is

VHR-derived **animal detection points**: the authors trained a U-Net to detect individual
wildebeest (and, in the paper, zebra) in ~0.3–0.5 m GeoEye-1 / WorldView-2/3 satellite
imagery over the Serengeti-Mara ecosystem, and released the **detected point locations**
(not the imagery). The Zenodo record is a 15 MB snapshot of the GitHub repo; the labels
live in `Results_Detected_wildebeest_dataset/` as six ESRI point shapefiles, one per
source scene, each a `FeatureCollection` of `id` + point geometry (no per-animal species
or count attribute):

| shapefile        | sensor / date | # points | extent (km) |
|------------------|---------------|---------:|-------------|
| GE1_20090811     | GeoEye-1 2009-08-11  | 125,993 | 20 × 13 |
| GE1_20100924     | GeoEye-1 2010-09-24  |  77,758 | 33 × 16 |
| GE1_20130810     | GeoEye-1 2013-08-10  | 160,260 | 28 × 32 |
| WV3_20150717     | WorldView-3 2015-07-17 | 16,174 | 15 × 18 |
| **GE1_20180802** | GeoEye-1 2018-08-02  |  51,795 | 30 × 29 |
| **WV2_20201008** | WorldView-2 2020-10-08 |  70,822 | 34 × 61 |

All are **UTM Zone 36S (EPSG:32736)**, so georeferencing is exact (lon/lat recoverable) —
the "no geocoordinates" rejection does **not** apply. The two **bold** scenes are the
post-2016 (Sentinel-era) subset (~122.6k points), consistent with the manifest
`time_range [2018, 2020]`; the 2009/2010/2013/2015 scenes are pre-2016 and would be
dropped under §8's mixed-era rule.

Note on classes: the manifest lists `["blue wildebeest", "plains zebra"]`, but the
**released** point data is wildebeest-only and carries **no species attribute** (only an
`id`), so the two manifest classes are not separable from this release. The zebra layer is
not in the open data.

## Observability triage — FAILS (this is the rejection ground)

The manifest note is explicit: *"VHR-annotated locations; use for habitat/context, not
direct detection at 10-30 m."* Each label marks **one ~2 m animal** (~1.5–3 m² footprint),
one to two orders of magnitude below the 10–30 m ground sampling distance of
Sentinel-2/Sentinel-1/Landsat. Individual animals are unresolvable, so per §4/§8 the only
possible salvage is an **aggregate herd-density / presence mask at 10 m**. I inspected the
released points to test that — and it does not hold up:

1. **Animals are mostly dispersed at 10 m, not clustered into detectable herds.** Binning
   the two post-2016 scenes onto a 10 m grid:
   - **2018 (GE1)**: 51,795 animals in 21,405 occupied 10 m pixels → **mean 2.4, median 2**
     animals/pixel, **max 26**. Only **12%** of occupied pixels have ≥5 animals and **1.9%**
     have ≥10.
   - **2020 (WV2)**: 70,822 animals in 46,609 occupied pixels → **mean 1.5, median 1**,
     **max 18**. Only **2.6%** of occupied pixels have ≥5 animals and **0.3%** have ≥10.

   Even the densest pixels (≤26 animals × ~1.5–3 m² ≈ 10–25% areal cover by dark bodies
   against bright, heterogeneous savanna grass/soil/shadow) do not yield a spectrally
   distinct, reliably-detectable "herd" signal at Sentinel-2's radiometry, and such pixels
   are a tiny (<2%) minority. A density mask built from this would be dominated by 1–2
   animal pixels that carry no observable signal — a mask of noise, not of an observable
   phenomenon.

2. **Decisive: the labels are instantaneous snapshots of a *mobile* herd, so they cannot
   be co-located with pretraining imagery.** Each shapefile is a single VHR acquisition on
   one date. Migrating wildebeest move kilometres per day, so the herd's position is valid
   only at that exact instant. OlmoEarth pretraining pairs each label with an
   **independently-acquired** Sentinel/Landsat scene by geography + a time window; there is
   essentially zero chance the paired scene was captured at the same moment, so the animals
   would **not** be in the labeled pixels of the pretraining image. Unlike a burn scar,
   clear-cut, or filled reservoir, an animal herd leaves **no persistent post-event state**
   in the landscape, so §5's "persistent state → recast as presence/state classification"
   escape hatch does not apply either. This spatiotemporal mismatch is fatal independent of
   the density argument.

Because the phenomenon is unobservable at 10–30 m **and** no aggregate/mask representation
salvages it (dispersed density + non-persistent, un-co-locatable mobile snapshots), this is
a clean **reject** under §8, matching the manifest's own guidance.

## Access / disk (for completeness)

- Access is fully open: `download.download_zenodo("7810487", raw_dir)` fetched the 15 MB
  archive with no credential. The point shapefiles were extracted to
  `raw/serengeti_mara_wildebeest_zebra_detections/shapefiles/` for the triage inspection.
  No large / VHR imagery was downloaded.
- Disk at triage: ~32.6 TB free on `/weka/dfive-default` (≥5 TB precondition satisfied).

## What (if anything) could change this verdict

Nothing with the released data. A usable wildlife label here would require either (a)
sustained, spatially-fixed super-aggregations that persist across a multi-day imaging
window (not the case for free-ranging migratory ungulates), or (b) a habitat proxy
(e.g. grazing-lawn / trampling land-cover change) with its own georeferenced labels — which
this dataset does not provide. The dataset remains valuable for VHR-native animal detection,
just not as a 10–30 m open-set-segmentation label.

## Reproduce the triage

```bash
# Inspect the Zenodo record (15 MB code+data zip):
python3 -c "import urllib.request,json; \
print(json.load(urllib.request.urlopen('https://zenodo.org/api/records/7810487'))['files'])"
# Download + extract the detection shapefiles, then bin to a 10 m grid and print the
# per-pixel animal-count distribution for the post-2016 scenes (GE1_20180802, WV2_20201008):
#   -> mean 1.5-2.4, median 1-2, max 18-26 animals per 10 m pixel; <2% of pixels >=10.
# (see download.download_zenodo + geopandas.read_file on
#  raw/serengeti_mara_wildebeest_zebra_detections/shapefiles/*.shp)
```
