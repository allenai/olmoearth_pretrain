# Plastic Litter Project (PLP) — plastic_litter_project_plp

- **Status**: completed
- **Task type**: classification (polygons rasterized to a mask)
- **Samples**: 16 label patches (32x32, UTM EPSG:32635, 10 m, uint8, nodata=255)
- **Source**: Zenodo record 7085112 — *Plastic Litter Project 2021 dataset* (Marine Remote Sensing Group, University of the Aegean; ESA Discovery). Open access.
- **URL**: https://zenodo.org/records/7085112
- **Access**: public Zenodo download, no credentials. Only the georeferenced orthophoto label signal is used — imagery is not retained.

## What PLP2021 is

A controlled field campaign that deployed artificial floating-plastic targets in the Gulf of Gera (Lesvos, Greece) and imaged them with Sentinel-2 on 22 dates (Jun-Oct 2021), together with UAS RGB/hyperspectral data and georeferenced UAS orthophoto maps. The targets are large HDPE-mesh rafts designed to be detectable at Sentinel-2's 10 m resolution. The archive ships one ~5-10 GB zip per date (S2 L1C + ACOLITE product + orthophoto + UAS image) and **no** ready-made vector label of the target footprints.

## Why we did not bulk-download

The full archive is ~170 GB and only the *labels* are needed (pretraining supplies its own imagery). We therefore extracted a single georeferenced UAS orthophoto (`20210716_ortho.tif`, 525 MB) from `20210716.zip` via an HTTP range-read of the deflate-compressed member + inflate (no full-zip download), segmented the two bright mesh targets against the dark water, and cached their convex-hull footprints to `raw/plp_targets.geojson`. The orthophoto is not retained.

## Labels & processing

- **Two target footprints** derived from the 20210716 orthophoto: an oval mesh target (~24 x 32 m) and a square structured-mesh target (~33 x 34 m), both offshore in open water ~270 m north of the coastline. The deployment mooring is fixed across all 2021 acquisitions, so the same footprints are reused for every date.
- **Rasterization**: footprints rasterized (`all_touched=True`) into a 32x32 UTM 10 m tile centered on the targets — class 1 = plastic target (27 pixels), class 0 = water background. 320 m tile keeps the whole context on water.
- **Class scheme**: manifest classes `[plastic target, water]` remapped so water (the natural background) = id 0 and plastic target = id 1.
- **Time range**: each sample is a specific Sentinel-2 acquisition (a transient surface object), so `time_range` is a ~1-hour window centered on the S2 acquisition instant parsed from the L1C `.SAFE` product name (well under 1 year).
- **Observability filter**: target surface state per date comes from `ancillary_data_log.pdf`. Dates where the target was *submerged* or *mostly submerged* (not detectable by Sentinel-2) were excluded (6 dates: 20210815, 20210820, 20210914, 20210919, 20210924, 20211004). Kept 16 S2-observable dates (floating / part-sub / mix-floating / mix-part-sub).

## Caveats

- **Single deployment location**: all tiles share the same geometry and footprint; diversity is temporal only (one site, one target pair, 16 dates). This is a small, high-precision controlled positive-signal dataset for marine plastic.
- Footprints were derived from one date's orthophoto and reused; per-date target configuration/exact position may vary slightly, but at 10 m the fixed footprint is a faithful approximation and the mooring is fixed.
- Older PLP campaigns (2018/2019) are separate Zenodo records and are not included here (this record is PLP2021).

## Verification

Output tifs are single-band uint8, UTM EPSG:32635 at 10 m, 32x32, values in {0 water, 1 plastic target} (nodata 255 declared but unused); each tif has a matching JSON with a ~1-hour `time_range` around the S2 acquisition instant.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.plastic_litter_project_plp
```
(Re-derives `raw/plp_targets.geojson` from the orthophoto only if it is missing.)
