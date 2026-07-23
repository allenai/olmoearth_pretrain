# EuroMineNet (Sentinel-2 Mining/Quarry Benchmark)

- **Slug**: `eurominenet_sentinel_2_mining_quarry_benchmark`
- **Status**: completed
- **Task type**: classification (binary dense segmentation)
- **Family / region**: mining / EU (14 countries)
- **Samples**: 1114 label tiles (64×64, 10 m UTM), spanning all 133 sites and years 2016–2024.

## Source

- Paper: Yu et al. (2026), "EuroMineNet: A multitemporal Sentinel-2 benchmark for
  spatiotemporal mining footprint analysis in the European Union (2015–2024)", ISPRS J.
  Photogramm. Remote Sens. 237:409–425. DOI
  [10.1016/j.isprsjprs.2026.04.046](https://doi.org/10.1016/j.isprsjprs.2026.04.046)
  (arXiv:2510.14661).
- Data: RODARE [record 4656](https://rodare.hzdr.de/record/4656), DOI
  [10.14278/rodare.4656](https://doi.org/10.14278/rodare.4656). Single 18.4 GB
  `EuroMineNet.zip`. **License: CC-BY-4.0** (no credential/registration required; public).
- Code: https://github.com/EricYu97/EuroMineNet

Archive layout (per site): `EuroMineNet/{Site}/image/Year{YYYY}.tif` (10-band Sentinel-2,
int16, EPSG:4326, ~10 m) and `EuroMineNet/{Site}/label/Year{YYYY}.tif` (single-band uint8
mask). 133 mining sites (Austria, Bulgaria, Czechia, Finland, Germany, Greece, Hungary,
Italy, Poland, Portugal, Romania, Slovakia, Spain, Sweden), annual 2015–2024.

## Triage & georeferencing (SOP §8)

EuroMineNet is a **georeferenced GeoTIFF** benchmark → accepted. The critical check: the
**label** tifs ship with an identity transform (no CRS), but each label shares the exact
pixel grid (same width/height) of its sibling **image** tif, which *is* georeferenced
(EPSG:4326 + affine transform). Verified across sites/years that per site the image
CRS/transform/size are **constant across all years** and equal the label size, so label
georeferencing is fully recoverable from the image header. (Not a "no-recoverable-coords"
rejection.)

## Label semantics — binary, not the manifest's 5 classes

The manifest lists `["large quarry","non-metallic mining","active extraction","waste
deposits","tailings ponds"]`. These are **not** per-pixel classes: the paper's per-pixel
annotation is a **single binary mining footprint** ("classify each pixel as either mine or
non-mine", §3.4), and the mine-type breakdown (50 metallic / 56 coal / 8 non-metallic / 19
large-quarry) is a **site-level** attribute. No site→type table ships in the archive, so we
use the honest per-pixel scheme:

| id | name | source value |
|----|------|--------------|
| 0 | background (non-mine) | 0 |
| 1 | mining footprint | 255 |
| 255 | nodata (tile pixels outside a source site) | — |

## Download — thin label extraction (no bulk pull)

Downloading 18.4 GB is unnecessary (pretraining supplies its own imagery; the images are
the bulk). Using new shared HTTP-range helpers (`download.remote_zip_index` /
`extract_remote_zip_member`, which parse the remote zip's Zip64 central directory and
range-fetch + inflate individual members), we pulled only: (a) each site's image-tif
**header** (first 64 KB → CRS + transform + size) and (b) the small label tifs
(2016–2024). **~50 MB total.** Recovered georeferencing is baked into
`raw/{slug}/labels/{Site}/Year{YYYY}.tif` (georeferenced binary masks, 0/255) plus
`raw/{slug}/sites_georef.json`.

## Processing (SOP §4 dense_raster, §5)

- **2015 dropped**: its 1-year window largely predates usable Sentinel-2 (mission ramp-up
  mid/late-2015). Kept 2016–2024 (9 years). All other years processed.
- **Scan**: each raw label scanned in its native EPSG:4326 grid in 64 px (~640 m) blocks;
  record class ids present + block-center lon/lat (cap ~30 tiles/class per site-year to
  bound candidates → 52,122 candidates; 48,441 contain background, 30,508 contain mining).
- **Balance**: tiles-per-class (rarest first), ≤1000/class, 25k cap → **1114 tiles**
  (1000 contain background, 1058 contain mining; a tile counts for every class it holds).
- **Write**: each selected tile reprojected to a local UTM 64×64 patch at 10 m with
  **nearest** resampling (categorical); pixels outside the source site → 255 (nodata).
- **Time**: annual state maps → 1-year window anchored on the tile's year;
  `change_time = null`. This uses EuroMineNet's *footprint-mapping* task (per-year presence),
  **not** its change-detection task, so no ≤1–2-month change-timing constraint applies.

Selected-tile distribution by year: 2016:117, 2017:139, 2018:132, 2019:115, 2020:125,
2021:125, 2022:129, 2023:129, 2024:103. Spans all 133 distinct sites.

## Verification (SOP §9)

- 1114 `.tif` + 1114 matching `.json`; all single-band uint8, 64×64, projected UTM at 10 m.
- Union of pixel values across **all** tiles = `{0, 1, 255}` (exactly the declared class ids
  + nodata); `metadata.json` class ids cover them.
- Every sample JSON has a ≤1-year `time_range` and `change_time = null`.
- Sampled UTM zones (32629/32632/32634 …) correctly match the sites' EU countries.
- **Spatial/spectral sanity**: within the source Sentinel-2 imagery, mean NDVI of mining
  pixels ≪ background at every checked site (Romania_1 0.32 vs 0.74; Spain_1 0.18 vs 0.61;
  Bulgaria_6 0.20 vs 0.29) — the mask sits on bare/disturbed ground, confirming correct
  label↔imagery alignment. (Alignment is also guaranteed by construction: the label shares
  the S2 image pixel grid the georef was recovered from.)
- Idempotent: re-running skips existing raw labels and output tiles and re-selects the
  identical 1114 tiles (deterministic seeds).

## Caveats

- Binary task only; mine-type sub-classes are not per-pixel recoverable from the release.
- Native pixels are ~10 m in latitude but ~7 m in longitude (EPSG:4326 at ~45°N); scan
  composition uses native 64-px blocks (approximate footprint) while the written tiles use
  exact UTM 10 m reprojection. Balancing is unaffected; per-sample `classes_present` is
  recomputed from the final tile.
- Edge tiles near a site boundary contain some 255 (nodata) where the 640 m tile extends
  past the source site — expected and treated as ignore.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.eurominenet_sentinel_2_mining_quarry_benchmark --workers 64
```

Outputs on weka: `datasets/eurominenet_sentinel_2_mining_quarry_benchmark/`
(`metadata.json`, `locations/{000000..001113}.tif`+`.json`); raw at
`raw/eurominenet_sentinel_2_mining_quarry_benchmark/labels/{Site}/Year{YYYY}.tif` +
`sites_georef.json`.
