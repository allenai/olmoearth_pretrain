# TreeSatAI Benchmark Archive — 12762 label patches (classification)

- **Slug**: `treesatai_benchmark_archive`
- **Source**: Zenodo record 6598390 (https://doi.org/10.5281/zenodo.6598390); Ahlswede et al. 2023, ESSD.
- **Region / annotation**: Lower Saxony, Germany; state forest inventory (field reference, NLF/BI).
- **License**: CC-BY-4.0. Public, no credentials.
- **Task**: classification (tree genus), 15 classes, uint8, nodata 255.

## What it is
Multi-sensor benchmark pairing 60 m aerial + Sentinel-1 + Sentinel-2 patches with tree genus labels from the German forest inventory. We use the **Sentinel-2 200 m** patches (20x20 px, native EPSG:326xx UTM 32N at 10 m) — real GeoTIFFs with embedded CRS + geotransform, so labels drop straight onto the S2 grid.

## Labels & class scheme
`labels/TreeSatBA_v9_60m_multi_labels.json` gives each patch a multi-label list `[[genus, area_fraction], ...]` over 15 genera (14 tree genera + `Cleared`). Each 200 m patch is cut around a single inventoried stand; where one genus covers >= 0.7 of the patch we emit a **uniform single-genus tile** (spec 4 scene-level coherent land-cover patch). Class ids 0-14 assigned by descending dominant-patch frequency:

| id | genus | selected patches |
|----|-------|------------------|
| 0 | Pinus | 1000 |
| 1 | Quercus | 1000 |
| 2 | Fagus | 1000 |
| 3 | Picea | 1000 |
| 4 | Cleared | 1000 |
| 5 | Larix | 1000 |
| 6 | Pseudotsuga | 1000 |
| 7 | Acer | 1000 |
| 8 | Fraxinus | 1000 |
| 9 | Betula | 1000 |
| 10 | Alnus | 1000 |
| 11 | Abies | 901 |
| 12 | Populus | 428 |
| 13 | Prunus | 250 |
| 14 | Tilia | 183 |

## Georeferencing / tiles
Reused each S2 patch's exact CRS + geotransform (origin snapped to the integer 10 m pixel grid, sub-metre shift). Tiles are single-band uint8, 20x20, UTM 32N at 10 m; every pixel = the dominant genus's class id.

## Time range
Forest-stand genus is persistent, so each sample gets a 1-year window anchored on the inventory `YEAR` clamped to the Sentinel era (>= 2016). Inventory years span 2011-2020; pre-2016 stands are anchored at 2016. `change_time` is null (state, not change). Caveat: a few `Cleared` patches with pre-2016 inventory years may have regrown by the anchored window.

## Sampling
Class-balanced, up to 1000/class subject to the 25k cap (`balance_by_class`). Total written: 12762. Rare genera (Tilia, Prunus, Populus, Abies) contribute all available patches.

## Reproduce
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.treesatai_benchmark_archive
```
