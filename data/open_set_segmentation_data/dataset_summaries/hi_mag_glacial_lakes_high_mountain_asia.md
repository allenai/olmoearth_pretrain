# Hi-MAG Glacial Lakes (High Mountain Asia)

- **Slug:** `hi_mag_glacial_lakes_high_mountain_asia`
- **Status:** completed
- **Task type:** classification (binary dense segmentation / water extent)
- **Family / region:** glacier / High Mountain Asia
- **Source:** Chen, F. et al. *"Annual 30 m dataset for glacial lakes in High Mountain Asia
  from 2008 to 2017"*, Earth System Science Data (ESSD), 2021.
  Zenodo record **4275164**, DOI `10.5281/zenodo.4275164`. License **CC-BY-4.0**.
- **Access:** public Zenodo download, no credentials. `Hi-MAG database.zip` (58.7 MB) +
  `Metadata for Hi-MAG database.docx`. Verified against the Zenodo MD5
  (`abd64cb0a4a3584954e6ba1c04465c94`). No imagery pulled — pretraining supplies its own.
- **Num samples:** **14,827** label tiles (well under the 25k per-dataset cap).

## Source data

Annual glacial-lake **polygon** shapefiles, one per year 2008–2017
(`Hi_MAG_database_YYYY.shp`), in **Asia North Albers Equal Area Conic** (ESRI:102025,
metres). Per-lake attributes: `GL_Type` (proglacial / supraglacial / unconnected-glacial /
ice-marginal), `GL_Area` (m²), `GL_Elev`, `GL_SubR` (HMA sub-region), `GL_Peri`, `GL_ID`
(lon/lat code), `Distance` (to nearest glacier). Lakes were semi-automatically delineated on
~30 m Landsat and manually refined by experts.

**Year used: 2017** — the most recent Hi-MAG year and post-2016 (Sentinel era). The 2017
layer has **15,348 lakes**: proglacial 7,923, unconnected-glacial 7,136, supraglacial 216,
ice-marginal 73.

**Observability / size filter:** the source already applies a minimum mapped-lake area. In
2017 the smallest lake is **~0.0081 km² (~81 pixels at 10 m; ~9×9 px)**, median ~0.029 km²,
so every lake is comfortably observable at 10 m. No additional size filter was applied.

## Encoding

Binary **water-extent** dense segmentation (polygons rasterized to a UTM 10 m grid):

| id | name | meaning |
|----|------|---------|
| 0 | `background` | surrounding HMA terrain within the tile (glacier ice, moraine/debris, rock, snow, vegetated valley floor). Genuine, spatially-meaningful negatives around the lake — not fabricated. |
| 1 | `glacial_lake` | glacial-lake water surface (any Hi-MAG lake type). |

The four Hi-MAG **lake types are collapsed into one water class**: the type is a
positional/connectivity attribute (distance/contact to the parent glacier), not spectrally
separable from a single S2/S1/Landsat tile, so binary water extent is the well-posed target.
The per-lake type is retained in `metadata.json` (`source_lake_type_counts`) for provenance.
`nodata = 255` (unused here; every pixel is either background or lake).

## Tiling & sampling

Mirrors the `blue_ice_areas_of_antarctica_tollenaar_et_al` recipe. Each lake centroid is
projected to its **local UTM zone at 10 m** and snapped to a **64-px (640 m) grid**; the
unique grid cells become candidate tiles, and **every** lake polygon intersecting a tile is
rasterized into it (`rasterio.features.rasterize`, `all_touched=False`, fill=background).
One tile per unique grid cell → 14,829 candidates → 14,827 with lake pixels (2 empty cells
dropped, e.g. lakes that fell just outside their snapped cell). All kept (< 25k cap).

Lakes are small relative to a 640 m tile, so tiles range from lake-sliver (surrounded by
real terrain) to lake-interior:
- `lake_frac`: min 0.006, **median 0.066**, max 1.000.
- Class tile-appearance: background in 14,797 tiles, glacial_lake in all 14,827 (30 tiles
  are 100% lake — interiors of large lakes — hence no background there).

Tiles: **64×64 px, single-band uint8, local UTM at 10 m/pixel**, north-up.

## Time range & change handling

Glacial lakes are **persistent surface-water bodies**, so this is a **static** label:
a representative 1-year Sentinel-era window **2017** (`[2017-01-01, 2018-01-01)`),
`change_time = null`.

The optional multi-year **change/expansion** variant was **deliberately not used**: Hi-MAG
snapshots are annual, so lake growth is only resolvable to ~a year — too coarse for the
spec's <=1–2-month change-timing requirement (§5). The presence/extent state is genuinely
persistent, so the static classification encoding is the correct fit.

## Verification (spec §9)

- Sampled tiles: single-band **uint8**, **64×64**, **UTM 10 m** (e.g. EPSG:32642/32643/
  32646/32647), `nodata=255`, values ⊆ {0,1}. All sampled tiles conform.
- Every `.tif` has a matching `.json` with a 1-year `time_range` and `change_time=null`.
  `metadata.json` class ids {0,1} cover all raster values.
- Centroids land in High Mountain Asia (lon ~71–97°E, lat ~27–42°N: Pamir, Karakoram,
  Himalaya, Nyainqêntanglha, Hengduan, Tien Shan).
- **Georeferencing cross-check:** for 8 random tiles, the center of a rasterized
  `glacial_lake` pixel was reprojected back to the source Albers CRS and confirmed inside a
  2017 Hi-MAG lake polygon — **8/8** pass, confirming the pixel-bounds/CRS pipeline is exact.
  (A full Sentinel-2 image overlay was not rendered; georeferencing is exact by construction
  via rslearn's `GeotiffRasterFormat` and validated by the polygon cross-check.)

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.hi_mag_glacial_lakes_high_mountain_asia
```

Idempotent: re-running skips already-written `locations/{id}.tif`.

## Caveats

- Supraglacial (216) and ice-marginal (73) lakes are rare, but types are collapsed to a
  single water class, so class balance is not an issue for the binary target.
- The source is a derived (semi-automated + manually refined) product; polygon interiors are
  treated as high-confidence water (validated in the ESSD publication).
- Labels are lake-water extent only; the surrounding `background` is real within-tile HMA
  terrain, not fabricated negatives.
