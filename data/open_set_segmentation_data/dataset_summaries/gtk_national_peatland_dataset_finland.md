# GTK National Peatland Dataset (Finland)

- **Slug**: `gtk_national_peatland_dataset_finland`
- **Status**: completed
- **Task type**: classification (dense_raster)
- **Samples**: 1,444 label patches (64×64, 10 m, local UTM)
- **Classes**: 0 undrained mire, 1 forestry-drained peatland, 2 agricultural organic soil, 3 peat production area (255 = ignore)

## Source

GTK ("Geological Survey of Finland") product **"Peatland site types of Finland 1.0/2023"**
(Finnish *Suotyypit ja turvekankaat 1.0/2023*), published 2023-11-03, **GTK Open Licence**
(free reuse with attribution). A national 10 m raster covering all Finnish peatlands
(~9 Mha) where each pixel is a machine-learning-modelled peatland **site type** (40 types),
its **drainage state** (undrained/ojittamaton vs drained/ojitettu), plus **land-use** classes
(cultivated/abandoned organic-soil fields) and a companion **peat-extraction-area** product.
Modelled in the MaaTi project from Sentinel-1/-2, MML lidar-DEM derivatives and National
Forest Inventory field reference data. Native CRS EPSG:3067 (ETRS-TM35FIN), 10 m.
Catalogue: https://hakku.gtk.fi (location id 229, productOid 1.2.246.563.1.127231).

## Access method (no credential, no transaction)

The Hakku **file download** for this product is delivered only through an order/checkout
flow that submits a customer identity (POST `/api/orders` requires `items` + `customer`);
placing such an order autonomously with a fabricated identity is not appropriate, so it is
**not used**. Instead the identical raster is read anonymously through GTK's **open ArcGIS
MapServer** `Rajapinnat/GTK_Maapera_WMS` — the sanctioned machine-access interface (the
product metadata itself advertises open WMS/WFS access). Three colour-mapped raster layers
are used:

- layer **90** = undrained (site-type codes 2xxx)
- layer **89** = drained / *turvekangas* (codes 3xxx)
- layer **96** = peat extraction areas (codes 1101–1104)

We `export` PNG tiles at native 10 m (nearest; verified to render a **clean discrete colour
set with no interpolation**). The MapServer's rendered colormap does **not** match its REST
legend swatches, so the colour→code map was recovered empirically **once** via the MapServer
`identify` operation (cached at `raw/{slug}/color_decoder.json`). Layers 89 and 90 share the
rendered colormap (colour = site type; the *layer* gives drainage), so only three special
colours are needed plus the rule "any other opaque colour = a peatland site type":

- `(101,101,101)` = *Turvepelto* (cultivated organic-soil field) → class 2
- `(213,213,213)` = *Kytöheitto* (abandoned peat field) → class 2
- `(0,0,0)` = *Negatiivinen (mineraalimaa)* (modelled mineral/non-peat soil) → **255 ignore**

Raw files kept: `SOURCE.txt`, `color_decoder.json`, `peatland_site_types_fertilitylevels.pdf`
(the official code table), and cached per-block class-id GeoTIFFs under `blocks/`.

## Class mapping

| id | class | source codes |
|----|-------|--------------|
| 0 | undrained mire | layer 90 opaque, any 2xxx site type except specials (2011–2103) |
| 1 | forestry-drained peatland | layer 89 opaque, any 3xxx *turvekangas* type except specials (3011–3103) |
| 2 | agricultural organic soil | Turvepelto (2120/3120) + Kytöheitto (2130/3130) |
| 3 | peat production area | layer 96 (peat/vegetated/tree/water-covered, 1101–1104) |
| 255 | ignore | mineral soil (2140/3140) and all not-modelled/transparent pixels |

## Processing (spec §4/§5, dense_raster)

National derived-product map → **bounded-tile sampling**. A grid of 160 20 km blocks (80 km
spacing) over the peatland extent was exported once from the MapServer, combined per-pixel
into a class-id raster (EPSG:3067, nodata 255) and cached; 58 blocks contained peat. Each
kept block was scanned for non-overlapping 64×64 (640 m) windows with **≥20% peat coverage**;
a class counts toward a window if it has ≥25 px there. This produced 21,799 candidate windows
(per class: undrained 19,022 / drained 17,619 / agricultural 4,424 / peat-production 649).
Windows were selected **tiles-per-class balanced** (`select_tiles_per_class`, rarest first,
≤1000/class, 25k cap), then each was reprojected from EPSG:3067 10 m to a **local UTM**
projection (zones 34/35/36 N) at 10 m with **nearest** resampling. Output tiles keep the
**true class of every pixel** (full multi-class segmentation), not a single dominant class.

**Selected tiles per class** (a tile counts toward every class it contains):
undrained mire 1,055 · forestry-drained peatland 1,414 · agricultural organic soil 1,000 ·
peat production area **649** (all available — the rarest class; kept per §5). Total 1,444 tiles.

## Time / change

Quasi-static land classification (product v1.0/2023, trained on 2016–2023 EO). Static **1-year
window on 2023**, `change_time = null`. No pre-2016 labels.

## Verification

- 1,444 `.tif` + 1,444 matching `.json`; all single-band uint8, 64×64, local UTM at 10 m,
  nodata 255; pixel values ⊆ {0,1,2,3,255}, all covered by `metadata.json`; time_range = 2023.
- **Label-correctness cross-check vs the authoritative source**: random output pixels compared
  to MapServer `identify` at the same coordinates agreed on all four classes and background
  (10/12 exact at single pixels; the 2 differences were 1-px class-boundary reprojection slop).
  A full-tile check of a peat-production tile (000118) against a fresh layer-96 render over its
  exact UTM footprint agreed 2,060 vs 2,067 class-3 px (99.7%), confirming the colour-decode +
  combine + reproject chain.

## Caveats

- The product is a **modelled** map ("not applicable to small-scale detailed examinations")
  and is a derived-product fallback (no in-situ reference alternative in the manifest).
- Site-type detail is collapsed to the manifest's 4 land-use/drainage classes; the full 40
  site types and fertility levels are available in `peatland_site_types_fertilitylevels.pdf`
  if a finer scheme is ever wanted.
- Modelled mineral soil (Negatiivinen) is treated as ignore (255), not a class.
- Peat production area (649 samples) is the sparsest class; kept in full per §5 (downstream
  assembly drops too-small classes if needed).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gtk_national_peatland_dataset_finland
```
Idempotent: cached block GeoTIFFs under `raw/{slug}/blocks/` and existing
`locations/{id}.tif` are skipped on re-run.
