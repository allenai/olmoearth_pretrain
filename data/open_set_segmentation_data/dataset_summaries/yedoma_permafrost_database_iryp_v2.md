# Yedoma Permafrost Database (IRYP v2)

- **Slug:** `yedoma_permafrost_database_iryp_v2`
- **Status:** completed
- **Task type:** classification (polygon → dense region classification)
- **Samples:** 3,992 (2,992 Yedoma-positive tiles + 1,000 background-only negatives)

## Source

Strauss, J. et al. (2022): *Database of Ice-Rich Yedoma Permafrost Version 2 (IRYP v2).*
PANGAEA, https://doi.org/10.1594/PANGAEA.940078 (CC-BY-4.0). Supplement to Strauss et al.
(2021), *Circum-Arctic Map of the Yedoma Permafrost Domain*, Frontiers in Earth Science 9,
https://doi.org/10.3389/feart.2021.758360.

Yedoma is a late-Pleistocene ice-rich (syngenetic) permafrost deposit. IRYP v2 harmonizes
and digitizes geological/stratigraphic source maps plus field/expert knowledge into a
pan-Arctic vector map of Yedoma occurrence with three presence-confidence levels.

## Access method

The manifest DOI record (PANGAEA.940078) is delivered as a tab-separated *point/site* table
(field exposures, photo sites, map-source metadata) — NOT the polygons. The actual vector
layers are bundled as per-layer shapefile ZIPs downloadable openly (no account) from
`https://download.pangaea.de/dataset/940078/files/`:

- `IRYP_v2_yedoma_confidence_Shapefile.zip` — **USED**: 13,833 Yedoma-deposit polygons in
  EPSG:3571 (WGS84 North Pole LAEA Bering Sea, metres) with a `confidence` attribute
  (confirmed / likely / uncertain). ~569,000 km² total.
- `IRYP_v2_yedoma_domain_Shapefile.zip` — downloaded but NOT used as the label. This is the
  20 km-buffered *domain envelope* (~2.59 M km², no confidence attribute); it is a derived
  buffer around Yedoma, much coarser than the deposit extent, so the confidence (deposit)
  layer is the better per-pixel label.

Note: the AWI/APGC portal (`apgc.awi.de`, `maps.awi.de` WMS/WFS) that also serves these
layers was returning 502/503/timeouts during processing; the PANGAEA `download.pangaea.de`
mirror is the stable source used here.

Reproduce:
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.yedoma_permafrost_database_iryp_v2
```
(idempotent — skips already-written `locations/{id}.tif`). Downloads + extracts the
shapefiles to `raw/{slug}/` automatically.

## Class / label mapping

Single-band uint8 tiles, class ids from the `confidence` attribute:

| id | name | source | tiles containing class |
|----|------|--------|------------------------|
| 0 | background | non-Yedoma terrain inside a tile | 2,229 |
| 1 | yedoma_confirmed | confidence=confirmed (conf_id 11–14) | 1,000 |
| 2 | yedoma_likely | confidence=likely (conf_id 21–22) | 1,000 |
| 3 | yedoma_uncertain | confidence=uncertain (conf_id 31) | 1,002 |
| 255 | nodata/ignore | (declared, unused) | — |

Source polygon areas: confirmed 485,018 km²; uncertain 61,841 km²; likely 21,873 km².

**Confidence classes are mapping-certainty tiers of the SAME phenomenon** (Yedoma presence),
not visually distinct land classes. A model cannot distinguish "confirmed" from "uncertain"
Yedoma from imagery — the difference is source-map certainty. They are kept separate per
spec §5 (do not drop classes; downstream can filter/merge). **A consumer training binary
Yedoma presence should merge classes 1–3.**

## Processing

- Polygon → dense region classification. 64×64 tiles at 10 m/pixel (640 m) in the local UTM
  zone. Yedoma polygons intersecting a tile are rasterized (`all_touched=True`) at their
  confidence class value; the rest of the tile is background (0).
- **Positive sampling:** area-weighted random interior points **per confidence class** (so
  rare classes reach quota), deduplicated on a ~1 km grid; 6,000 candidates generated, then
  `balance_tiles_by_class` (per_class=1000, cap=25,000) selected 2,992.
- **Negatives:** 1,000 background-only tiles drawn 20–150 km from Yedoma and verified
  Yedoma-free (the polygon boundary is a genuine land/region delineation, so background is a
  real negative here, cf. peatmap — not a fabricated negative).
- Bounded, regionally-diverse sample of a large circum-Arctic derived product (not
  exhaustive coverage). Samples span 60–77°N; Siberia/Eurasia ~85%, Alaska/Yukon ~15%.

## Time / change handling

Static geomorphic region (like a lithology map) → `change_time = null`, 1-year window
anchored on **2020** (representative Sentinel-era year, within the manifest's 2016–2021
validity). Not a dated change event.

## Verification

- Sampled GeoTIFFs: single-band, uint8, 64×64, 10 m, north UTM zones (EPSG:326xx), values ∈
  {0,1,2,3}; every `.tif` has a matching `.json` with a 1-year `time_range` and
  `change_time=null`; metadata class ids cover all values in the tifs.
- Spatial correctness (vs. authoritative source polygons, EPSG:3571): 300/300 sampled
  positive tiles contain Yedoma within the tile; 300/300 sampled negatives are genuinely
  Yedoma-free. (Direct polygon-containment check in lieu of S2 overlay, since the label is
  rasterized straight from the source vector map.)

## Caveats

- **Coarse label.** Yedoma is a *subsurface* deposit; its extent was compiled/harmonized
  from geological maps at heterogeneous (often coarse) scales. Boundaries are coarse relative
  to 10 m → expect boundary label noise, and background pixels just outside a polygon may in
  reality be Yedoma. Surface expression (thermokarst uplands, ice-wedge-polygon texture,
  thermocirques) is only *partly* resolvable at 10–30 m from S2/S1/Landsat. This is a valid
  but coarse per-pixel region class — usable, but downstream should weight/interpret
  accordingly and consider merging the confidence tiers into binary presence.
- The 20 km-buffered `yedoma_domain` layer (2.59 M km²) was deliberately not used; the
  deposit `confidence` layer is the meaningful extent label.
