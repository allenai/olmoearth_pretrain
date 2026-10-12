# GLiM (Global Lithological Map)

- **Slug:** `glim_global_lithological_map`
- **Status:** completed
- **Task type:** classification (per-pixel surface lithology)
- **Family / label_type:** geology / polygons
- **Samples:** 15,000 label tiles (1,000 per class × 15 classes)
- **License:** CC-BY 3.0

## Source

Hartmann, J. & Moosdorf, N. (2012), *The new global lithological map database GLiM: A
representation of rock properties at the Earth surface*, G-Cubed 13, Q12004,
[doi:10.1029/2012GC004370](https://doi.org/10.1029/2012GC004370). GLiM represents the
emerged land surface with **1,235,259 lithology polygons** compiled from 92 regional
geological maps (average source scale ~1:3,750,000). The classification is a 3-level
hierarchy; **level 1 (field `xx`) has 16 classes**.

### Access method (no credential)

The manifest URL points at the PANGAEA record
([doi:10.1594/PANGAEA.788537](https://doi.org/10.1594/PANGAEA.788537)), but that is only
the **0.5° gridded raster** (~55 km cells) — far too coarse for 10 m label tiles. The
value is the **original vector GIS database** `LiMW_GIS 2015.gdb` (an ESRI file
geodatabase, ~1.1 GB zipped), distributed under CC-BY by CCGM / Univ. Hamburg and linked
from the [GLiM project page](https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glim.html)
and [CCGM](https://www.ccgm.org/en/product/lithological-map-of-the-world/). It was
downloaded from the authors' Dropbox endpoint
(`https://www.dropbox.com/s/9vuowtebp9f1iud/LiMW_GIS%202015.gdb.zip?dl=1`) into
`raw/glim_global_lithological_map/`. The GDB has one layer `GLiM_export` in ESRI:54012
(World Eckert IV, equal-area, metres); attributes: `IDENTITY_`, `Litho` (full level-1/2/3
code), `xx` (level-1 code), `Shape_Length`, `Shape_Area`.

## Class mapping (level-1 `xx` → id)

Ids assigned in **descending global polygon frequency**. `nd` (No Data) is **dropped** (it
is not a lithology), leaving **15 classes**:

| id | code | name | id | code | name |
|----|------|------|----|------|------|
| 0 | su | unconsolidated_sediments | 8 | vi | intermediate_volcanic_rocks |
| 1 | ss | siliciclastic_sedimentary_rocks | 9 | wb | water_bodies |
| 2 | sc | carbonate_sedimentary_rocks | 10 | pb | basic_plutonic_rocks |
| 3 | sm | mixed_sedimentary_rocks | 11 | pi | intermediate_plutonic_rocks |
| 4 | pa | acid_plutonic_rocks | 12 | py | pyroclastics |
| 5 | mt | metamorphics | 13 | ev | evaporites |
| 6 | vb | basic_volcanic_rocks | 14 | ig | ice_and_glaciers |
| 7 | va | acid_volcanic_rocks | | | |

`wb` (inland water) and `ig` (ice/glaciers) are retained as legitimate, 10–30 m-observable
surface types. Per-class descriptions (GLiM legend) are in `metadata.json`.

## Processing decisions

GLiM is a **coarse, generalized derived product**, so per spec §5 (large derived product)
we sample **bounded tiles from spatially-homogeneous regions** rather than tracing polygon
boundaries precisely:

- **Polygon filter:** keep only polygons with equal-area footprint **≥ 2 km²** (728,134
  of 1.23M polygons), large enough to (nearly) fully contain a 640 m tile. Every one of
  the 15 kept classes still has ≥ 1,000 such polygons (smallest: `ig` = 1,291).
- **Tiling:** each selected polygon seeds **one 64×64 (640 m) tile** in a local UTM
  projection at 10 m/pixel, centered on the polygon's **interior representative point**
  (guaranteed inside, even for multipart/L-shaped polygons). The seed polygon is clipped
  to a small local window, reprojected, and rasterized (`all_touched=True`) with its class
  id.
- **Outside-polygon pixels → 255 (nodata/ignore), not a background class.** On a lithology
  map every land pixel is *some* rock type; we deliberately do not resolve the neighboring
  lithology at this coarse scale (positive-only foreground mask, spec §5). Downstream
  assembly supplies negatives from other datasets.
- **Homogeneity filter:** a candidate is kept only if the seed class covers **≥ 0.5** of
  the tile. In practice coverage is near 1.0 — only **3,419 / 15,000** tiles have any
  ignore border (i.e. straddle a polygon edge); the rest are uniform single-class patches.
- **Selection:** class-balanced by seed lithology (`sampling.balance_by_class`), **1,000
  tiles per class**, well under the 25,000 cap. All 15 classes reached the full 1,000.

### Time range

Lithology is a **static / time-invariant** label with no per-polygon date → a
representative Sentinel-era **1-year window** (`REP_YEAR = 2020`, i.e.
2020-01-01…2021-01-01); `change_time` is null.

## Caveats

- **Coarseness (important):** GLiM's ~1:3,750,000 source scale means lithology is only
  *partially* inferable from S2/S1/Landsat at 10–30 m, via its influence on terrain, soils
  and vegetation. Labels are homogeneous regional rock-type context, **not** sharp,
  pixel-accurate boundaries. Boundary geometry is approximate; the homogeneity filter keeps
  tiles well inside polygons to mitigate this.
- `wb`/`ig` polygons come from the source geology maps (mapped as surface units), so their
  extents reflect the mapping epoch, not a specific Sentinel-era date.
- `nd` (No Data) polygons dropped (not a lithology).

## Verification

- 15,000 `.tif` each with a matching `.json`. Sampled tiles: single-band **uint8**, 64×64,
  local **UTM at 10 m**, nodata **255**. Values across a 400-tile scan = `{0..14, 255}`
  with no out-of-range values; `metadata.json` class ids cover all observed values.
- All `time_range`s are the 1-year 2020 window; `change_time` null.
- **Georeferencing sanity check:** tile centers mapped back to lon/lat land where expected
  for the visually-verifiable classes — `ice_and_glaciers` tiles fall in the Swiss Alps
  (~8.4°E 46.5°N), NE Greenland (~−20°E 79°N), Yukon/St. Elias (~−138.6°E 60.1°N) and SE
  Tibet (~97.3°E 29.4°N); `water_bodies` tiles on NW-Russian lakes and US reservoirs. The
  `source_id` country-code prefixes (CHE, GRL, CAN_YT, CHN, …) match the coordinates,
  confirming the projection/pixel math end-to-end.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.glim_global_lithological_map
```

Idempotent: the raw GDB is skipped if already extracted; existing `locations/{id}.tif` are
skipped on re-run. Tunables: `--min_area_m2` (default 2e6), `--per_class` (1000),
`--cand_per_class` (2000), `--workers` (64).
