# US National Wetlands Inventory (NWI)

- **Slug:** `us_national_wetlands_inventory_nwi`
- **Status:** completed
- **Task type:** classification (positive-only multi-class segmentation, `label_type: polygons`)
- **Num samples:** 3,168 label tiles (64x64, single-band uint8, local UTM @ 10 m)
- **Source:** US Fish & Wildlife Service, National Wetlands Inventory. Public domain.
- **Landing:** https://www.fws.gov/program/national-wetlands-inventory/data-download

## Source & access

NWI is the authoritative national wetland polygon layer for the US, produced by
photointerpretation and classified with the full Cowardin hierarchy. Data are distributed
as **per-state File Geodatabase downloads** (no credentials required) at:

```
https://documentst.ecosphere.fws.gov/wetlands/data/State-Downloads/{ST}_geodatabase_wetlands.zip
```

Each state GDB has a `{ST}_Wetlands` polygon layer (EPSG:5070 CONUS Albers) with fields
`ATTRIBUTE` (raw Cowardin code, e.g. `PEM1C`, `E2EM1P`, `R2UBH`) and `WETLAND_TYPE` (NWI's
own simplified legend derived from the Cowardin code).

## Bounded sampling (spec §5)

The national layer is enormous (hundreds of millions of polygons across all states), so we
did **not** pull all of CONUS. We downloaded a **bounded, diverse set of 4 states** chosen
to cover every Cowardin system across distinct biogeographic settings, and sampled tiles
from them:

| State | Polygons | Why |
|-------|---------:|-----|
| LA Louisiana | 657,311 | Gulf coast: Marine, Estuarine, Riverine, Lacustrine, Palustrine |
| FL Florida | 1,061,514 | Everglades + coasts: Marine, Estuarine, freshwater forested/emergent |
| ND North Dakota | 2,075,122 | Prairie Pothole Region: freshwater emergent, ponds, lakes |
| NC North Carolina | 589,941 | Atlantic coastal plain: estuarine, riverine, forested swamps |

Raw GDBs live at `raw/us_national_wetlands_inventory_nwi/` on weka (~3.3 GB zipped).

## Class scheme (code → class)

We use NWI's `WETLAND_TYPE` simplified legend as a manageable, semantically-clean scheme
(8 classes, ordered by frequency across the sampled states). Each `WETLAND_TYPE` string is
derived by NWI from the leading Cowardin system/class letters:

| id | WETLAND_TYPE | Cowardin origin |
|----|--------------|-----------------|
| 0 | Freshwater Emergent Wetland | Palustrine emergent (PEM) |
| 1 | Freshwater Forested/Shrub Wetland | Palustrine forested / scrub-shrub (PFO/PSS) |
| 2 | Riverine | Riverine system R1–R5 |
| 3 | Freshwater Pond | Palustrine open water (PUB/PAB/PUS ponds) |
| 4 | Other | NWI freshwater catch-all (farmed / misc. palustrine); common in ND potholes |
| 5 | Estuarine and Marine Wetland | Intertidal E2/M2 (salt marsh, flats, mangrove, reef) |
| 6 | Estuarine and Marine Deepwater | Subtidal E1/M1 (bays, sounds, nearshore ocean) |
| 7 | Lake | Lacustrine L1/L2 |

The raw Cowardin `ATTRIBUTE` code is available in the source GDB for anyone wanting a finer
scheme, but a per-pixel model at 10 m cannot resolve most Cowardin subclass/modifier
distinctions, so the 8-class `WETLAND_TYPE` legend is used.

**Positive-only:** NWI maps only wetland/deepwater features. Pixels outside every polygon
are left as **nodata/ignore (255)**, NOT a fabricated "upland" background — the pretraining
assembly step supplies negatives from other datasets (spec §5). We deliberately do not
assert upland everywhere unmapped, since NWI's minimum mapping unit omits small features.

## Tiling & sampling

- 64x64 windows (640 m), local UTM at 10 m/pixel, north-up.
- Candidate windows are **seeded from polygons of every class** (up to 1,500 seeds/class/
  state, so rare classes — Lake, estuarine/marine — get coverage even where scarce),
  snapped to a 64-px UTM grid so nearby seeds deduplicate into shared tiles. Seed placement
  is fully vectorized (batch pyproj reprojection of polygon centroids 5070→lon/lat→UTM).
- For each window, **every NWI polygon intersecting it is rasterized in** (value =
  `WETLAND_TYPE` class id; outside-polygon = 255) via `rasterio.features.rasterize`,
  yielding dense multi-class tiles.
- Tiles selected **tiles-per-class balanced** (`sampling.select_tiles_per_class`, rarest
  class first, up to 1,000 tiles/class, 25k total cap).

Because dense wetland tiles typically contain several classes at once, filling the rare
classes to their target also satisfies the common classes, so only **3,168 unique tiles**
were needed (well under the 25k cap). Resulting balance is even:

```
class tile counts: {0:1652, 1:1023, 2:961, 3:1134, 4:1066, 5:1048, 6:1006, 7:1067}
state tile counts: {FL:626, LA:812, NC:555, ND:1175}
```

## Time range

Wetlands are persistent/static features, so per spec §5 each sample gets a representative
1-year Sentinel-era window: **2020-01-01 → 2021-01-01** (`io.year_range(2020)`),
`change_time = null`. (This uses the repo's shared `year_range` convention, as other static
datasets do.)

## Verification

- Sampled tiles confirmed: single-band uint8, UTM CRS at 10 m, ≤64×64, values ∈ {0–7, 255},
  nodata=255; every `.tif` has a matching `.json` with a ≤1-year `time_range` and
  `change_time=null`; metadata class ids cover all tif values (0–7).
- Geolocation sanity check (tile centers → lon/lat): estuarine/marine tiles land on the
  LA (−90.05, 29.38 — Barataria/Mississippi delta) and FL (−82.44, 27.76 — Tampa Bay)
  Gulf coasts; freshwater emergent/pond tiles in central ND (−99.04, 46.23 — prairie
  potholes); forested/riverine/lake in coastal-plain NC (−76.50, 35.80 — Pamlico). All
  consistent with expectations. A full Sentinel-2 pixel overlay was not run, but
  georeferencing is exact by construction (NWI polygons reprojected 5070→UTM via rslearn;
  reprojection round-trip validated).

## Caveats

- Only 4 states sampled (bounded, spec §5) — not nationally representative of every US
  wetland region (e.g. no Alaska tidal/tundra, no arid Southwest playas, no Great Lakes /
  Pacific NW). The 8-class legend is national, but class *appearance* is drawn from these 4
  states' imagery footprints.
- Class 4 "Other" is a vague NWI catch-all (largely farmed/misc. palustrine, heavy in the
  ND Prairie Pothole Region); kept because it is a real mapped legend category.
- Cowardin subclass/modifier detail is collapsed to the 8-class legend (not resolvable at
  10 m).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.us_national_wetlands_inventory_nwi --workers 64
```

Idempotent: existing `locations/{id}.tif` are skipped. To change states, edit `STATES`.
