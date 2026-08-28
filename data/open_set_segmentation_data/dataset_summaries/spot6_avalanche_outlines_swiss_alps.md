# SPOT6 Avalanche Outlines (Swiss Alps) — `spot6_avalanche_outlines_swiss_alps`

**Status:** completed · **task_type:** classification (single-class avalanche-presence
segmentation, change/event label) · **num_samples:** 24,647

## Source

EnviDat / WSL Institute for Snow and Avalanche Research **SLF**, *"SPOT6 Avalanche
outlines 24 January 2018"* (Hafner, E. & Bühler, Y., 2019), **doi:10.16904/envidat.77**.
18,737 avalanche outlines were **manually mapped** (photointerpretation) from a single
SPOT6 satellite acquisition on **24 January 2018**, documenting an extreme avalanche
period (avalanche danger level 5) over the Swiss Alps (Bühler et al. 2019, *The
Cryosphere* 13, 3225–3238).

- Landing page / DOI: `https://doi.org/10.16904/envidat.77`
- **Access used (no credentials):** direct HTTP download of the EnviDat resource zip
  `aval_outlines2018.zip` (157 MB) →
  `raw/spot6_avalanche_outlines_swiss_alps/extracted/outlines2018.shp` (+ sidecars +
  `ExampleKey_AvalMapping.pdf` attribute key). The download URL is recorded in the script
  and in `raw/.../SOURCE.txt`.
- **License:** Open Database License (**ODbL**) with Database Contents License (**DbCL**).
  Free to use with attribution. **Attribution:** *Data: WSL Institute for Snow and
  Avalanche Research SLF / EnviDat (Hafner & Bühler 2019), ODbL/DbCL.* (recorded in
  `metadata.json` → `provenance.attribution`).

Each shapefile feature is one avalanche-outline **polygon** (source CRS **EPSG:2056**,
CH1903+/LV95) with per-avalanche attributes: `typ` (SLAB / LOOSE_SNOW / FULL_DEPTH /
UNKNOWN), `aval_shape` (outline quality: 1=exact, 2=estimated, 3=created), `sze` (size
class), `aspect`, `start_zone`/`dpo_alt` (altitudes). Polygon area: min 121 m², median
~21,000 m² (~210 px at 10 m), p95 ~195,000 m², max ~1.95 M m². Only **0.24 %** of
outlines are < 400 m² (< 4 px), so essentially all avalanches are resolvable at 10 m.

## Label design

**Single-class avalanche-presence segmentation** (uint8):
- `0`   = avalanche (interior of a mapped avalanche outline — full extent: release +
  track + deposit)
- `255` = nodata / ignore (everything else)

This is a **positive-only foreground** dataset: outlines were mapped only where
avalanches occurred, and the absence of an outline is not a verified "no avalanche". Per
spec §5 we therefore do **not** fabricate negatives — non-avalanche pixels are nodata,
and the downstream assembly step supplies negatives from other datasets.

Per-avalanche attributes (`typ`, `aval_shape`, `sze`, aspect, altitudes) are **not
observable per-pixel** from 10–30 m S2/S1/Landsat imagery, so they are kept as
**provenance metadata only** (`provenance.typ_codes`, `provenance.aval_shape_codes`),
not as label classes. Source attribute distribution (all 18,737): typ = {SLAB 13,492,
UNKNOWN 2,622, FULL_DEPTH 2,011, LOOSE_SNOW 612}; outline quality = {2/estimated 10,871,
1/exact 6,117, 3/created 1,749}.

## Change semantics (this is a change/event dataset)

All avalanches released during the 22–24 January 2018 storm cycle and were mapped from
the **24 Jan 2018** SPOT6 image — the event date is known to within days (well inside the
§5 ~1–2 month precision requirement). Each sample carries `change_time = 2018-01-24`,
retained as the reference date used to build two adjacent six-month windows via
`io.pre_post_time_ranges(change_time, ...)`: `pre_time_range` — the ~6 months (≤183 days)
immediately before `change_time` — and `post_time_range` — the ~6 months (≤183 days)
immediately after. The two windows are adjacent, split exactly at `change_time` (total
span still ~1 year), and `time_range` is set to null. Pretraining pairs a "before" image
stack with an "after" stack and probes on their difference, so it always straddles the
late-Jan-2018 debris period (undisturbed snowpack before → avalanche-debris texture
after). `metadata.json` sets `is_change_dataset: true`.

**Why change, not static presence:** avalanche debris is snow — visible for weeks after
the event but gone by the following summer. A static full-year presence label anchored on
2018 would be misleading (summer-2018 imagery shows no debris), so the change framing
(splitting the before/after windows at the event) is the faithful representation.

## Tiling & sampling

- Each outline reprojected EPSG:2056 → WGS84 → **local UTM at 10 m/pixel** (Swiss Alps
  fall in EPSG:32632, UTM zone 32N).
- Small avalanche (footprint ≤ 64×64 px = 640 m): **one centered 64×64 tile**.
- Large avalanche: gridded into **non-overlapping 64×64 windows**; windows intersecting
  the outline are kept, up to **`MAX_TILES_PER_AVAL = 20`** sampled per avalanche.
- Inside outline → 0, everything else → 255 (`rasterize_shapes`, `all_touched=True` so
  thin avalanche tracks stay visible at 10 m).
- **Selection:** round-robin across avalanches (every avalanche contributes ≥1 tile
  before large ones add more), capped at 25,000 (`sampling.MAX_SAMPLES_PER_DATASET`).
  Candidate pool = 24,647 across all 18,737 avalanches → all 24,647 selected (under cap;
  every one of the 18,737 avalanches is represented, large ones contributing extra tiles).

**Counts:** 24,647 tiles, all containing class 0 (avalanche). Positive-only, so nodata
dominates each tile (avalanche footprint is small relative to a 640 m tile; sampled tiles
run ~96–99 % nodata).

## Verification (§9)

- 5 random tifs: single-band `(1,64,64)`, uint8, UTM 10 m (EPSG:32632), values ⊆ {0,255}. ✓
- Every `.tif` has a matching `.json` (24,647 / 24,647; 0 missing). ✓
- `time_range` = null with an adjacent `pre_time_range`/`post_time_range` pair (each
  ≤183 days) split at `change_time` for all sampled; `change_time` = 2018-01-24 for all;
  `classes_present` = [0] for all. ✓
- `metadata.json`: `task_type=classification`, `num_samples=24647`, `nodata_value=255`,
  classes = [(0, avalanche)], `is_change_dataset=true`, `change_time=2018-01-24`. ✓
- Geographic sanity: 300 random tile centroids all fall inside the Swiss Alps box
  (lon 6.81–10.47, lat 45.89–47.15; 0 outliers). ✓
- Full Sentinel-2 overlay not performed (S2 fetch is heavy/out-of-band); georeferencing is
  exact because tiles are written via `GeotiffRasterFormat` in the same UTM projection the
  outline was rasterized in.

## Judgment calls / caveats

- **Single class** kept as specified; per-avalanche type/quality/size are not per-pixel
  observable → metadata only.
- **Change vs static presence:** chose the dated-change framing (change_time = 24 Jan
  2018) because debris does not persist a full year; documented above.
- **Snow-on-snow visibility:** avalanche debris is a texture/albedo change on snow; large
  debris tongues are clear at 10 m but the faintest small releases mapped at VHR SPOT6
  resolution may be marginal at 10 m S2. The 0.24 % of sub-400 m² outlines are the
  weakest; kept (per §5, downstream filtering handles rare/marginal cases).
- Positive-only tiles are mostly nodata by construction; this is intended (assembly adds
  cross-dataset negatives).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.spot6_avalanche_outlines_swiss_alps
```
Idempotent: existing `locations/{id}.tif` are skipped. Raw shapefile is cached at
`raw/spot6_avalanche_outlines_swiss_alps/extracted/outlines2018.shp`.
