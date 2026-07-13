# GABAM (Global Annual Burned Area Map)

- **Slug**: `gabam_global_annual_burned_area_map`
- **Status**: completed
- **Task type**: classification (per-pixel, 2 classes)
- **Num samples**: 1145 label patches (64x64, 10 m UTM GeoTIFFs)
- **Source**: Zenodo record [13858799](https://zenodo.org/records/13858799) —
  "Updated 30 m resolution global annual burned area map, 2014-2024" (GABAM; Long et al.,
  Aerospace Information Research Institute, CAS). License **CC-BY**.

## Source description

GABAM is a Landsat-derived global product that maps, for each calendar year, whether each
30 m pixel burned at least once during that year. Each year is distributed as one ZIP of
~1000 GeoTIFF tiles, one per 5x5-degree cell in **EPSG:4326** at 0.00025 deg (~30 m).
Per-pixel value:

- `0` = not burned
- `1` = burned (burned at least once during the year)

The Zenodo record's file set covers years **2014-2021** (the title says 2014-2024 but only
2014-2021 ZIPs are published). We use only **post-2016 (Sentinel-era)** years.

## Class mapping

Matches the manifest's two classes exactly:

| id | name       | source value |
|----|------------|--------------|
| 0  | not burned | 0 (background/negative) |
| 1  | burned     | 1            |

nodata / ignore = 255 (used only for out-of-source fill; in practice all sampled windows
are fully covered by land tiles, so final patches contain only {0, 1}).

## Why STATIC PRESENCE, not a dated change label

GABAM resolves a burn only to the **calendar year** (the pixel burned *sometime* during
that year), NOT to within ~1-2 months. Per the task spec (§5), a dated **change** label
requires placing the event confidently inside the pretraining pairing window; at year
resolution we cannot, so GABAM is **not** usable as a change label. Instead we apply the
spec's *persistent-post-change-state* exception: a post-fire burn scar (charring,
vegetation loss) stays visible for many months after the fire, so "burned vs not-burned"
is a legitimate **static presence classification**. Therefore:

- `change_time = null`
- `time_range` = the full GABAM calendar year of the source tile (a 1-year window)

## Sampling (bounded-tile, global derived product, §5)

GABAM is global; we do NOT attempt global coverage. We downloaded a bounded set of year
ZIPs (2018, 2019, 2020) and extracted **34 curated 5x5-deg source tiles** from
representative fire-prone biomes across both hemispheres and three post-2016 years:

- Sub-Saharan African savannas (Sahel/Sudanian + Angola/Zambia/DRC miombo) — 2019
- South American cerrado / arc-of-deforestation — 2020
- Northern-Australian savanna — 2020
- Mainland SE Asia dry forest — 2020
- Boreal Siberia — 2018
- Western North America (boreal Canada + California) — 2018
- Central-Asian steppe — 2018
- Mediterranean (Iberia / NW Africa) — 2018

For each tile, non-overlapping ~64px-footprint (BLOCK=22 native px) windows are scanned. A
window is a **burn** candidate if it is >= 10% burned (`BURN_MIN`, a strong high-confidence
burn-scar signal) and a **background** candidate if it is pure not-burned; windows with a
weak/ambiguous partial burn (0 < frac < 10%) are skipped. **Tiles-per-class balanced**
selection (`select_tiles_per_class`, rarest class first) draws up to 1000 tiles/class.
Native 30 m EPSG:4326 windows are reprojected to a local UTM projection at **10 m** with
**nearest** resampling (categorical labels).

### Selected counts

- tiles-per-class: `{not burned: 1000, burned: 1024}` (a tile counts toward every class it
  contains; total 1145 unique patches, well under the 25k cap)
- per region: S Africa 654, N Africa 322, S America 47, N Australia 34, SE Asia 29,
  Central Asia 24, Boreal Siberia 24, W North America 9, Mediterranean 2

African savannas dominate the selection, which **mirrors reality** — Sub-Saharan Africa
accounts for the large majority of global burned area — so this is a representative rather
than skewed sample.

## Verification

- 1145 `.tif` each with a matching `.json`.
- Inspected patches: single-band uint8, UTM CRS at 10 m/px, 64x64, nodata=255; values are
  valid class ids {0, 1}; `change_time=null`, 1-year `time_range` anchored on the source
  tile's year.
- Geographic sanity: sample centers decoded back to lon/lat land inside their source
  tile's 5-deg cell (e.g. `S10E025` -> 27.3E/13.9S Zambia; `N65E100` -> 104.7E/64.3N
  Siberia; `S05E015` -> 19.5E/5.7S DRC).

## Caveats

- Burn labels are a derived product, not in-situ reference; commission/omission errors of
  GABAM propagate. Burn windows require >=10% burned pixels to keep the signal
  high-confidence.
- GABAM masks/leaves non-land as 0; we sample interior fire-prone tiles so value 0
  corresponds to genuine unburned land, but coastal edges of a tile could in principle
  include a little water labeled "not burned" (low impact).
- Post-fire scar persistence is strong in savanna/forest but shorter in some fast-recovering
  grasslands; the 1-year static window is a reasonable approximation across the sampled
  biomes.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gabam_global_annual_burned_area_map --workers 64
```

Idempotent: re-running skips already-written `{sample_id}.tif`, and re-downloading/
re-extracting is skipped when the ZIPs / extracted tiles already exist. Raw ZIPs and the
extracted source tiles live under
`raw/gabam_global_annual_burned_area_map/` on weka.
