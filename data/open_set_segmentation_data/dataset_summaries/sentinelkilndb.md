# SentinelKilnDB

- **Slug:** `sentinelkilndb`
- **Status:** completed
- **Task type:** classification (oriented-box detection encoded as per-pixel classes)
- **Num samples:** 4,500 (3,000 positive kiln tiles + 1,500 background negatives)

## Source

SentinelKilnDB (NeurIPS 2025 Datasets & Benchmarks), a hand-validated benchmark of 62,671
brick kilns across the Indo-Gangetic Plain (India, Pakistan, Bangladesh, Afghanistan),
annotated as **oriented bounding boxes (OBBs)** on free Sentinel-2 surface-reflectance
imagery. Three kiln types: **FCBK** (Fixed Chimney Bull's Trench Kiln), **CFCBK** (Circular
FCBK), **Zigzag**.

- URL: https://huggingface.co/datasets/SustainabilityLabIITGN/SentinelKilnDB
- License: **CC-BY-NC-4.0** (non-commercial; used here for research pretraining, recorded in
  `metadata.json`).
- `have_locally: false`. Downloaded the three parquet files (train 2.35 GB, val 0.78 GB,
  test 0.61 GB; ~3.7 GB total) to `raw/sentinelkilndb/{train,val,test}/*.parquet`.

## On-disk form / access

Each parquet row is one **128x128 px @ 10 m Sentinel-2 patch**:
- `image_name = "{lat}_{lon}.png"` — the patch **center** lon/lat (see georeferencing).
- `image` — PNG bytes of the S2 patch (**not used**; we need only labels + georef).
- `dota_label` — list of OBB strings `"x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult"`, the
  8 corner coords in the patch's 128-px pixel space (also mirrored as normalized
  `yolo_obb_label` / `yolo_aa_label`, unused).

We read only the `image_name` + `dota_label` columns. Patches tile a lat/lon grid with a
30-px overlap (grid step 128-30 = 98 px ≈ 0.0088° lat), so a physical kiln can appear in up
to four overlapping patches.

## Georeferencing (spec §8.2 check)

The filename lon/lat is the **patch center**; each patch is a north-up S2 crop at 10 m, so
image→UTM is a pure translation: take the patch's local UTM projection from (lon, lat), find
the UTM pixel of the center, and map image pixel `(px, py)` → UTM pixel
`(center_col - 64 + px, center_row - 64 + py)`.

Center-vs-corner convention (a 640 m ambiguity) was **verified against Sentinel-2**: for
several patches I fetched a 256×256 S2 window centered on the filename coordinate and
cross-correlated the stored patch PNG against it. The best NCC offset was consistently
(64, 64) = center convention (NCC 0.66–0.77), not (128, 128) = corner. A visual overlay of
produced label tiles on freshly-fetched S2 (Planetary Computer, same Nov 2023–Feb 2024
window) shows the kiln labels sitting exactly on the elongated brick-kiln structures.

## Encoding (label_type = oriented boxes → detection, spec §4)

One **64×64 UTM 10 m context tile** centered on each (deduplicated) kiln. The kiln's OBB
footprint is rasterized (`all_touched`) as its class id, ringed by a **5-px nodata (255)
buffer** to absorb annotation/georef slop, with **background (0)** filling the rest. Any
other kiln whose footprint falls inside the tile (same UTM zone) is rasterized too.
Background-only **negative tiles** are emitted from empty patches (detection exception,
spec §5), centered on the empty patch center.

- **Classes:** `0=background, 1=FCBK, 2=CFCBK, 3=Zigzag` (kiln ids follow the manifest class
  order; background prepended for detection). uint8, nodata 255.
- **Tile/buffer:** `tile_size=64`, `buffer_size=5` px. Buffer is 5 (not the point-detection
  default 10) because the label is a real rasterized footprint, not an imprecise point; 5 px
  still leaves ample background in a 64-px tile.

### Deduplication (judgment call)

The dataset ships 97,648 OBB annotations but only ~62,671 unique physical kilns — overlapping
patches re-annotate the same kiln. Naive integer-pixel-centroid dedup left 93,985 "unique"
kilns (heavy residual duplication) because sub-pixel reprojection + independent per-patch
annotation push the same kiln 1–3 px apart. I use **tolerance-based spatial clustering
(radius 5 px = 50 m)**, giving **69,535 unique kilns** (FCBK 37,766; CFCBK 2,122; Zigzag
29,647) — close to the paper's 62,671 (residual gap is genuinely-nearby kilns). 5 px (50 m)
is safe: kiln footprints are ~100–150 m, so distinct kiln centers are never within 50 m.
This matters most for the rare CFCBK class (avoids selecting near-duplicate tiles).

## Sampling (spec §5)

Class-balanced up to **1,000 tiles per kiln class** (rare CFCBK prioritized), + **1,500
background negatives**. Well under the 25k cap. Realized per-class tile counts (a tile counts
toward every kiln class actually rendered in it):

| class | tiles |
|-------|-------|
| FCBK (1)  | 1,250 |
| CFCBK (2) | 1,018 |
| Zigzag (3)| 1,142 |
| background negatives | 1,500 |
| **total samples** | **4,500** |

(FCBK/Zigzag exceed 1,000 slightly because a tile selected for one kiln class often also
contains neighbouring kilns of another class.)

## Time range

Brick kilns are **persistent structures**; the source imagery is Nov 2023 – Feb 2024. Each
sample uses a static 1-year window **[2023-11-01, 2024-11-01)** with `change_time=null`
(spec §5, static/persistent label). Not a change dataset.

## Caveats

- Non-commercial license (CC-BY-NC-4.0).
- Dedup radius 5 px is a heuristic; the produced kiln count (69,535) slightly exceeds the
  paper's stated 62,671 unique kilns.
- Difficult-flagged boxes (`difficult` field in DOTA) are kept (not filtered).
- CFCBK is the rarest class (~2,122 unique); ~1,018 tiles is essentially the full usable set.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sentinelkilndb
```

Idempotent (skips already-written tiles; recomputes metadata counts from sidecars on rerun).
Outputs: `datasets/sentinelkilndb/{metadata.json, locations/{000000..004499}.{tif,json}}`
on weka.
