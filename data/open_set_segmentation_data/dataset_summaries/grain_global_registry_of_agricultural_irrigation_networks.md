# GRAIN (Global Registry of Agricultural Irrigation Networks)

- **Slug**: `grain_global_registry_of_agricultural_irrigation_networks`
- **Status**: completed
- **Task type**: classification (positive-only line segmentation)
- **Num samples**: 2586 tiles (64×64, local-UTM, 10 m)
- **Per-class tile counts**: irrigation_canal 974, urban_canal 1054, navigational_waterway 1065
  (a tile counts toward every canal class present in it, so counts sum to > num_samples)

## Source

GRAIN v.1.0 — a global, OpenStreetMap-derived vector dataset of the world's canal
centerlines, refined by an ML classifier that separates **agricultural** canals from
urban / navigational / natural waterways. ~3.8 M km of canal `LineString`s across 95
countries.

- Zenodo record **16786488** (doi:10.5281/zenodo.16786488), license **CC-BY-4.0**.
- Codebase: https://github.com/SarathUW/GRAIN (published via ESSD).
- Distribution: a single **1.9 GB** zip (`GRAIN_v.1.0.zip`) containing per-country/region
  **GeoParquet** and ESRI shapefiles. Geometry CRS **EPSG:4326** (WGS84 degrees).

### Access method (selective, bounded — no bulk download)

The 1.9 GB is dominated by the shapefile copies (e.g. `germany` .dbf alone is ~1 GB); the
GeoParquet copies of the same data are far smaller. We did **not** download the whole zip.
The per-dataset script uses HTTP **Range requests** into the remote Zenodo zip
(`download.HttpRangeFile` + `zipfile`) to extract **only** the GeoParquet members for a
representative bounded country subset — ~419 MB fetched total — into
`raw/{slug}/GeoParquet/`. This satisfies the spec S8 "impractical-download / selective
extraction" rule and the S5 "large global derived-product → bounded sampling" rule.

## Classes (from the `canal_use` attribute)

GRAIN's per-segment `canal_use` field gives exactly the three manifest classes; `Other` is
dropped (semantically ambiguous), and `predicted_class == 'Canal_natural'` rows are dropped
(natural channels, not built canals — only `predicted_class == 'canal'` kept).

| id | name | GRAIN `canal_use` |
|----|------|-------------------|
| 0 | irrigation_canal | `Agricultural` (dominant; the dataset's namesake) |
| 1 | urban_canal | `Urban Waterway` |
| 2 | navigational_waterway | `Navigational Waterway` (rarest; typically widest) |

Non-canal pixels are **nodata (255)**. This is a **positive-only** mask (spec S5): no
background class or synthetic negatives are fabricated; the pretraining-assembly step
supplies negatives from other datasets.

Segment class distribution over the sampled countries (2,644,635 segments):
irrigation_canal 2,460,476 · urban_canal 160,058 · navigational_waterway 24,101.

## Label recipe (spec S4 "lines")

- Canal segments are partitioned onto a **~640 m latitude-aware geographic grid**; each
  occupied cell → one **64×64** (640 m) tile in the local **UTM** projection at **10 m**,
  centered on the cell center.
- Every canal segment in the cell is reprojected to UTM pixel space, its centerline
  **buffered ~1 px** (→ ~2–3 px, **20–30 m** wide, `all_touched=True`), and rasterized with
  value = class id. Rarer classes are drawn **last** so they win pixel conflicts
  (irrigation → urban → navigational).
- Tiles whose rasterized canal mask has **< 3 px** are dropped (75 dropped).

### Observability at 10 m (caveat)

GRAIN carries **no width attribute**. Major irrigation/navigational canals and aqueducts
are 10–50 m wide and clearly resolvable at 10 m in Sentinel-2; the narrowest field ditches
are sub-pixel. We rasterize every centerline with ~1 px dilation so each canal becomes a
nominal ~20–30 m linear label — meaningful linear-infrastructure signal for S2/S1/Landsat —
while acknowledging the narrowest ditches are near/below the 10 m limit and the dilated
label is nominal for those. OSM omissions/misclassifications and a few pixels of positional
error are possible.

## Sampling (spec S5, bounded)

GRAIN is a global product; global coverage was **not** attempted. A representative bounded
**country subset** was selectively extracted, spanning every inhabited continent and
arid/temperate/tropical climates, and including the navigational-canal-rich networks of
Europe / China / the US so the rare `navigational_waterway` class reaches its target:

`netherlands, france, poland, spain, sweden, england, china, india_northern-zone,
india_southern-zone, pakistan, bangladesh, indonesia, vietnam, japan, us-midwest, us-south,
us-west, mexico, brazil, argentina, egypt, south-africa, australia, iran, iraq`

Candidate cells are **tiles-per-class balanced** (rarest class first) to **≤ 1000
tiles/class** via `sampling.balance_tiles_by_class` (25k total cap; not binding here).

## Time range (spec S5, static labels)

Canals are persistent static infrastructure (OSM; update date 2025; manifest range
2016–2025). Each tile gets a **static 1-year window** (`change_time = null`) spread
deterministically over **2019–2024** for imagery diversity (≈420–440 tiles/year).

## Verification (spec S9)

- Opened sample `.tif`s: single band, **uint8**, local **UTM** CRS at **10 m**, **64×64**,
  values ∈ {0, 1, 2, 255} with nodata 255. 44 distinct UTM zones (good global spread).
- All 2586 `.tif`s have a matching `.json`; time-range span ≤ 366 days (1-year windows).
- `metadata.json` class ids {0,1,2} cover all non-nodata values in the tiles.
- **Spatial sanity**: tile centers reverse-project to canonical canal regions — Phoenix AZ
  (urban_canal), Miami FL (navigational_waterway), California Central Valley & France
  (irrigation_canal), UK midlands (urban_canal), Kalimantan/Indonesia (irrigation_canal),
  Patagonia AR (navigational_waterway). Placement is consistent with real canal networks.
  (A pixel-level Sentinel-2 raster overlay was not run; coordinate plausibility across
  representative samples was used instead.)
- Re-running the script is **idempotent** (all 2586 tiles skipped on re-run).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.grain_global_registry_of_agricultural_irrigation_networks
```

Outputs on weka under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/grain_global_registry_of_agricultural_irrigation_networks/`
(`metadata.json`, `locations/{id}.tif|.json`, `registry_entry.json`); selectively-extracted
raw GeoParquet under `raw/grain_global_registry_of_agricultural_irrigation_networks/`.
