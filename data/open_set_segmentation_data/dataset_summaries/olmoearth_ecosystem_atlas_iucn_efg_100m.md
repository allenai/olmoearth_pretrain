# OlmoEarth Ecosystem Atlas IUCN EFG (100m)

- **Slug:** `olmoearth_ecosystem_atlas_iucn_efg_100m`
- **Status:** completed · classification (presence-only points) · **18,931 points**
- **Source:** OlmoEarth Ecosystem Atlas (internal) · **License:** internal
- **Annotation method:** expert visual interpretation (IUCN Global Ecosystem Typology / EFG).

## Source & access

Local artifact (`have_locally=true`, not copied — see `raw/{slug}/SOURCE.txt`):
`/weka/dfive-default/rslearn-eai/artifacts/ecosystem_atlas_labels_20260716.geojson`. Sparse
IUCN Ecosystem Functional Group (EFG) reference points interpreted at **100 m** resolution (the
coarser-resolution companion to `olmoearth_ecosystem_atlas_iucn_efg_10m`). No imagery pulled.

## Label type — presence-only points

Emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a): each coded
point carries its canonical EFG code as the class. One point per location; no GeoTIFF context,
buffer, or negative tiles. Paired with S2/S1/Landsat at pretraining time by lon/lat + time
overlap.

## Classes / counts

**82 classes**, one per IUCN EFG code, ids 0–81 in descending point-frequency order.
**All coded points are kept — no class balancing and no 25k cap** (per the data owner), so counts
are the raw interpreted distribution. **18,931 points total.** Full class list (code → EFG name)
is in `metadata.json`.

Most frequent classes:

| code | EFG name | pts |
|------|----------|-----|
| T5.5 | Hyper-arid deserts | 1561 |
| T3.4 | Young rocky pavements, lava flows and screes | 1526 |
| T5.1 | Semi-desert steppe | 1426 |
| T7.1 | Annual croplands | 1412 |
| T5.4 | Cool deserts and semi-deserts | 1403 |
| T7.4 | Urban and industrial ecosystems | 984 |
| T6.4 | Temperate alpine grasslands and shrublands | 833 |
| T6.3 | Polar tundra and deserts | 775 |

The long tail includes many rare classes (4 classes have a single point), all retained.

## Class normalization

- Two on-disk code string formats are normalized to the canonical EFG codes.
- `Unknown` / `Data-deficient` codes are **dropped**.
- `Open ocean` is **kept** as `M_OPEN_OCEAN`.
- Secondary codes are ignored (primary EFG code only).

## Time handling

1-year `time_range` on each point's `start_time` year (mostly 2025; fallback 2025 when missing).
`change_time = null`.

## Output

- `datasets/olmoearth_ecosystem_atlas_iucn_efg_100m/points.geojson`
- `datasets/olmoearth_ecosystem_atlas_iucn_efg_100m/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_ecosystem_atlas_iucn_efg_100m
```

Idempotent (rewrites `points.geojson`).

## Caveats

- 1×1 point labels carry no spatial context by design (sparse point segmentation).
- Highly imbalanced (no per-class cap applied, per the data owner); many single-point classes.
