# OlmoEarth Ecosystem Atlas IUCN EFG (10m)

- **Slug:** `olmoearth_ecosystem_atlas_iucn_efg_10m`
- **Status:** completed · classification (presence-only points) · **12,130 points**
- **Source:** OlmoEarth Ecosystem Atlas (internal) · **License:** internal
- **Annotation method:** expert visual interpretation (IUCN Global Ecosystem Typology / EFG).

## Source & access

Local artifact (`have_locally=true`, not copied — see `raw/{slug}/SOURCE.txt`):
`/weka/dfive-default/rslearn-eai/artifacts/ecosystem_atlas_labels_20260716.geojson`. Sparse
IUCN Ecosystem Functional Group (EFG) reference points interpreted at **10 m** resolution. No
imagery pulled.

## Label type — presence-only points

Emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a): each coded
point carries its canonical EFG code as the class. One point per location; no GeoTIFF context,
buffer, or negative tiles. Paired with S2/S1/Landsat at pretraining time by lon/lat + time
overlap.

## Classes / counts

**81 classes**, one per IUCN EFG code, ids 0–80 in descending point-frequency order.
**All coded points are kept — no class balancing and no 25k cap** (per the data owner), so counts
are the raw interpreted distribution. **12,130 points total.** Full class list (code → EFG name)
is in `metadata.json`.

Most frequent classes:

| code | EFG name | pts |
|------|----------|-----|
| T5.4 | Cool deserts and semi-deserts | 1392 |
| T7.1 | Annual croplands | 944 |
| T6.4 | Temperate alpine grasslands and shrublands | 796 |
| T4.5 | Temperate subhumid grasslands | 673 |
| T3.4 | Young rocky pavements, lava flows and screes | 657 |
| T5.1 | Semi-desert steppe | 611 |
| T7.4 | Urban and industrial ecosystems | 588 |
| T1.1 | Tropical/Subtropical lowland rainforests | 580 |

The long tail includes many rare classes (3 classes have a single point, e.g. M4.1, M2.1, M1.8),
all retained.

## Class normalization

- Two on-disk code string formats are normalized to the canonical EFG codes.
- `Unknown` / `Data-deficient` codes are **dropped**.
- `Open ocean` is **kept** as `M_OPEN_OCEAN`.
- Secondary codes are ignored (primary EFG code only).

## Time handling

1-year `time_range` on each point's `start_time` year (mostly 2025; fallback 2025 when missing).
`change_time = null`.

## Output

- `datasets/olmoearth_ecosystem_atlas_iucn_efg_10m/points.geojson`
- `datasets/olmoearth_ecosystem_atlas_iucn_efg_10m/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_ecosystem_atlas_iucn_efg_10m
```

Idempotent (rewrites `points.geojson`).

## Caveats

- 1×1 point labels carry no spatial context by design (sparse point segmentation).
- Highly imbalanced (no per-class cap applied, per the data owner); many single-point classes.
