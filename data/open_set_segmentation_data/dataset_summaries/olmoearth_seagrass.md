# olmoearth_seagrass

- **Status:** completed
- **Task type:** classification (sparse point segmentation)
- **Num samples:** 2000 (1000 background, 1000 dense_seagrass)
- **Output:** `datasets/olmoearth_seagrass/points.geojson` (spec §2a point table) + `metadata.json`

## Source

Local rslearn project at `/weka/dfive-default/piperw/rslearn_projects/data/seagrass`
(`have_locally: true`; no download — `raw/olmoearth_seagrass/SOURCE.txt` points at it).
Manual Sentinel-2 seagrass annotation over the Balearic Islands (Mallorca, Menorca,
Pitiusas/Ibiza/Formentera).

Two window groups exist:
- `baleares_official_2025` — **40,000 point-label windows** (used). Each window is a 64×64,
  10 m, EPSG:326xx (UTM) raster with exactly **one labeled pixel** (rest = nodata 255). The
  class id, class name, and the point's lon/lat live in window `metadata.json` `options`
  (`label`, `label_name`, `longitude`, `latitude`). Distribution: 20,000 `background`
  (source label 0) + 20,000 `dense_seagrass` (source label 2).
- `baleares_official_eval` — 276 dense 512×512 polygon-derived evaluation tiles. **Excluded:**
  a different label modality (dense polygon rasters, held-out eval), not point supervision.

## Processing decisions

- **Pure sparse-point dataset** → one dataset-wide GeoJSON point table (`points.geojson`),
  no per-sample GeoTIFFs (spec §2a).
- **Classes remapped to contiguous ids:** source label 0→0 `background`, source label 2→1
  `dense_seagrass`. The config also lists `sparse_seagrass` (id 1) but **no such points exist**
  in the data, so it is dropped. Matches the manifest 2-class scheme.
- **Balancing:** `balance_by_class(per_class=1000)` → 1000 per class = 2000 points (well under
  the 25k cap). Ample raw points remain (20k/class) if a larger cap is desired later.
- **Time range:** each point uses its source window `time_range` (2025-01-01 → 2025-12-31,
  ~1 year, post-2016). No change labels (`change_time=null`).
- **Coordinates:** taken directly from the manually-annotated source windows (WGS84 lon/lat).
  Sanity-checked to fall in the Balearics (lon 1.18–4.34, lat 38.58–40.11).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.olmoearth_seagrass
```

Idempotent: rescans window metadata (multiprocessing.Pool(64)) and rewrites the same
deterministic (seeded) point table.

## Caveats

- Only 2 of 3 config classes present; `sparse_seagrass` unused in the point labels.
- Eval polygon tiles not incorporated (see above); could be added later as a dense_raster
  path if desired.
- Assembly step supplies additional negatives per §5; no synthetic negatives fabricated here
  (dataset already carries a real `background` class from the source).
