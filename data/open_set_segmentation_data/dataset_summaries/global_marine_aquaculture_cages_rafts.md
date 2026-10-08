# Global Marine Aquaculture (cages & rafts)

- **slug:** `global_marine_aquaculture_cages_rafts`
- **status:** completed
- **task_type:** classification (detection-encoded)
- **num_samples:** 553 tiles (64×64, single-band uint8)
- **classes:** `0 = background`, `1 = finfish_cage` (nodata/ignore = 255)

## Source

Reglab, *"Remote sensing and computer vision for marine aquaculture,"* **Science Advances**
2024, DOI [10.1126/sciadv.adn4944](https://www.science.org/doi/10.1126/sciadv.adn4944).
Code + labels: `github.com/reglab/aquaculture`, archived on **Zenodo record 10933921**
(v1.0.0). We download the 73 MB repo snapshot zip (no credentials needed) and use two
georeferenced GeoJSON layers, both in EPSG:3857:

- `output/humanlabels.geojson` — **4142 manual finfish-cage bounding-box annotations**
  (validation rounds) on French-Mediterranean aerial ortho-imagery, 2002–2021. Each is a
  small circular/square net-pen cage footprint (~12 m median max-dim, 1–3 px @ 10 m), with
  a `year` and cage `type`. **Reference / ground truth.**
- `output/ocean_detections.geojson` — **17 252 YOLOv5 model detections** over the whole
  French-Med coast, 2000–2021, each with a `det_conf`. Used as a **high-confidence fallback
  map** (det_conf ≥ 0.7) to expand coverage beyond the small manual validation set.

Access: unauthenticated Zenodo download. Note the paper's Google-Earth training imagery is
"available on request," but the two georeferenced label layers above are published openly
and are sufficient.

## Scope / provenance judgment calls (please review)

1. **Not global; finfish cages only — no rafts.** The manifest names this "Global … cages &
   rafts" with classes {finfish cage, bivalve/algae raft}. The DOI-matched source is
   **French-Mediterranean finfish net-pen cages only**. There are **no raft annotations** and
   the region is not global. We therefore emit a single foreground class `finfish_cage`; the
   `bivalve/algae raft` class cannot be fabricated and the "Global" region label is
   inaccurate for this source.
2. **Sentinel-era filter.** Labels span 2000–2021; per the ≥2016 rule we keep only
   `year ≥ 2016`. Post-2016: 1152 human cages, 1987 detections (after conf filter).
3. **Reference + fallback-map mix.** Manual GT alone yields only **37 tiles** (cages are
   heavily clustered into ~a few dozen farm-sites; mean ~31 cages / 640 m cell). To make the
   class useful we added high-confidence (`det_conf ≥ 0.7`) model detections as a documented
   fallback map, per the "maps as high-confidence fallback" allowance. Manual GT takes
   precedence over detections within any (year, UTM-zone, cell). Final: 37 human + 516
   detection tiles. Detection tiles may carry occasional false positives; provenance is
   recorded per tile in `source_id` (`human:…` vs `detection>=0.7:…`).

## Encoding (detection → per-pixel classification)

`label_type` in manifest is "polygons / bounding boxes"; cages are sub-/near-resolution
objects marking aquaculture presence, so we use the detection recipe (spec §4):

- Features grid-snapped to **64×64 (640 m) local-UTM tiles** keyed by
  `(year, utm_epsg, cell)`. Two UTM zones present: 32631 (mainland Gulf-of-Lion → Var) and
  32632 (Var → Corsica).
- Per tile: cage footprints rasterized as class **1** (`all_touched=True`, so tiny cages
  keep ≥1 px), each footprint ringed by a **10 px nodata (255) buffer** (ortho-derived
  coordinates may sit a couple px off the Sentinel grid), rest of tile = **background 0**.
- **Negatives:** in-tile background (finfish farms never fill a 640 m tile) supplies
  spatially-meaningful negatives. We do **not** fabricate separate all-ocean negative tiles —
  "confirmed-empty ocean" cannot be reliably derived from this release (human labels cover
  only selected validation image tiles).
- **Time range:** the label's aerial-image `year` as a 1-year window (`year_range`); finfish
  cages are persistent structures, so an annual window is appropriate.

## Counts

- Total tiles: **553** (cap is 1000/class; not reached).
- By source: human = 37, detection = 516.
- By year: 2016:28, 2017:273, 2018:34, 2019:35, 2020:142, 2021:41.
- Class tile counts (a tile counts toward every class present): background in 553, finfish_cage in 553.
- Every tile contains ≥1 `finfish_cage` pixel plus background; nodata = buffer rings.

## Verification (§9)

- Opened 7 tiles: all single-band `uint8`, UTM CRS (EPSG:326xx) @ 10 m, 64×64, nodata=255,
  pixel values ⊆ {0, 1, 255}. Global value set across all 553 tifs = {0, 1, 255}.
- All 553 `.tif` have a matching `.json`; 0 samples with a >1-year `time_range`.
- Geolocation sanity: human-GT tile centers land on documented French-Med finfish sites —
  Corsica (lon ~8.6–9.1, lat ~41.4–41.9), Marseille (5.30, 43.27), Toulon/Bandol
  (5.90, 43.08), Var coast (6.94, 43.48). Tile coordinates cross-validate against the
  source `im_center` fields. (A rendered Sentinel-2 overlay was not produced in this
  headless run; coordinates were validated against the source instead.)
- Re-running the script is idempotent (skips existing `{id}.tif`).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_marine_aquaculture_cages_rafts
```

Raw source (Zenodo zip + unzipped repo) is at
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/global_marine_aquaculture_cages_rafts/`.
Outputs at `…/datasets/global_marine_aquaculture_cages_rafts/` (`metadata.json`,
`locations/{id}.tif` + `.json`).

## Caveats

- Single foreground class only (`finfish_cage`); manifest's raft class and global scope are
  not represented in this source.
- ~93% of tiles come from model detections (conf ≥ 0.7), not manual GT — expect a small
  false-positive rate; `source_id` distinguishes them.
- Small, geographically concentrated dataset (French Mediterranean). Downstream
  pretraining-assembly handles rarity/negatives; not rejected for sparsity per §5.
