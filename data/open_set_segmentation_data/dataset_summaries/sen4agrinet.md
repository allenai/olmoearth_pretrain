# Sen4AgriNet (`sen4agrinet`)

**Status:** completed · **Task:** per-pixel classification (crop type) · **Samples:** 3,670 label tiles · **Classes:** 95 (of the 168-code FAO/ICC taxonomy)

## Source

Sen4AgriNet (National Observatory of Athens; Sykas et al. 2022, IEEE JSTARS
10.1109/JSTARS.2022.3164771). A Sentinel-2 multi-year, multi-country crop-type benchmark
annotated from **LPIS farmer declarations**, harmonized to an extended **FAO Indicative
Crop Classification (ICC)** taxonomy.

- Repo: https://github.com/Orion-AI-Lab/S4A (models: .../S4A-Models)
- Data: Hugging Face dataset `orion-ai-lab/S4A` (10,013 NetCDF patches, ~26 MB each).
- License: CC-BY-4.0 (open LPIS declarations; S4A code MIT).
- Coverage: **Catalonia (ES) 2019–2020** and **France (FR) 2019**, over 11 Sentinel-2 MGRS
  tiles, all in UTM zone 31. (The task blurb said "Greece/Catalonia" — the published S4A
  data contain **no Greece**; it is Catalonia + France. Noted as a manifest discrepancy.)

## Georeferencing check (spec §8.2) — PASS

Each `.nc` patch is 366×366 at 10 m. The `labels` group is a georeferenced `uint32` raster
carrying an affine `transform` (e.g. `[10,0,262200,0,-10,4559760]`) and `crs`
(`epsg:32631`) attribute. Patches are therefore fully placeable on the S2 grid — **not**
coordinate-free. Verified sampled tiles land correctly in Catalonia (~40.6–43.5 °N,
0.4–2.1 °E) and northern France (~49.8 °N). Georeferencing is derived directly from the
source raster's native UTM transform (no reprojection needed).

## Label / class mapping

- Source `labels` values are FAO/ICC taxonomy **codes** (`uint32`; vendored name→code map
  from S4A `encodings_en.py`, 168 codes, stored at
  `open_set_segmentation_data/datasets/_sen4agrinet_encoding.py`).
- Code **0** = "no LPIS declaration" / background → written as **nodata 255** (ignore). We
  do **not** invent a background class (spec §5): non-declared pixels are ignore, and the
  assembly step supplies negatives from other datasets.
- Present crop codes are assigned class ids **0..N-1 by descending global pixel frequency**.
  Names come from the taxonomy (e.g. 0=Unknown crops, 1=Olives, 2=Wheat, 3=Barley,
  4=Sunflower, 5=Fallow land, 6=Grapes, 7=Almonds, 8=Rice, 9=Temporary grass crops, …).
- **254-class cap:** the taxonomy has 168 codes (< 254), so all fit in `uint8` and **no
  truncation** was needed. Only **95** codes actually appear in the sampled patches; the
  other 73 taxonomy codes (crops not grown in / not sampled from these regions) are absent
  and simply do not get an id. Downstream assembly drops too-small classes.

## Processing recipe (dense_raster, spec §4)

The `labels` raster is already UTM 10 m, so we **reuse the source CRS** and cut
**non-overlapping 64×64 windows** on a 5×5 grid over the top-left 320×320 of each patch
(the 46-px remainder is dropped). Windows with < 5 % declared (non-zero) pixels are
discarded. Each window → single-band `uint8` GeoTIFF (codes→ids, 0→255) written with
`GeotiffRasterFormat` at `Projection(EPSG:32631, 10, -10)`; pixel bounds computed from the
patch transform (same convention as `cems_wildfire_dataset.py`).

**Sampling:** tiles-per-class balanced (`sampling.select_tiles_per_class`), rarest class
first, `per_class = min(1000, 25000 // 95) = 263`, 25k total cap. Yielded **3,670** tiles
(tiles-per-class balancing saturates at 3,670 because common crops co-occur in almost every
window, so filling rare classes already covers the common ones). Per-class *tile presence*
counts (a tile counts toward every class it contains): Fallow land 2496, Unknown crops
2155, Barley 1399, Olives 1290, Wheat 1173, Almonds 989, Temporary grass 926, Sunflower
873, … down to many single-tile rare crops (kept per spec §5; assembly filters the too-small
ones).

**Time range:** crop labels are seasonal/annual → a 1-year window on each patch's year
(`patch_year`, 2019 or 2020). All post-2016 (Sentinel era). No change labels.

## Bounded sample & caveat (HF rate limit)

The full product is 10,013 patches (~260 GB). Per spec §5 (large products) we sampled a
bounded, geographically-diverse subset: up to 120 patches per (year, tile). **However, HF's
unauthenticated rate limit (HTTP 429; ~3000 requests / 300 s per IP) throttled the download
heavily**, so the sweep only completed the first combos before stalling at ~180 s/patch.
The dataset was therefore built from the **546 patches successfully cached**, distributed as:

```
2019/31TBF:120  2019/31TCF:120  2019/31TCG:120  2019/31TCJ:120  2019/31TCL:33
2019/31TDF:3  2019/31TDG:3  2019/31TDK:3  2019/31TDM:3  2019/31UCP:3  2019/31UDR:3
2020/31TBF:3  2020/31TCF:3  2020/31TCG:3  2020/31TDF:3  2020/31TDG:3
```

So coverage is weighted toward three Catalonia tiles (31TBF/CF/CG) + one French tile
(31TCJ), with the remaining 7 tiles and 2020 only lightly sampled. Both countries and a
range of crops are still represented; class balancing makes the label set class-balanced
regardless of patch skew. **This is not a fundamental limitation** — re-running the script
(with an `HF_TOKEN`, or after the quota resets) pulls the full even sample and expands
coverage. The run is idempotent: cached patches and already-written tiles are skipped.
(Judgment call: chose `completed` on the substantial bounded sample rather than
`temporary_failure`, since a correct full-sized class-balanced dataset was produced; the
skew is documented and trivially expandable.)

## Verification (spec §9)

- 3,670 `.tif` each with a matching `.json`; all single-band `uint8`, EPSG:32631, 10 m,
  64×64, nodata 255.
- All pixel values are valid class ids (0–94) or 255; `metadata.json` class ids cover every
  value present.
- Every sample JSON has a 1-year `time_range` (2019/2020); `change_time` null.
- Spatial sanity: sampled tile centers fall in Catalonia and France (coordinate check).
  Full Sentinel-2 pixel-overlay eyeballing was not performed (would require configuring an
  S2 source); georeferencing math matches the validated `cems_wildfire` convention and the
  source raster's native transform.

## Reproduce

```
# Full even sample (needs HF_TOKEN to avoid 429 throttling; downloads ~50 GB, idempotent):
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sen4agrinet \
    --per-combo 120 --dl-workers 4

# Process only patches already cached in raw/ (no network; what was used here):
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sen4agrinet --offline
```

Outputs on weka: `datasets/sen4agrinet/{metadata.json, registry_entry.json,
locations/{id}.tif,.json}`; raw `.nc` under `raw/sen4agrinet/data/{year}/{tile}/`.
