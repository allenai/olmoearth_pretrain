# CloudSEN12 — processing summary

- **Slug**: `cloudsen12`
- **Manifest name**: CloudSEN12
- **Status**: **completed**
- **Task type**: classification (per-pixel cloud / cloud-shadow semantic segmentation)
- **Num samples**: **1880** label tiles (64×64, 10 m, local-UTM single-band GeoTIFFs)
- **Family / region**: cloud / **global** (patches across all continents except Antarctica)

## Source

CloudSEN12 / CloudSEN12+ (Aybar et al. 2022, *Sci Data*; Aybar et al. 2024, *Data in Brief*)
— the largest global benchmark of Sentinel-2 patches with hand-crafted pixel labels for
cloud and cloud-shadow semantic segmentation.

- **Access**: public, **no credentials**. Read from the cloud-optimized "tortilla" release
  on HuggingFace `tacofoundation/cloudsen12` (dataset id `tacofoundation:cloudsen12-l1c`),
  via the `tacoreader` (<1.0) client. Manifest URL also points at the Zenodo mirror
  (record 7034410). CloudSEN12 is CC-BY-4.0 (the CloudSEN12+ refresh is CC0).
- **What we used**: only the **high-quality manual** tier (`label_type == "high"`) at the
  standard **509×509, 10 m** patch size (`real_proj_shape == 509`) — 10,000 patches, each a
  single Sentinel-2 L1C acquisition already stored in its native local UTM zone at 10 m.
  Acquisition dates span **2018–2020** (entirely Sentinel-2 era, post-2016 — no filtering
  needed). The 2000×2000 "high" patches and the scribble/nolabel tiers were not used.
- **Annotation method**: manual pixel-level expert annotation with IRIS/CloudApp, reviewed
  under the CloudSEN12 labeling protocol (reference-grade).

## Triage decision

**Accept**, classification. Georeferenced (every patch carries CRS + geotransform in its
native UTM at 10 m), post-2016, public/no-credential, and the phenomenon (clouds/shadows)
is exactly what S2 observes at 10 m. This is a **per-image `dense_raster` label, NOT a
change task** — a cloud mask describes one acquisition, so `change_time = null`.

## Class mapping (4-class high-tier scheme → output ids 0–3, identity)

| id | name | description |
|----|------|-------------|
| 0 | clear | pixels free of clouds and cloud shadows (clear land/water) |
| 1 | thick cloud | opaque clouds that fully block the surface signal |
| 2 | thin cloud | semi-transparent / cirrus clouds; ground still partly visible |
| 3 | cloud shadow | shadows cast on the surface by clouds |

`nodata = 255`. Every pixel in a "high" patch is labeled, so no nodata occurs in practice;
255 is declared only for the open-set contract. **Reads are validated to contain only
{0,1,2,3}** — the scribble tier's 0..6 scheme and the 99/fill values were rejected by an
explicit validator (this guards against the read bug described in Caveats).

## Processing recipe (dense_raster, tiles-per-class balanced)

1. **Index once, cache offline.** The tortilla index is built with `tacoreader` and cached
   to `raw/cloudsen12/index.parquet`; every later run reconstructs the `TortillaDataFrame`
   from that parquet with **no HuggingFace request** (the HF index build is heavy and the
   first thing anonymous rate-limiting kills).
2. **Selective, class-diverse download** (spec §5 bounded sampling). CloudSEN12 is large;
   we download only a bounded subset of **2,500 of the 10,000** high patches, chosen with
   `select_patch_subset` to prioritize the rarest class (**thin cloud**, present in only
   ~3.2k of 10k patches vs 6–9k for the others). This is far more than enough to fill
   ≥1000 tiles/class. `--max-patches 0` would download all 10k.
3. **One HTTP request per patch.** For each patch a single byte-range GET fetches its
   tortilla blob; the nested footer is parsed **locally**, the tiny single-band `target`
   label GeoTIFF is sliced out (rasterio `MemoryFile`), and the **S2 imagery bytes are
   discarded** (pretraining supplies imagery). Labels are cached per-patch as `.npy`
   (`raw/cloudsen12/labels/`), so runs are resumable and idempotent. This ~1-request/patch
   path replaces tacoreader's multi-request vsicurl reads, keeping us under HF's anonymous
   limit (~3000 resolver requests / 300 s).
4. **Tile** each 509×509 label into **64×64** windows on an 8×8 grid (offsets
   `[0,64,127,191,254,318,381,445]`, evenly covering the valid extent, never touching the
   512-pad border) → 64 candidate tiles/patch, 160,000 total.
5. **Tiles-per-class balanced** selection (`sampling.select_tiles_per_class`, rarest-first,
   ≤1000 tiles/class, 25k cap): 160,000 → **1880** tiles.
6. GeoTIFFs (uint8, nodata 255) written in the patch's native UTM at 10 m using the source
   geotransform directly (exact georeferencing) + per-sample JSON; idempotent (skips
   existing tifs). Parallelized with `multiprocessing.Pool(32)` + `star_imap_unordered`.

**Time range**: cloud masks are per-image/transient labels, so `change_time = null` and
`time_range` is a short window **±15 days centered on the S2 acquisition date** (~1 month,
well under the 1-year cap). This keeps paired pretraining imagery temporally near the
labeled scene (spec §5 specific-image rule).

## Sample counts

Candidate tiles: **160,000** from 2,500 patches (0 patches failed). Candidate tiles per
class: clear 90,377; thick 75,471; thin 85,693; shadow 45,615. Final selected: **1880**
tiles. Per-class tile counts (a tile counts toward every class present in it):

| id | class | tiles |
|----|-------|------:|
| 0 | clear | 1000 |
| 1 | thick cloud | 1220 |
| 2 | thin cloud | 1053 |
| 3 | cloud shadow | 1270 |

Every class reached ≥1000 tiles (common classes overshoot because they co-occur in tiles
selected for rarer classes). Only 4 classes → no 254-class-cap issue.

## Caveats / judgment calls

- **Fixed a latent read bug from the interrupted prior attempt.** `tacoreader.load` sorts
  rows by `tortilla:id` but leaves the pandas index **labels randomized (they differ every
  load)**, while `TortillaDataFrame.read(i)` is **positional**. The prior script passed a
  per-load pandas label as the positional index to workers (each of which had its own
  load), so it read the **wrong rows** — yielding scribble-tier (0..6) and fill (99) values
  instead of the 4-class labels. The rewrite avoids `.read(idx)` across processes entirely
  (workers fetch by parsed `url`/`blob_offset`/`blob_length`), and a validator rejects any
  label containing values outside {0,1,2,3}. Output labels are confirmed clean.
- **HuggingFace anonymous rate-limiting (HTTP 429).** The source is public but limited to
  ~3000 resolver requests/300 s and there is **no HF token in `.env`**
  (CloudSEN12 is not gated, just rate-limited). The 1-request-per-patch design + local
  index cache + `.npy` resume + exponential backoff (honoring `Retry-After`) keep the run
  well under the limit; the full download of 2,500 patches finished in ~1 min with 0 429s.
- **Bounded sampling**, not global coverage: 2,500 of 10,000 high patches. Documented per
  spec §5; the subset still spans many countries and all four classes abundantly.
- **Downloaded blobs include S2 imagery bytes** (the tortilla packs image+label together);
  we range-GET the whole ~1.3 MB blob but **write only the label** and never persist the
  imagery. This is the pragmatic way to keep to 1 request/patch under the rate limit.
- **Verification** (spec §9): 60 random tifs are all single-band uint8, 64×64, local UTM
  (diverse zones 326xx/327xx) at 10 m, values ⊆ {0,1,2,3} with nodata 255; all 1880 tifs
  have a matching JSON with a ≤1-year `time_range` and `change_time = null`; metadata class
  ids cover all values present. Tile centroids reproject to sensible worldwide land/ocean
  locations (e.g. Ukraine, Colombia, China, E. Siberia, W. Sahara).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.cloudsen12 \
    --workers 32 --max-patches 2500
# first run primes raw/cloudsen12/index.parquet from HuggingFace; later runs are offline
# for the index and resume label downloads from raw/cloudsen12/labels/*.npy (idempotent).
```
