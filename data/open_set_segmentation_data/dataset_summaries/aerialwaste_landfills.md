# AerialWaste (landfills) — REJECTED

- **Slug:** `aerialwaste_landfills`
- **Name:** AerialWaste (landfills)
- **Source:** Zenodo record 7034382 (Torres & Fraternali 2023, *Scientific Data* 10:63,
  "AerialWaste dataset for landfill discovery in aerial and satellite images"). Project
  site <https://aerialwaste.org/>, code <https://github.com/rnt-pmi/AerialWaste>.
  DOI <https://doi.org/10.5281/zenodo.7034382>.
- **Family / region:** waste / Lombardy, Italy. **Manifest label_type:** "dense_raster +
  image labels". **Time range (manifest):** 2016–2021.
- **License:** Zenodo/manifest say CC-BY; the GitHub README says **CC-BY-NC-ND** (see
  license caveat below).
- **Status:** **REJECTED**
- **Reason:** `no recoverable geocoordinates` — the released ML-ready PNG image tiles are
  named only by a numeric id and carry no lon/lat, no CRS, no geotransform, no bbox, and
  no MGRS/tile index, so neither the image-level labels nor the (pixel-space) masks can be
  placed on the Sentinel-2 grid. (Secondary blockers: sub-metre waste objects unobservable
  at 10–30 m; scene-level candidate-site crops; restrictive NC-ND license variant.)

## What the dataset is

AerialWaste is an illegal-landfill *discovery* (image-classification / weak-localization)
benchmark: **10,434** aerial/satellite image crops of candidate waste-dump sites in
Lombardy, Italy, expert-photointerpreted by ARPA analysts. Annotations come at three
granularities: (1) **binary** waste presence/absence (`is_candidate_location`), (2)
**multi-class multi-label** waste object / storage-mode categories on a subset
(`valid_fine_grain=1`, 715 images; 22 categories across two supercategories
`Type_of_object` and `Storage_mode`), and (3) **weakly-supervised segmentation masks**
around relevant waste objects (COCO-style, in image-pixel space) on a further subset.

Image sources: **GE** (Google Earth) 6,750, **AGEA** (Italian agricultural aerial
orthophotos) 3,450, **WV3** (WorldView-3) 234 — all delivered as `.png`. Aerial/GE imagery
is ~0.2–0.5 m/px.

## Triage — checked cheaply, before any bulk download (spec §8.2)

The image data is ~18 GB (six `imagesN.zip`, ~3 GB each). Per the spec's "check
georeferencing cheaply first" rule, only the two small COCO annotation files were pulled
(`training.json` 2.9 MB, `testing.json` 1.7 MB → `raw/aerialwaste_landfills/`); the image
zips were **not** downloaded.

Every image record has exactly these keys (verified across all 10,434 records in both
splits):

```
file_name, id, categories, img_source, site_type, severity, evidence,
valid_fine_grain, width, height, is_candidate_location
```

There is **no** latitude/longitude, CRS, EPSG, geotransform, corner-coordinate, bounding
box, or MGRS/tile field anywhere — at the top level (`info`, `categories`, `images`,
`meta_properties`), in any image record, or referenced by the README schema. `width`/
`height` are pixel dimensions of the crop only. Images are keyed solely by an integer
`id` (`3456.png`, …).

## Why rejected — no recoverable geocoordinates (fundamental, spec §8.2 / §2)

1. **No recoverable geocoordinates (primary).** OlmoEarth pretraining co-locates labels
   with Sentinel-2/S1/Landsat imagery by geography + time. AerialWaste's labels (image-
   level classes and the COCO pixel masks alike) live in the pixel space of anonymous PNG
   crops with no lon/lat and no CRS, so they cannot be projected onto the S2/UTM grid. A
   per-image numeric id is not sufficient (spec §8: "A per-sample tile/region id alone …
   is not sufficient"). This is the canonical fast "ML-ready PNG tiles strip lon/lat"
   rejection. The omission is by design: exact locations of (often illegal) landfills are
   deliberately withheld for enforcement/privacy reasons, so no per-image coordinate table
   is published on Zenodo, the GitHub repo, or aerialwaste.org.

2. **Phenomenon not observable at 10–30 m (secondary).** Even with coordinates, the
   fine-grained targets — heaps, full containers, big bags, pallets, drums/bins, tyres,
   scrap, corrugated (asbestos) sheets, etc. — are sub-metre to a few metres and were
   annotated on ~0.2 m aerial imagery; they are not resolvable by Sentinel-2/Landsat at
   10–30 m (spec §8). Only whole large landfill *sites* could conceivably be observable,
   which reduces the usable signal to scene-level presence.

3. **Scene-level patch classification (secondary).** The dominant annotation is binary
   waste presence over heterogeneous candidate-site crops (production sites, farms,
   degraded/abandoned areas). That is scene/patch classification, not per-pixel
   segmentation of a coherent land-cover patch (spec §4 "scene-level"), and would be
   rejected on that ground too.

4. **License caveat.** The GitHub README states **CC-BY-NC-ND** and that Google Imagery
   use must follow Google Earth terms; NC-ND forbids derivative redistribution. The Zenodo
   record and manifest list plain CC-BY. This discrepancy is a further reason for caution,
   but the decisive, permanent blocker is (1).

Because the primary blocker is permanent (the georeferencing simply is not in the release
and is intentionally withheld), this is `rejected` (fundamental), not `temporary_failure`
or `needs-credential`. No `datasets/` label outputs were written; the only weka artifacts
are the two triage JSONs under `raw/aerialwaste_landfills/` and this dataset's
`registry_entry.json`.

## If this is ever revisited

Salvage would require the authors/ARPA to release a per-image **coordinate or footprint
table** (lon/lat centre or UTM bounds + acquisition date per `id`). With that, only the
large-site scene labels could be recast — e.g. as a static-window `waste_site` presence
class over a small ≤64×64 tile — and only for sites genuinely observable at 10–30 m; the
fine waste-object masks would remain unusable at Sentinel resolution. Absent that table
the dataset cannot be co-located with pretraining imagery.

## Reproduce (triage only)

```bash
python3 - <<'PY'
from olmoearth_pretrain.open_set_segmentation_data import download, io
rd = io.raw_dir("aerialwaste_landfills")
download.download_zenodo("7034382", rd, filenames=["training.json", "testing.json"])
import json
d = json.load(open(rd / "training.json"))
print(sorted({k for im in d["images"] for k in im}))  # no lat/lon/crs/bbox present
PY
```
