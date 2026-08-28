# S2-SHIPS

- **Slug:** `s2_ships`
- **Status:** rejected (`needs-credential: email request to dataset authors`)
- **Source:** GitHub / MDPI — https://github.com/alina2204/contrastive_SSL_ship_detection
  (Ciocarlan & Stoian, "Ship Detection in Sentinel-2 Multi-Spectral Images with
  Self-Supervised Learning", *Remote Sensing* 13(21):4255, 2021,
  https://www.mdpi.com/2072-4292/13/21/4255)
- **Label type / task:** dense_raster → per-pixel classification (classes: ship, water, land)
- **Region / time:** European ports, Panama Canal, Suez Canal; 2018–2021.

## What the dataset is

S2-SHIPS is a pixel-level Sentinel-2 ship-segmentation dataset over ports, straits, and
the Suez Canal. Per the repo README it comprises COCO annotation files, the 12 spectral
bands for each S2-SHIPS tile (GeoTIFF or numpy), the S2-SHIPS segmentation masks, and
water masks. As a georeferenced dense raster of manually annotated ship pixels it would be
a good fit for this pipeline (per-pixel classification, observable at 10 m for larger
vessels; complements the internal box-based vessel evals per the manifest note).

## Why it was rejected

The dataset is **not publicly downloadable**. The repository README states:

> "Please contact alina.ciocarlan@polytechnique.edu to access the dataset."

Access requires an out-of-band email request to the authors — a credential/access gate,
not an open download. Investigation performed before rejecting:

- The GitHub repo (`alina2204/contrastive_SSL_ship_detection`) contains **only code**
  (COCO-mask generation, U-Net training, SSL pretext scripts); no data files, no data URL.
  `grep` across `s2ships_gen_data.py`, `gen_patches.py`, `coco_create_ships_masks.py`,
  `datasets.py` found no download link (the only `url` field is an empty string).
- **No GitHub releases / release assets** on the repo (checked via the GitHub API).
- **No public mirror** on Zenodo or Hugging Face for *this* dataset. Web/HF search surfaces
  only *different* Sentinel-2 ship datasets — the Danish-waters "Sentinel-2 dataset for ship
  detection" (Zenodo 3923841 / 10418786), "Ship-S2-AIS" (Zenodo 7229756 /
  HF `isaaccorley/ships-s2-ais`), and the Finnish-coast vessel set (Zenodo 15019034) —
  none of which are the S2-SHIPS ports/straits/Suez ship-*segmentation* dataset with the
  ship/water/land pixel masks described here.
- `.env` holds no credential applicable to a personal-email access
  request to a university researcher (its creds cover NASA Earthdata, Copernicus, CDS,
  USGS M2M, Planet, GEE — none apply here), so no authorized access path exists.

Per the task spec §8, a dataset blocked only on a missing credential / out-of-band access
grant is a **`rejected`** with `notes: "needs-credential: ..."` (not `temporary_failure`,
since this is a permanent access gate, not a transient source/infra error).

## How to reproduce / recover

If the authors grant access (email alina.ciocarlan@polytechnique.edu), the delivered
archive should contain per-tile 12-band Sentinel-2 GeoTIFFs plus ship + water segmentation
masks. If those GeoTIFFs retain their Sentinel-2 UTM georeferencing, processing is
straightforward: place the archive under
`raw/s2_ships/`, reproject/crop each tile's mask into ≤64×64 UTM 10 m windows via the
shared `dense_raster` path (classes background/land/water/ship or ship-vs-nonship),
tiles-per-class balanced (`sampling.select_tiles_per_class`), time range = 1 year anchored
on each source S2 acquisition date (available from the S2 product id). Verify the masks are
georeferenced (not anonymized pixel arrays) before committing effort — if the delivered
masks are ungeoreferenced numpy/PNG patches, the dataset would instead be rejected for
"no recoverable geocoordinates."
