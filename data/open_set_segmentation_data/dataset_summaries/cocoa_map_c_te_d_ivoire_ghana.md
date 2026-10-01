# Cocoa Map (Côte d'Ivoire & Ghana) — REJECTED (needs-credential)

- **Slug**: `cocoa_map_c_te_d_ivoire_ghana`
- **Status**: `rejected` (`needs-credential`)
- **Manifest**: label_type `points`, family `plantation`, classes `[cocoa, non-cocoa]`,
  region Côte d'Ivoire & Ghana, time_range 2018–2021, license "open (research)",
  have_locally false.
- **Source**: GitHub / Nature Food — Kalischek et al. 2023, "Cocoa plantations are
  associated with deforestation in Côte d'Ivoire and Ghana", *Nature Food* 4:384–393
  (https://doi.org/10.1038/s43016-023-00751-8); preprint arXiv:2206.06119; code repo
  https://github.com/D1noFuzi/cocoamapping .

## What the manifest asks for
The manifest note is explicit: *"prefer the GPS ground-truth samples over the derived
map."* The preferred label signal is the ~100k GPS-mapped cocoa/non-cocoa field points
(manual field survey), i.e. the `points` reference data.

## Why rejected
**1. The preferred GPS ground-truth points are not distributable (copyright).**
The official GitHub repo (`D1noFuzi/cocoamapping`) ships **only a dummy dataset** — two
HDF5 files (`dataset/data/{train,val}.hdf5`), each 100 synthetic patches of shape
`patches (100, 10, 15, 15)` + `gt (100, 1, 15, 15)`, with **no geocoordinates** of any
kind (no lon/lat, CRS, tile id, or transform). The README states verbatim:

> "Sadly, we cannot share our ground truth data due to copyright restrictions. We have
> included a dummy dataset showcasing what our dataloader expects as input."

The paper corroborates that the reference/field data came from industrial partners,
cocoa foundations, and non-profits and is not publicly released. So the ground-truth
points cannot be obtained from any authorized source — a permanent access gate, not a
transient failure.

**2. The fallback derived map is Earth-Engine-gated and the authorized GEE credential is
missing.**
The Kalischek cocoa probability map (and its thresholded cocoa/non-cocoa version, best
threshold P ≥ 0.65) is distributed via a **Google Earth Engine** app only:
`https://nk.users.earthengine.app/view/cocoa-map`. The paper's data-availability
statement says the map "will be released for download and will be available in the Google
Earth Engine" — in practice the GEE app is the sole public access point.
- No public direct-download GeoTIFF was found. The Trase open-data cocoa datasets provide
  only **aggregated hectare statistics per admin unit**, not the 10 m pixel raster. The EU
  Africa Knowledge Platform exposes only a **WMS render service**
  (`geospatial.jrc.ec.europa.eu/geoserver`), i.e. styled map images, not classification
  values, and tiling all of Côte d'Ivoire + Ghana at 10 m through WMS GetMap is both
  low-fidelity and impractical (§8 impractical-download).
- Using the GEE asset would require Earth Engine authentication. The only authorized
  credential per the task spec is in `.env`
  (`TEST_GEE_SERVICE_ACCOUNT_CREDENTIALS`), which points to the key file
  `/etc/credentials/gee_key.json` — **that file does not exist on disk**, so Earth Engine
  cannot be initialized. The spec forbids using GEE credentials found under other paths.

Because the preferred reference points are copyright-withheld and the only fallback
(derived map) is gated behind Earth Engine with no usable authorized credential, the
dataset cannot be processed as-is.

## How to unblock (retry path)
Provide **either**:
- a usable Earth Engine credential — restore/emplace the authorized GEE service-account
  key at `/etc/credentials/gee_key.json` (the path `.env` already references) — and the
  GEE asset ID for the Kalischek cocoa map behind `nk.users.earthengine.app/view/cocoa-map`.
  Then the derived map can be processed as a fallback per §8.2 (sample high-confidence /
  homogeneous windows: P ≥ 0.65 cocoa vs strong non-cocoa), sampled over a bounded set of
  tiles across Côte d'Ivoire & Ghana, 1-year time window anchored in 2019–2021; **or**
- a copy of the authors' GPS ground-truth field samples (the preferred `points` data),
  which are not publicly distributable.

## Reproduce this investigation
```
# repo ships only dummy data, no coordinates:
git clone --depth 1 https://github.com/D1noFuzi/cocoamapping.git
python3 -c "import h5py; f=h5py.File('cocoamapping/dataset/data/train.hdf5'); \
  print({k:f[k].shape for k in f})"   # {'gt': (100,1,15,15), 'patches': (100,10,15,15)}
# map is GEE-only: https://nk.users.earthengine.app/view/cocoa-map
```

## Registry
`write_registry_entry("cocoa_map_c_te_d_ivoire_ghana", "rejected",
notes="needs-credential: Earth Engine ...")`.
