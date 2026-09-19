# MADOS (Marine Debris and Oil Spill) — mados_marine_debris_and_oil_spill

- **Status**: rejected
- **Reason**: no recoverable geocoordinates (labels cannot be placed on the S2 grid)
- **Task type (if it were usable)**: classification (dense_raster)
- **Source**: Zenodo record 10664073 (Kikaki, Kakogeorgiou, Hoteit, Karantzalos, ISPRS
  J. Photogramm. Remote Sens. 2024); CC-BY-4.0
- **URL**: https://doi.org/10.5281/zenodo.10664073
- **Access**: public Zenodo download, no credentials — `MADOS.zip` (4.0 GB) downloaded
  successfully to
  `raw/mados_marine_debris_and_oil_spill/MADOS.zip`. Access was **not** the problem.

## What MADOS is

Successor to MARIDA. Manually photo-interpreted (sparse) Sentinel-2 pixel annotations of
marine pollutants and sea-surface features. The release is structured as 174 scene
folders (`Scene_0` … `Scene_173`), each a unique S2 acquisition, split into 240×240
crops (`_1`, `_2`, … per scene) — 2803 crops total (train 1433 / val 642 / test 728).
Each crop's `10/` folder holds Rayleigh-corrected reflectance bands (`_rhorc_492/559/665/833`),
a class mask `_cl_N.tif` (uint8; 0 = unlabeled, 1–15 = the 15 classes), a confidence mask
`_conf_N.tif` (1 High / 2 Moderate / 3 Low), a report mask `_rep_N.tif`, an RGB png, and
ACOLITE turbidity products; `20/` and `60/` hold the coarser bands.

Class scheme (from the project repo `utils/assets.py`, `mados_cat_mapping`) — 15 classes,
source ids 1–15, 0 = unlabeled (would remap 1–15 → output 0–14, unlabeled → 255 nodata):

1 Marine Debris · 2 Dense Sargassum · 3 Sparse Floating Algae · 4 Natural Organic Material ·
5 Ship · 6 Oil Spill · 7 Marine Water · 8 Sediment-Laden Water · 9 Foam · 10 Turbid Water ·
11 Shallow Water · 12 Waves & Wakes · 13 Oil Platform · 14 Jellyfish · 15 Sea snot.

The class set and pixel labels are a good fit for open-set segmentation at 10 m (this is
MARIDA's successor and adds discrete oil-spill and sediment-plume masks). The problem is
purely georeferencing.

## Why rejected — no recoverable geocoordinates

The public MADOS release strips georeferencing from every crop:

- **Verified all 2803 `10/*_cl_*.tif` class rasters**: every one has an **identity
  geotransform** `(1,0,0, 0,1,0)` and no CRS/GCPs (rasterio reports a default `EPSG:4326`
  with a `NotGeoreferenced` warning). Band rasters (`_rhorc_*`) are the same. Bounds are
  pixel bounds `(0,0,240,240)`, not map coordinates.
- **No sidecar metadata**: the zip contains no `.xml`/`.tfw`/`.wld`/`.aux`/`.csv`/`.json`
  files (only `splits/{train,val,test}_X.txt`, which list `Scene_i_j` patch ids and carry
  no coordinates).
- **No scene→location crosswalk**: scenes are named `Scene_0`…`Scene_173` with a crop
  index only — no MGRS/UTM tile id, no lon/lat, no S2 product id. The project repo
  (`gkakogeorgiou/mados`) and README provide none; the training/extraction code
  (`utils/spectral_extraction.py`, `utils/dataset.py`) consumes the crops purely as arrays
  (mmsegmentation `NonGeo`-style), never as georeferenced rasters.

Because a crop cannot be tied to a lon/lat (or even an S2 tile + within-tile pixel offset),
the labels cannot be co-located with pretraining imagery on the S2 grid. Per the task spec
(§2 / §8.2 "No recoverable geocoordinates"), this is a fundamental, non-transient reject —
not a credential or infra issue (the data downloaded fine).

This contrasts with its sibling **MARIDA** (`marida_marine_debris_archive`, completed),
whose 256×256 patches ship georeferenced in local UTM at 10 m and processed normally.

## What would unblock it (retry conditions)

- A georeferenced MADOS release (crops carrying UTM/CRS + geotransform), **or**
- A published `Scene_i` → (S2 product id / MGRS tile + crop pixel-offset) crosswalk from
  which coordinates could be reconstructed.

If either appears, the processing recipe mirrors MARIDA exactly: crop each 240×240 `_cl`
raster into ≤64×64 UTM 10 m tiles, keep tiles with ≥1 labeled pixel, remap ids 1–15 → 0–14
(unlabeled 0 → 255 nodata), tiles-per-class balanced under the 25k cap, 1-day time_range
per S2 acquisition (transient sea-surface features). The raw zip is retained at
`raw/mados_marine_debris_and_oil_spill/MADOS.zip` to enable such a retry.

## Reproduce (the triage)

```
# download (already done): download_zenodo('10664073', raw_dir)  -> MADOS.zip
python3 - <<'PY'
import zipfile, rasterio, warnings; warnings.filterwarnings('ignore')
z=zipfile.ZipFile('/weka/dfive-default/helios/dataset_creation/open_set_segmentation/'
                  'raw/mados_marine_debris_and_oil_spill/MADOS.zip')
cls=[n for n in z.namelist() if '/10/' in n and '_cl_' in n and n.endswith('.tif')]
bad=sum(1 for p in cls if tuple(rasterio.open(f'/vsizip/{z.filename}/{p}').transform)[:6]
        ==(1,0,0,0,1,0))
print(len(cls),'cl tifs;',bad,'with identity (no) geotransform')  # -> 2803 2803
PY
```
