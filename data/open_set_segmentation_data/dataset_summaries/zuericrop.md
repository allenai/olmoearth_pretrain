# ZueriCrop — REJECTED (no recoverable geocoordinates)

- **slug:** `zuericrop`
- **manifest name:** ZueriCrop
- **family / label_type:** crop_type / polygons (per manifest)
- **region / year:** Zurich & Thurgau, Switzerland / 2019
- **source:** GitHub `0zgur0/ms-convSTAR` (a.k.a. `multi-stage-convSTAR-network`), ISPRS/RSE
  Turkoglu et al. 2021 (https://doi.org/10.1016/j.rse.2021.112603)
- **Final status:** `rejected`
- **Reason:** No recoverable geocoordinates (AGENT_SUMMARY §8.2 / §2 triage rejection).

## What the dataset actually is

The manifest lists ZueriCrop as `label_type: polygons` (~116k farmer-declared field
instances, 48-class hierarchical crop taxonomy). However, the dataset is **not** released
as georeferenced polygons. It is distributed only as an **anonymized "ML-ready" HDF5 of
24×24 patches** — exactly the coordinate-free tensor-release case §8.2 warns about.

The single distributed file `ZueriCrop.hdf5` (41.3 GB) contains three arrays and nothing
else:

| key          | shape                        | dtype | meaning                                   |
|--------------|------------------------------|-------|-------------------------------------------|
| `data`       | (27977, 142, 24, 24, 9)      | int16 | Sentinel-2 L2A time series (9 bands)      |
| `gt`         | (27977, 24, 24, 1)           | int16 | per-pixel semantic crop label             |
| `gt_instance`| (27977, 24, 24, 1)           | int32 | per-pixel instance id (within-patch only) |

File-level HDF5 attrs are **empty** (`{}`). There is **no** longitude, latitude, CRS,
bounding box, UTM offset, MGRS tile id, or patch grid index anywhere in the file. The
companion `labels.csv` is only the crop-type taxonomy (GT id → German/English names +
4-tier hierarchy); it carries no coordinates. The HF mirror repo
(`isaaccorley/zuericrop`) contains just `.gitattributes`, `README.md`, `ZueriCrop.hdf5`,
`labels.csv` — no coordinate sidecar.

The 27977 patches are provided in an opaque order with no spatial index, so there is no
way to reassemble the mosaic or place any patch on the Sentinel-2 grid. `gt_instance` ids
are local to each patch and do not link to the original field polygons' geometry.
`torchgeo` correspondingly implements ZueriCrop as a `NonGeoDataset` (no CRS/bounds).

Because labels cannot be co-located with pretraining imagery by geography, the dataset
fails the hard "No recoverable geocoordinates" rejection criterion.

## How it was checked (cheap-first, per §8.2)

No 41 GB download was performed. The HDF5 **header only** was read remotely from the HF
mirror via `fsspec` + `h5py` (1 MB block size), which returned the top-level keys, array
shapes/dtypes, and empty attrs above. Cross-checked against the upstream `ms-convSTAR`
`dataset.py` loader (reads only `data`/`cloud_cover`/`gt`/`gt_instance`) and torchgeo's
`ZueriCrop(NonGeoDataset)` loader (downloads only `ZueriCrop.hdf5` + `labels.csv`).

## Reproduce the check

```python
import fsspec, h5py
url = ('https://hf.co/datasets/isaaccorley/zuericrop/resolve/'
       '8ac0f416fbaab032d8670cc55f984b9f079e86b2/ZueriCrop.hdf5')
with fsspec.open(url, block_size=1024*1024) as fo:
    h = h5py.File(fo, 'r')
    print(list(h.keys()))          # ['data', 'gt', 'gt_instance']
    print(dict(h.attrs))           # {}  -> no georeferencing
```

## Caveats / possible future path

The underlying source (Swiss cantonal crop declarations over a 50×48 km Zurich/Thurgau
scene) is inherently georeferenced. If the authors' original georeferenced polygon/raster
export (with CRS + geometry, not the anonymized patch tensor) can be obtained out of band,
this dataset would be a good `polygons` crop-type fit and could be reprocessed. Until such
a coordinate-bearing source is provided, it is rejected. This is a permanent-drop
`rejected` (not `temporary_failure`): the blocker is the coordinate-free release format,
not a transient source/infra error.
