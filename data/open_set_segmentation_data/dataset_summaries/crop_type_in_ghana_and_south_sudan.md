# Crop Type in Ghana and South Sudan — REJECTED

- **Slug**: `crop_type_in_ghana_and_south_sudan`
- **Manifest name**: Crop Type in Ghana and South Sudan
- **Source**: Source Cooperative — `stanford/africa-crops-ghana`
  (https://source.coop/stanford/africa-crops-ghana; formerly Radiant MLHub
  `su_african_crops_ghana`, DOI 10.34911/rdnt.ry138p)
- **Family / label_type**: crop_type / polygons (delivered as per-pixel raster chips)
- **License**: README states CC-BY-SA-4.0 (manifest says CC-BY-4.0); not the blocker.
- **Final status**: **rejected** — reason: **no recoverable geocoordinates**
- **task_type intended**: classification (per-pixel crop type)

## What the source actually is

First smallholder crop-type semantic-segmentation dataset for Africa (Rustowicz et al.,
CVPR-W 2019). Delivered as 4,040 Ghana chips, each a 64×64 tile with:
- `truth/truth_ghana_{id}/truth_ghana_{id}_label.tif` — per-pixel crop-type label,
- Sentinel-1 (`s1/`), Sentinel-2 (`s2/`), PlanetScope (`planet/`) image time series,
- `original/*_npy/` — the same arrays as `.npy`.

The label legend (documentation.pdf, Appendix D) is a real crop taxonomy:
0=unknown, 1=ground nut, 2=maize, 3=rice, 4=soya bean, 5=yam, 6=intercrop, 7=sorghum,
8=okra, 9=cassava, 10=millet, 11=tomato, 12=cowpea, 13=sweet potato, 14=babala beans,
15=salad vegetables, 16=bra and ayoyo, 17=watermelon, 18=zabla, 19=nili, 20=kpalika,
21=cotton, 22=akata, 23=nyenabe, 24=pepper. (Manifest's 5-class list is a subset.)
This is exactly the kind of per-pixel crop label the effort wants — the label *semantics*
are fine. It is the **geolocation** that is unrecoverable.

## Why rejected (evidence)

The publishers deliberately anonymized every tile's location, and the mirror carries **no**
coordinate metadata. Concretely:

1. **All GeoTIFFs are stripped of georeferencing.** Downloaded and opened 5 label tifs
   (ids 000000, 000001, 000100, 002000, 004039) and an S2 `..._data.tif` with rasterio:
   every one reports `crs = None`, an identity transform, `res = (1.0, 1.0)`, and bounds
   `(0, 0, 64, 64)` (pixel space). No UTM/WGS84 georeferencing exists to place a tile on
   the S2 grid.
2. **Locations were intentionally anonymized.** documentation.pdf Appendix F/G states:
   *"Each tile is identified with a unique 6-digit integer identifier, which has been
   **randomized in order to preserve location anonymity**"* and *"Up to **3 km of random
   jitter** have been added to locations for privacy."* 3 km ≈ 300 pixels at 10 m — far
   larger than a whole 640 m tile — so even an approximate recovered centroid could not
   correctly co-locate a 64×64 label with pretraining imagery.
3. **The bundled imagery cannot be used as a georeference anchor either.** The same
   Appendix notes *"A small amount of random noise was added to all satellite imagery,"*
   and those image tifs are likewise `crs = None` — so pixel-matching the dataset's own S2
   against a georeferenced S2 mosaic to recover coordinates is not viable.
4. **No sidecar coordinate/STAC files in the repo.** Repo contents are `README.md`,
   `documentation.pdf`, `.source/metadata.json` (695 B, description+tags only), and the
   `truth/ s1/ s2/ planet/ original/` chip directories — only `.tif` and `.npy` files.
   There is no GeoJSON, STAC item geometry, bounding-box table, or lon/lat CSV. (The
   companion South Sudan repo `stanford/africa-crops-south-sudan` does not exist on Source
   Cooperative — `NoSuchBucket` — so no alternate georeferenced copy there.)

Per AGENT_SUMMARY §8.2 this is the canonical "no recoverable geocoordinates" rejection:
an ML-ready tensor/chip release that strips lon/lat, so labels cannot be placed on the S2
grid. A per-tile randomized id (with added jitter) is explicitly not sufficient.

## Access notes (for anyone revisiting)

- Data is public/unauthenticated at S3-compatible endpoint `https://data.source.coop`,
  bucket `stanford`, prefix `africa-crops-ghana/` (boto3 UNSIGNED works; the old
  `us-west-2.opendata.source.coop` bucket path is empty — use `data.source.coop`).
- Only a **new source with real georeferencing** (e.g. the original field-polygon
  shapefiles / un-jittered coordinates from the authors) could rescue this dataset. That
  would be new information, not a transient failure — hence `rejected`, not
  `temporary_failure`.

## Reproduce the triage

```python
import boto3, botocore, rasterio
s3 = boto3.client('s3', endpoint_url='https://data.source.coop',
                  config=botocore.config.Config(signature_version=botocore.UNSIGNED))
s3.download_file('stanford',
    'africa-crops-ghana/truth/truth_ghana_000000/truth_ghana_000000_label.tif', 'l.tif')
with rasterio.open('l.tif') as ds:
    print(ds.crs, ds.transform, ds.bounds)   # -> None, identity, (0,0,64,64)
```
