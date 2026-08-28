# SICKLE

- **slug**: `sickle`
- **status**: **rejected** — no recoverable geocoordinates (plot coordinates deliberately
  withheld for privacy; imagery released as coordinate-free tensor stacks). Fundamental,
  not a retry candidate. Secondary: access is gated behind a manual-approval registration
  form (`needs-credential`), but the georeferencing failure is primary and moot the gate.
- **task_type** (intended, had it been usable): classification (crop type, 21 classes) via
  rasterized parcel polygons → dense_raster/polygons.
- **num_samples**: 0

## Source

- Manifest name: `SICKLE` (family `crop_type`, label_type `polygons`, region Tamil Nadu,
  India, have_locally: false, license "open (research)").
- Paper: Sani et al., "SICKLE: A Multi-Sensor Satellite Imagery Dataset Annotated with
  Multiple Key Cropping Parameters", WACV 2024 (oral). arXiv:2312.00069.
- Repo: <https://github.com/Depanshu-Sani/SICKLE>; site
  <https://sites.google.com/iiitd.ac.in/sickle/home>.
- Content: multi-sensor (Sentinel-1, Sentinel-2, Landsat-8) time series over the Cauvery
  Delta, Tamil Nadu; 2,370 season-wise samples from 388 surveyed plots (avg 0.38 acre),
  ~209k images, Jan 2018 – Mar 2021. Annotated with crop type (21 classes), phenology
  (sowing/transplanting/harvesting dates), variety, and yield. Multi-resolution masks at
  3 m / 10 m / 30 m.

## What the released data actually is

From the official data loader (`utils/dataset.py`) and demo notebook:

- **Imagery**: per-sample, per-satellite **`.npz` numpy stacks** under
  `images/{satellite}/npy/{uid}/*.npz`, loaded with `np.load` by band name. Sample tensors
  are fixed-size (e.g. S1 `(T=23, bands=2, 32, 32)`) — **coordinate-free tensor stacks**
  with no CRS, geotransform, or lon/lat.
- **Masks**: `masks/{res}m/{uid}.tif` read with `rasterio` — but the loader reads **only
  pixel values** (`fp.read()`), a 6-layer stack: plot mask, crop type, sowing/transplanting/
  harvesting day, and yield. No spatial metadata is consumed.
- **Tabular**: `sickle_dataset_tabular.csv` with `UNIQUE_ID`, `PLOT_ID`,
  `STANDARD_SEASON`, `YEAR`, `SOWING_DAY`, `HARVESTING_DAY`, and train/val/test split — **no
  coordinate columns**.

## Why rejected — georeferencing (SOP §8.2)

The task explicitly instructed a georeferencing check: "if released as coordinate-free
tensor stacks, reject." Both the imagery release and the authors' stated policy fail it:

1. **Coordinates deliberately withheld for privacy.** The paper/authors state the "specific
   coordinates of each plot are withheld for privacy reasons." The parcels were digitized in
   QGIS around field GPS points, but those locations are **intentionally not distributed**.
   For a dataset whose whole privacy premise is hiding plot locations, the released 32×32
   tiles cannot be placed on the Sentinel-2 grid by geography — precisely the
   "no recoverable geocoordinates" case in §8.2. A per-plot `uid` alone (analogous to an
   MGRS-tile id without within-tile pixel index) is not a sufficient geolocation.
2. **Imagery is coordinate-free `.npz`.** Even setting privacy aside, the primary sample
   arrays carry no CRS/transform. While the `.tif` masks are nominally GeoTIFFs, retaining
   true CRS+transform in them would directly contradict the privacy-withholding claim, so
   they cannot be relied on to yield real coordinates; the loader itself never reads any.

Because the labels cannot be co-located with pretraining imagery by geography, this is a
**fundamental `rejected`** (§8.2 "no recoverable geocoordinates"), not `temporary_failure`.

## Secondary note — access gate (needs-credential)

Access to the full and toy datasets is only via a **manual-approval Google Form**
(<https://docs.google.com/forms/d/e/1FAIpQLSdq7Dcj5FF1VmlKozrQ7XNoq006iVKrUIMTK2jReBJDuO1N2g/viewform>).
No open direct download, Zenodo, or mirror was found (checked GitHub, the project site
home + `/download` page, and web search on 2026-07-11). This alone would warrant
`needs-credential`, but it is secondary: even with access granted, the deliberate
coordinate-withholding makes the labels ungeoreferenceable, so the rejection stands.

## Judgment calls

- **Reject on georeferencing (fundamental), not `temporary_failure`.** The source site is
  reachable; the block is not a transient outage. Obtaining the data would not make the
  labels geolocatable.
- Chose `rejected` over pure `needs-credential` framing because acquiring credentials would
  not resolve the core defect (withheld coordinates); noted the credential gate as
  secondary so the user can still weigh contacting the authors for coordinate-bearing data.
- If the authors could later provide the original **georeferenced parcel vector files**
  (the QGIS polygons with true CRS), this dataset would be a good crop-type polygons →
  dense_raster fit (21 classes, 2018–2021 seasonal → 1-year windows) and should be
  reconsidered. Absent those coordinates it is unusable here.

## Reproduce

No outputs were written to weka `datasets/sickle/` beyond `registry_entry.json`. To revisit:
request access via the Google Form above (or email the authors,
depanshus@iiitd.ac.in / sourabh19113@iiitd.ac.in) **and** ask specifically for the
georeferenced plot polygon vector files with intact CRS; only then could the crop-type
parcels be rasterized to 10 m UTM label tiles per SOP §2/§4.
