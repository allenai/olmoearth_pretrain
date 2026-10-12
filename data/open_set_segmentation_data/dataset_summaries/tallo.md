# Tallo — REJECTED

- **Slug:** `tallo`
- **Status:** `rejected` (fundamental fit problem; not a credential/transient issue)
- **Source:** Tallo database (Jucker et al. 2022, *Global Change Biology*), Zenodo record
  [10.5281/zenodo.6637599](https://doi.org/10.5281/zenodo.6637599). Files used:
  `Tallo.csv`, `Tallo_metadata.csv`, `Tallo_references.csv` (label metadata table only; no
  imagery). License CC-BY-4.0.
- **What it is:** A global compilation of ~498,838 georeferenced individual-tree records
  (stem diameter, height, crown radius) across 5,163 species / 1,453 genera / 187 families,
  drawn from ~69 field-allometry and forest-inventory sources.

## Access

Fully accessible — public Zenodo download, no credential needed (checked
`.env`; none required). Downloaded successfully to
`raw/tallo/`. So this is **not** a `needs-credential` or `temporary_failure` rejection.

## Why rejected

Two decisive, compounding reasons, both established by downloading and analyzing the actual
table (not from the manifest text):

### 1. Pre-2016 rule — no usable post-2016 subset (primary)

The published `Tallo.csv` contains **no per-record measurement date**. Columns are:
`tree_id, division, family, genus, species, latitude, longitude, stem_diameter_cm,
height_m, crown_radius_m, height_outlier, crown_radius_outlier, reference_id`. The
manifest's note *"records are dated; filter to Sentinel-2 era"* and its `time_range
[2016, 2022]` do **not** hold for this release — that window reflects the Zenodo
publication era, not the field-measurement dates.

The only temporal signal is the **publication year of each record's reference**, which is
not a valid measurement-era filter: field allometry campaigns predate their publications,
frequently by years to decades. Even using publication year as a generous upper bound:

| reference publication year (proxy only) | records |
|---|---|
| < 2016 | 200,775 |
| ≥ 2016 | 196,841 |
| undatable | 101,222 |

Tallo is a well-known compilation of largely pre-2016 field measurements (many sources are
2001–2015 papers whose field data is older still). Because **no record can be confidently
placed in the post-2016 Sentinel era**, there is no usable post-2016 subset to keep. Per
the spec pre-2016 rule (reject when labels are not resolvable to the post-2016 era), the
dataset is rejected on this ground.

### 2. Georeferencing / observability at 10–30 m (compounding)

Coordinates are **plot-centroid, not individual-tree GPS**: only **61,856 unique lon/lat
points for 498,838 records** (mean 8.1 trees per point; a single point stacks 23,249
trees), rounded to ~0.001–1° (≈100 m to >10 km). Sources include **FIA** (~1-mile
coordinate fuzzing, 5,407 records) and **NEON** plots (42,775 records). Individual trees at
plot-rounded/fuzzed coordinates are not reliably observable or placeable on the 10 m
Sentinel grid — the spec explicitly lists "individual small trees" and "coordinate-fuzzed
points like FIA ~1 mi" as not observable at 10–30 m.

### Classification vs regression considered

- **Species classification** (top-254 by frequency, like `globalgeotree`): rejected —
  unlike GlobalGeoTree, Tallo has no observation year to filter to the Sentinel era, and
  its coordinates are plot-centroid rather than per-observation GPS.
- **Height / biomass regression**: rejected — a single tree's height at a plot-rounded
  coordinate is not a meaningful 10 m-pixel canopy-height/biomass target, and the same
  no-date problem applies.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.tallo
```

The script downloads the label CSVs, prints the diagnostics above, and writes the
per-dataset rejection entry `datasets/tallo/registry_entry.json`. Re-run only if a future
Tallo release adds per-record measurement dates **and** individual-tree GPS coordinates.
