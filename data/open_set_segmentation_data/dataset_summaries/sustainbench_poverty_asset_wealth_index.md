# SustainBench Poverty (Asset Wealth Index)

- **Slug:** `sustainbench_poverty_asset_wealth_index`
- **Task type:** regression (point table)
- **Status:** completed
- **Num samples:** 5000
- **Regression target:** `asset_wealth_index`

## Source & access

SustainBench (Stanford Sustainability & AI Lab), the DHS survey-derived poverty
benchmark. Docs:
https://sustainlab-group.github.io/sustainbench/docs/datasets/sdg1/change_in_poverty.html

Cluster-level labels are published as `dhs_final_labels.csv` in the SustainBench poverty
Google Drive folder `1tzWDfd4Y5MvJnJb-lHieOuD-aVcUqzcu` (file id
`16OORDhlm5OufImAIRGRNW0kZc3rowrks`, ~18 MB), downloaded with `gdown`. **No DHS
credential needed:** the raw DHS household microdata is registration-gated, but the
aggregated cluster-mean wealth index + centroid lat/lon are the public SustainBench label,
which is what we use.

`dhs_final_labels.csv` columns used: `lat`, `lon` (cluster centroid, WGS84), `asset_index`
(regression target), `year`, `cname` (country), `DHSID_EA` (provenance id). Full file:
86,936 clusters with a valid asset index over 1996-2019 across 56 country-survey codes.

## Label mapping

`label = asset_index` = cluster-mean asset wealth index: a scalar computed per household by
PCA over asset-ownership / housing-quality variables, then averaged over the households in
a DHS survey cluster (enumeration area). Higher = wealthier; standardized/dimensionless.
Written as a **point-table regression** dataset (`points.json`, spec 2a) — each label is a
single-point continuous value, so no per-point GeoTIFFs. nodata sentinel `-99999`.

## Sampling & time range

- Restricted to survey **year >= 2016** (Sentinel-2 era, and the manifest's `[2016, 2019]`
  window): 14,407 clusters across 27 countries.
- **Randomly sampled 5000** (seed 42) from that pool — the regression cap. The
  asset-index distribution over the >=2016 pool is roughly symmetric (not strongly
  skewed), so a plain random sample is used rather than bucket balancing. It preserves the
  natural distribution.
- **Time range:** 1-year window anchored on the survey year, `[year-01-01, (year+1)-01-01)`
  (via `io.year_range`). All source splits used.

## Value distribution (5000 selected)

min -3.79, max 3.48, mean -0.11, std 1.76. Histogram (10 bins):

```
-3.79..-3.06: 118
-3.06..-2.33: 445
-2.33..-1.61: 665
-1.61..-0.88: 705
-0.88..-0.15: 587
-0.15.. 0.57: 554
 0.57.. 1.30: 621
 1.30.. 2.03: 628
 2.03.. 2.75: 417
 2.75.. 3.48: 260
```

## Caveats

- DHS cluster coordinates are privacy-jittered (up to 2 km urban, ~5-10 km rural), so the
  point does not mark an exact spot; the label is a neighborhood-scale wealth aggregate.
  Because the label is a scalar cluster value (not a spatial mask), an exact pixel overlay
  sanity check is not applicable; sampled coordinates were confirmed to fall within
  plausible country extents (e.g. Madagascar cluster at lat -18.97, lon 47.33).
- Pre-2016 clusters (the majority of the full file, incl. a large 2015 survey batch) are
  excluded to keep every label inside the Sentinel-2 era.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.sustainbench_poverty_asset_wealth_index
```
Idempotent: skips re-download if `raw/{slug}/dhs_final_labels.csv` exists; rewrites
`points.json` / `metadata.json`.
