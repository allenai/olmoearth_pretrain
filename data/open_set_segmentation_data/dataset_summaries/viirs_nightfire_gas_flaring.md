# VIIRS Nightfire Gas Flaring

- **Slug:** `viirs_nightfire_gas_flaring`
- **Status:** completed
- **Task type:** classification (single positive class, sparse points)
- **Num samples:** 25,000 points
- **Output:** `datasets/viirs_nightfire_gas_flaring/points.geojson` (+ `metadata.json`)

## Source

NASA ORNL DAAC — *Global Gas Flare Survey by Infrared Imaging, VIIRS Nightfire,
2012-2019* (Elvidge, C. D., & Zhizhin, M., 2021; produced by the Earth Observation Group,
Colorado School of Mines). DOI: https://doi.org/10.3334/ORNLDAAC/1874.

Annual global surveys of gas-flaring sites. Flares were identified from heat anomalies
detected by the VIIRS Nightfire (VNF) algorithm on Suomi-NPP; high-temperature biomass
burning was separated from lower-temperature persistent gas flaring by temperature and
persistence. Each annual `eog_global_flare_survey_{year}_flare_list.csv` gives one row per
flare site with: `cntry_name`/`cntry_iso`, `catalog_id`, `id_number`, `latitude`,
`longitude`, `flr_volume` (billion m³/yr), `avg_temp` (K), `ellip`, `dtc_freq`, `clr_obs`,
`flr_type` (upstream / refinery / gas / lng / gpp / opp / midstream / ...).

## Access method

ORNL DAAC `/daacdata/` files are gated behind NASA Earthdata (URS OAuth). Credentials were
sourced from `.env` (`NASA_EARTHDATA_USERNAME`/`_PASSWORD`) and
written to `~/.netrc` (`machine urs.earthdata.nasa.gov`, chmod 600); downloads use
`download.download_earthdata()` (a `requests.Session` that follows the URS redirect chain).
Downloaded to `raw/viirs_nightfire_gas_flaring/`: the 2016-2019 `*_flare_list.csv` files
and the `Methane_Flaring_Sites_VIIRS.pdf` user guide. Only the label CSVs are needed
(pretraining supplies its own imagery); the KMZ and country-summary CSVs were skipped.

## Label / class mapping

Positive-only, **single class**: `0 = "gas flare"`. There is no background/negative class —
every record marks the presence of a flare. Per spec §5, no synthetic negatives are
fabricated; the pretraining-assembly step supplies negatives from other datasets.

Because each label is a single 10 m pixel, this is a **point dataset** and is written as
ONE `points.geojson` FeatureCollection (spec §2a), not per-point GeoTIFFs. Per-site
temperature, flared-gas volume, flare type, and country are carried as auxiliary feature
`properties` (`avg_temp_k`, `flr_volume_bcm`, `flr_type`, `country_iso`, `detection_year`)
— informative provenance, **not** the classification label.

## Time-range handling

Annual catalog → each point gets a **1-year** `time_range` anchored on its detection year
(`[Jan 1 year, Jan 1 year+1)`), per spec §5 (seasonal/annual labels). `change_time` is
`null` — a flare is a persistent presence/state label, not a dated change event.

**Sentinel-era filter:** ORNL DAAC 1874 spans 2012-2019; per spec §8 we keep only
**2016-2019** and drop 2012-2015 (pre-Sentinel-2). (Newer EOG catalogs extend to ~2021+ on
eogdata.mines.edu, but the authoritative DOI product ends at 2019.)

## Sampling

Pooled valid 2016-2019 records: **31,358** (0 rejected for bad coordinates). This exceeds
the hard 25k per-dataset cap (`sampling.MAX_SAMPLES_PER_DATASET`), so we take a seeded
sample of **25,000** via `balance_by_class(recs, "label", per_class=25000)` (single class ⇒
the class is capped at the 25k total). Sampling is deterministic (seed 42) and reproducible.

Same physical flare in multiple years is kept as **distinct** samples (different years ⇒
different 1-year windows / imagery), so no cross-year deduplication is done.

### Distribution of the 25,000 selected points

- By year: 2016=8,530 · 2017=1,570 · 2018=8,486 · 2019=6,414. (2017's source catalog is
  much smaller, ~2k rows, so it contributes fewer points.)
- Top countries: USA 8,136 · RUS 3,417 · CAN 1,308 · CHN 1,285 · IRN 829 · SAU 609 ·
  DZA 585 · IRQ 504 · NGA 488 · IDN 483 · VEN 425 · ARG 402 — matches known global gas-
  flaring hotspots (Permian/Bakken, W. Siberia, Persian Gulf, Niger Delta), a good
  geographic plausibility check.
- `flr_type`: upstream 22,626 · refinery 1,777 · gas 300 · lng 149 · gpp 84 · others.
- `avg_temp_k`: 936–2463 K. `flr_volume_bcm`: 0.0–1.335 (billion m³/yr).

## Caveats

- **Flare-location precision.** VNF detections come from ~750 m VIIRS pixels; each
  catalog point is the centroid of a persistent detection, so true precision is roughly
  sub-pixel to a few hundred metres, not metre-accurate. Pretraining snaps each point to a
  single 10 m S2 pixel; a flare (and its plume/infrastructure) typically spans several 10 m
  pixels, so the point should still land on/near the flaring facility. Downstream detection
  encoding (if used) already applies a ≥10 px ignore buffer that absorbs this uncertainty.
- **Sensor-modality mismatch (spatial sanity).** Flares are detected at *night* in
  SWIR/NIR IR; a daytime optical Sentinel-2 overlay does not directly show the flame, so a
  literal "does the label sit on the phenomenon" S2 eyeball is not meaningful here. Instead,
  the geographic distribution over known flaring basins was used as the plausibility check
  (see above). At 10 m S2/Landsat the flaring **infrastructure** (well pads, stacks, gas
  plants) is generally visible even if the flame itself is not.
- **Positive-only.** No background class by construction (spec §5); negatives added at
  assembly time.
- **2017 catalog** is unusually small in the source (~2k rows vs ~8-10k other years); this
  is a property of the source release, not a truncated download (files verified complete).

## Reproduce

```bash
# .env creds -> ~/.netrc (machine urs.earthdata.nasa.gov, chmod 600) done by SOP
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.viirs_nightfire_gas_flaring
```

The script is idempotent: raw downloads skip if present; re-running regenerates
`points.geojson` / `metadata.json` deterministically (seed 42).
