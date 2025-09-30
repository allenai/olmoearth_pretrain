# Compute Min/Max Statistics for Eval Datasets

## Overview

`compute_eval_dataset_minmax.py` is a standalone script that computes the minimum and maximum values per band for each modality across all non-geobench evaluation datasets.

## What are Non-Geobench Datasets?

Non-geobench datasets are evaluation datasets that are **NOT** prefixed with `m_` or `m-`. These include:

1. **mados** - Multi-temporal Agriculture Dataset for Operational Monitoring
   - Modality: Sentinel-2 L2A (13 bands)

2. **sen1floods11** - Sentinel-1 Floods Dataset
   - Modality: Sentinel-1 (2 bands: VV, VH)

3. **pastis** - Panoptic Agricultural Satellite Time Series
   - Modalities: Sentinel-2 L2A (13 bands), Sentinel-1 (2 bands)

4. **pastis128** - PASTIS at 128x128 resolution
   - Modalities: Sentinel-2 L2A (13 bands), Sentinel-1 (2 bands)

5. **breizhcrops** - Crop Type Classification Dataset
   - Modality: Sentinel-2 L2A (13 bands)

6. **sickle** - Multi-sensor Crop Field Delineation
   - Modalities: Sentinel-2 L2A (13 bands), Sentinel-1 (2 bands), Landsat-8 (11 bands)

7. **cropharvest** - Global Cropland Mapping Dataset
   - Modalities: Sentinel-2 L2A (9 bands, partial), Sentinel-1 (2 bands), SRTM (1 band)

## What Does the Script Do?

The script:
1. Loads the **training split** for each non-geobench dataset
2. Reads all raw (un-normalized) image data
3. Computes the **minimum** and **maximum** value for each band in each modality
4. Handles NaN and infinite values appropriately
5. Saves the results to a JSON configuration file

## Output Format

The script generates a JSON file with the following structure:

```json
{
  "dataset_name": {
    "modality_name": {
      "band_name": {
        "min": <float>,
        "max": <float>
      }
    }
  }
}
```

Example:
```json
{
  "mados": {
    "sentinel2_l2a": {
      "01 - Coastal aerosol": {"min": 0.0, "max": 10000.0},
      "02 - Blue": {"min": 0.0, "max": 10000.0},
      ...
    }
  },
  "sen1floods11": {
    "sentinel1": {
      "vv": {"min": -50.0, "max": 10.0},
      "vh": {"min": -50.0, "max": 10.0}
    }
  },
  ...
}
```

## How to Run

### Prerequisites

The script requires access to the evaluation datasets stored on the cluster:
- MADOS: `/weka/dfive-default/presto_eval_sets/mados`
- Sen1Floods11: `/weka/dfive-default/presto_eval_sets/floods`
- PASTIS: `/weka/dfive-default/presto_eval_sets/pastis_r`
- PASTIS128: `/weka/dfive-default/presto_eval_sets/pastis_r_origsize`
- BreizhCrops: `/weka/dfive-default/skylight/presto_eval_sets/breizhcrops`
- SICKLE: `/weka/dfive-default/presto_eval_sets/sickle`
- CropHarvest: `/weka/dfive-default/presto_eval_sets/cropharvest`

### Running the Script

**Full run (process all samples):**
```bash
cd /path/to/helios
python scripts/compute_eval_dataset_minmax.py
```

**Smoke test (process only 10 samples per dataset):**
```bash
python scripts/compute_eval_dataset_minmax.py --max-samples 10
```

**Custom output location:**
```bash
python scripts/compute_eval_dataset_minmax.py --output /path/to/output.json
```

**CLI Options:**
- `--max-samples N`: Limit to N samples per dataset (for smoke testing)
- `--output PATH`: Custom output file path

### Output Location

The script saves the results to:
```
helios/evals/datasets/minmax_stats.json
```

## Implementation Details

### Data Loading Strategy

- **MADOS**: Loads entire training tensor at once (N, H, W, C)
- **Sen1Floods11**: Loads entire training tensor at once (N, C, H, W)
- **PASTIS/PASTIS128**: Iterates through individual .pt files for S2 and S1
- **BreizhCrops**: Uses the BreizhCrops library to load samples from regions
- **SICKLE**: Iterates through individual .pt files for S2, S1, and L8
- **CropHarvest**: Loads arrays from multiple country-specific datasets

### Handling Special Cases

1. **NaN/Inf Values**: Filtered out before computing min/max
2. **Multiple Countries (CropHarvest)**: Aggregates across all countries
3. **Time Series**: Computes min/max across all timesteps
4. **Missing Bands**: Only computes stats for available bands

## Error Handling

The script includes comprehensive error handling:
- Each dataset is processed in a try-except block
- Errors are printed with full traceback
- Failed datasets don't prevent other datasets from being processed
- Final summary shows which datasets were successfully processed

## Performance Notes

- The script may take significant time to run (hours) depending on dataset sizes
- Progress bars are shown for datasets that iterate through many files
- Memory usage should be reasonable as most datasets are processed iteratively

## Use Cases

This min/max configuration can be used for:
1. **Data normalization**: Understanding the value ranges for proper scaling
2. **Outlier detection**: Identifying unusual values in the datasets
3. **Model initialization**: Setting appropriate value ranges for model inputs
4. **Data validation**: Ensuring new data falls within expected ranges
5. **Visualization**: Setting appropriate color scales for imagery

## Modifying the Script

To add a new dataset:
1. Add the dataset path to the imports
2. Create a `compute_<dataset>_stats()` function
3. Add a try-except block in `main()` to call your function
4. Ensure the function returns a dict with the correct structure

## Notes

- The script only processes **training splits** as specified in the requirements
- All statistics are computed on **raw (un-normalized)** data
- Band names follow the Helios convention (e.g., "01 - Coastal aerosol")
- Geobench datasets (prefixed with m- or m_) are explicitly excluded
