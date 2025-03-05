# Worldcover sampling

### 1. Create features from WorldCover

This requires running `compute_worldcover_histograms.py`. The [`Dockerfile`](Dockerfile) and [`launch_beaker.py`](launch_beaker.py) allow this to be easily launched from Beaker.

This script will split the globe (according to WorldCover) into `TILE_SIZE x TILE_SIZE` pixels (in the script, `TILE_SIZE` is `100`, which corresponds to 1km.
Beaker jobs for each batch of WorldCover GeoTIFFs are launched in parallel.

```
cd helios/scripts/worldcover_sampling
docker build -t helios-worldcover-sampling .
beaker image create --name helios-worldcover-sampling helios-worldcover-sampling
python launch_beaker.py --out_dir /weka/dfive-default/helios/dataset_creation/worldcover_histogram_csvs/ --beaker_image favyen/helios-worldcover-sampling
```

Note that the output directory must be on a shared filesystem.

### 2. Use those features to train a
Once the file is created, we use a k-means clustering algorithm to find the centroids of k clusters, where k corresponds to the number of points we want to export.

Two strategies are adopted - one where we sample 50 points per tile (this is saved in `"esa_grid_subsampled.csv"`) and one where we take k points globally (this is saved in `"esa_grid_subsampled_global.csv"` and takes far longer to run).

```
# Concatenate the CSVs.
cd /weka/dfive-default/helios/dataset_creation/worldcover_histogram_csvs/
{ head -n 1 N00E006.csv; for f in *.csv; do tail -n +2 "$f"; done } | cat > ../worldcover_histogram_csvs_concat.csv
# Run the K-means.
cd /path/to/helios/scripts/worldcover_sampling
python worldcover_kmeans.py --csv_fname /weka/dfive-default/helios/dataset_creation/worldcover_histogram_csvs_concat.csv --subsampled_grid_path /weka/dfive-default/helios/dataset_creation/esa_grid_subsampled.csv --subsampled_global_grid_path /weka/dfive-default/helios/dataset_creation/esa_grid_subsampled_global.csv
```
