## optional prelim steps for HDBSCAN / t-sne clustering
- only needed for visualization with t-sne or hdbscan, not for candidate scoring/selection
install cuml:
```shell
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12
```
<br/>

---
---
# SECTION A: K-MEANS clustering and visualization
---
---
<br/>
## Launch KMEANS clustering
notes:
1. points within a cluster are based of a 2D PCA projection of the residual vectors (point minus centroid)
2. The distances between points and centroids are consistent across clusters
3. The distance between clusters is not consistent with distance between points and centroids
4. MDS minimizes the stress (distortion) between the input high-D distances and the output 2D distances. With only 15 clusters mapped to 2D, MDS should achieve very low stress, meaning clusters close in cosine space will be close on the plot, and far clusters will be far. The relative positioning is faithful to the high-D inter-centroid geometry.
```shell
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
python scripts/embeddings/cluster_embeddings.py   --input-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes   --seed 1234   --pca-dim 128   kmeans   --k 15   --spherical
```


### Viz with cluster labels - simple
```shell
python scripts/embeddings/visualize_embeddings.py --input-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes --darkmode --seed 1234 --labels /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/labels_kmeans.npy
```


### Viz with cluster labels - Distance based
```shell
python scripts/embeddings/visualize_embeddings.py --layout cluster --cluster-bundle /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_cc_4s_s2/_cluster/cluster_bundle_kmeans.npz  --metric cosine  --separation 3  --darkmode  --metadata /weka/dfive-default/henryh/helios/olmoearth_pretrain/v0_1_osm_sampling_scores.parquet  --output-name embedding_map_metadata_test.html   --metadata-columns filename elevation_mean is_temperate is_subtropical is_tropical is_boreal_polar is_mountainous ndwi_mean patch_uniformity spatial_homogeneity canopy_height_mean
```


### Viz: globe map with cluster colors
```shell
python scripts/embeddings/visualize_map.py   --index-csv /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/index.csv   --labels-npy /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/labels_kmeans_int.npy   --lat-bounds 40 85 --lon-bounds -145 -50   --darkmode   --output-name globe_map_kmeans_nolegendtitle.png --no-legend --no-title
```

<br/>
<br/>

---
---
# SECTION B1: Click-to-view image thumbnails support
---
---
<br/>
Three steps: generate thumbnails, build the map with `--thumbnail-url-prefix`, serve via HTTP.

### Optional Step0 : Delete pre-existing thumbnails
```shell
mkdir -p /weka/dfive-default/empty_dir_for_rsync_wipe
rsync -av --delete --dry-run /weka/dfive-default/empty_dir_for_rsync_wipe/ /weka/dfive-default/helios/dataset/osm_sampling/thumbnails/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/
```

### Optional Step 1: Generate thumbnails from H5 files
```shell
# Generate thumbnails (adapt RGB_BANDS / S2_KEY in the script if needed):
python scripts/embeddings/generate_thumbnails.py   --h5-dir /weka/dfive-default/helios/dataset/osm_sampling/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldpop_wri_canopy_height_map/1138828   --output-dir /weka/dfive-default/helios/dataset/osm_sampling/thumbnails/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldpop_wri_canopy_height_map/   --metadata /weka/dfive-default/henryh/helios/olmoearth_pretrain/v0_1_osm_sampling_scores.parquet   --workers 8
```

### Step 2: Build the map with image support
```shell
python scripts/embeddings/visualize_embeddings.py   --layout cluster   --cluster-bundle /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/cluster_bundle_kmeans.npz   --metric cosine   --separation 3   --darkmode   --metadata /weka/dfive-default/henryh/helios/olmoearth_pretrain/v0_1_osm_sampling_scores.parquet   --metadata-columns filename elevation_mean is_temperate is_subtropical is_tropical is_boreal_polar is_mountainous ndwi_mean patch_uniformity spatial_homogeneity canopy_height_mean   --output-name embedding_map_with_images.html   --thumbnail-url-prefix ./thumbnails/
```

### Step 3: Serve and open in browser
```shell
export IMAGES_DIR="/weka/dfive-default/helios/dataset/osm_sampling/thumbnails/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/"
# Option A: convenience script (handles thumbnail dir symlinks)
python scripts/embeddings/serve_map.py   --viz-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/_viz   --thumbnail-dir "$IMAGES_DIR"

# Option B: plain Python HTTP server (thumbnails must be inside _viz/thumbnails/)
python -m http.server 8765 -d /path/to/_viz
```
Cursor/VS Code auto-forwards the port

<br/>
<br/>

---
---

# SECTION B2: NESTED HDBSCAN WITHIN K-MEANS CLUSTERS
---
---
<br/>
## Launch cluster - KMEANS + residual HDBSCAN
notes:
1. this keeps the existing K-means cluster bundle as the geometry used by `--layout cluster`
2. within each K-means cluster, residuals are defined as `(point - parent centroid)`
3. those residuals are reduced to `30D` PCA before applying HDBSCAN
4. nested labels are written in a compact form like `K0_H2`
5. the nested HDBSCAN result is saved as a separate labels file and does not change the current capabilities

```shell
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
python scripts/embeddings/cluster_embeddings.py \
  --input-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes \
  --seed 1234 \
  --pca-dim 128 \
  kmeans-hdbscan \
  --k 15 \
  --spherical \
  --residual-pca-dim 30 \
  --min-cluster-size 100 \
  --min-samples 30
```

This produces, in addition to the existing K-means outputs:
1. `labels_kmeans_residual_hdbscan.npy`
2. `labels_kmeans_residual_hdbscan_int.npy`

## Viz with cluster bundles + nested HDBSCAN labels
The point placement still comes from `cluster_bundle_kmeans.npz`.
The nested HDBSCAN labels only override displayed colors / search / polygons.

```shell
python scripts/embeddings/visualize_embeddings.py \
  --layout cluster \
  --cluster-bundle /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/cluster_bundle_kmeans.npz \
  --labels /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/labels_kmeans_residual_hdbscan.npy \
  --metric cosine \
  --separation 3 \
  --darkmode \
  --metadata /weka/dfive-default/henryh/helios/olmoearth_pretrain/v0_1_osm_sampling_scores.parquet \
  --output-name embedding_map_kmeans_residual_hdbscan.html \
  --metadata-columns filename elevation_mean is_temperate is_subtropical is_tropical is_boreal_polar is_mountainous ndwi_mean patch_uniformity spatial_homogeneity canopy_height_mean
```

## Viz with thumbnails + nested HDBSCAN labels
```shell
python olmoearth_pretrain/dataset_creation/candidate_embeddings/visualize_embeddings.py \
  --layout cluster \
  --cluster-bundle /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/cluster_bundle_kmeans.npz \
  --labels /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/labels_kmeans_residual_hdbscan.npy \
  --metric cosine \
  --separation 3 \
  --darkmode \
  --metadata /weka/dfive-default/henryh/helios/olmoearth_pretrain/v0_1_osm_sampling_scores.parquet \
  --metadata-columns filename elevation_mean is_temperate is_subtropical is_tropical is_boreal_polar is_mountainous ndwi_mean patch_uniformity spatial_homogeneity canopy_height_mean \
  --output-name embedding_map_with_images_nested_hdbscan.html \
  --thumbnail-url-prefix ./thumbnails/
```

### Step 3: Serve and open in browser
```shell
export IMAGES_DIR="/weka/dfive-default/helios/dataset/osm_sampling/thumbnails/h5py_data_w_missing_timesteps_zstd_3_128_x_4/cdl_gse_landsat_openstreetmap_raster_sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/"
# Option A: convenience script (handles thumbnail dir symlinks)
python scripts/embeddings/serve_map.py   --viz-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/_viz   --thumbnail-dir "$IMAGES_DIR"
```

<br/>
<br/>

---
---
# SECTION C: Overlay new embeddings on frozen reference subclusters
---
---
<br/>
Use this when you want a fast visual sanity check that a newly computed
embedding set mixes into the existing reference subclusters instead of drifting
into obviously different geometry.

This mode:
1. keeps the reference cluster layout fixed
2. keeps the displayed subcluster partition from the reference labels
3. renders all reference points in gray
4. projects the new embeddings with the frozen reference PCA + parent clusters
5. renders the new points as black dots with a sharp light-green outline

```shell
python scripts/embeddings/visualize_embeddings.py \
  --layout cluster \
  --cluster-bundle /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/cluster_bundle_kmeans.npz \
  --labels /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_cluster/labels_kmeans_residual_hdbscan.npy \
  --overlay-input-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings" \
  --overlay-reference-model /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores/reference_model.npz \
  --metric cosine \
  --separation 3 \
  --darkmode \
  --output-name embedding_map_reference_overlay.html
```

Important:
1. `--cluster-bundle` and `--overlay-reference-model` must come from the same reference fit
2. if you want the nested HDBSCAN partition shown, pass the nested labels file with `--labels`
3. the new points are overlaid only for visualization; they do not change the reference partition

<br/>
<br/>

---
---
# SECTION D: SELECTION / DATASET EXPANSION
---
---
<br/>
## Fit frozen reference artifacts
This is the selection workflow used to compare/score a new candidate embedding set
against the current OLMoEarth pretraining embeddings.
Note: Use the same seed to obtain similar K-means cluster than other sections

It fits:
1. a frozen global PCA model
2. frozen parent spherical k-means clusters
3. per-parent novelty thresholds in residual space: for every reference point, compute its mean distance to the `--knn-k` nearest reference residuals **inside the same parent cluster**, and return the `--percentile` of the distribution per parent cluster.

```shell
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
python scripts/embeddings/reference_model.py \
  --input-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes \
  --output-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores \
  --seed 1234 \
  --pca-dim 128 \
  --k 15 \
  --knn-k 25 \
  --percentile 99
```

Main outputs:
1. `reference_model.npz`
2. `reference_residuals.npz`
3. `reference_parent_labels.npy`
4. `reference_scores.parquet`
5. `reference_summary.json`

## Novelty score
Best when you want isolated or genuinely novel points relative to the current
reference set.

Each candidate is:
1. projected with the frozen PCA
2. assigned to a frozen parent spherical k-means cluster
3. converted into a residual vector relative to that parent centroid
4. scored by mean distance to the `k` nearest reference residuals inside that parent
5. compared to the stored parent-specific percentile threshold

```shell
export OPENBLAS_NUM_THREADS=16
export OMP_NUM_THREADS=16
python scripts/embeddings/score_novelty.py \
  --input-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings" \
  --reference-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores \
  --output-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores"
```

Outputs:
1. `novelty_scores.parquet`
2. `novelty_accepted_mask.npy`
3. `novelty_accepted_sample_idx.npy`
4. `novelty_parent_assignments.npy`
5. `novelty_summary.json`

How it works:
1. the reference set is used to fit a frozen global PCA and frozen spherical k-means parent clusters
2. each candidate is projected into that same PCA space and assigned to its nearest parent cluster
3. inside that parent cluster, the candidate residual vector `(point - parent centroid)` is compared to reference residual vectors from the same parent
4. the raw novelty score is the mean distance to the `k` nearest reference residual neighbors
5. this score is then compared to the stored parent-specific percentile threshold from the reference set

Interpretation:
1. score is high when a candidate sits far from its nearest reference neighbors inside the assigned parent cluster
2. this is the most direct strategy for tail novelty / outlier discovery

Threshold:
1. `--percentile 99` means compute the same novelty score for each reference sample inside its own parent cluster
2. take the `99th` percentile of those scores separately for each parent cluster
3. accept a candidate only if it is more novel than that per-parent threshold

This avoids using one global absolute distance cutoff across all parent clusters.


## Score additional acquisition strategies
These scores reuse the same frozen reference artifacts from `fit-reference`,
but target different kinds of useful samples than pure novelty.

### X-global-cluster bridge
Best when you want candidates that sit between two parent clusters rather than
deep inside one of them.

```shell
python scripts/embeddings/score_acquisition.py \
  xglobal_bridge \
  --input-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings" \
  --reference-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores \
  --output-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores"
```

Outputs:
1. `xglobal_bridge_scores.parquet`
2. `xglobal_bridge_ranked_sample_idx.npy`
3. `xglobal_bridge_summary.json`

How it works:
1. candidates are projected with the frozen PCA into the same space as the parent cluster centroids
2. distances to all frozen parent centroids are computed and the two closest are selected
3. a balance term is high when the candidate is comparably close to both parent centroids rather than clearly owned by only one
4. a between-ness term is high when the candidate lies close to the line segment joining those two parent centroids
5. the perpendicular distance to that axis is normalized by a frozen reference-derived off-axis scale estimated separately for each parent-pair transition zone
6. a mild segment-position gate downweights points projected well beyond either end of that segment
7. the final score is the product of the balance term and the between-ness term

Interpretation:
1. score is high when a candidate lies geometrically between two parent cluster centroids
2. candidates that are equidistant from two parents but far off-axis score lower than true transition-zone points
3. these are useful transition / confusion / edge cases

### Sparse cluster infill
Best when you want points in sparse-but-still-supported regions of an existing
parent manifold, not just extreme outliers.

```shell
python scripts/embeddings/score_acquisition.py \
  sparse-infill \
  --input-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings" \
  --reference-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores \
  --output-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores" \
  --k-sparse 256 \
  --k-support 32 \
  --sparse-percentile 90 \
  --support-percentile 60
```

How it works:
1. candidates are assigned to a parent cluster and converted to residual vectors relative to that parent centroid
2. within each parent cluster, two separate reference-calibrated kNN distances are computed in residual space:
3. `d_sparse`: mean distance to `k_sparse` reference neighbors, used as a cluster-local sparsity statistic
4. `d_support`: mean distance to `k_support` reference neighbors, used as a cluster-local support statistic
5. `d_sparse` is compared against the per-parent reference `sparse-percentile` gate, so the sparsity term turns on once a candidate moves into the sparse tail of that parent distribution
6. `d_support` is compared against the per-parent reference `support-percentile` gate, so the support term turns off once a candidate becomes too weakly supported by a local neighborhood
7. the final score is `s_sparse * s_support`, which favors sparse-but-still-supported regions of the residual distribution

Interpretation:
1. score peaks in under-covered shoulders of the residual distribution
2. dense cores score lower
3. candidates supported by only one stray reference point are penalized
4. extreme isolated outliers are penalized

### X-local-cluster bridge
Best when you want candidates that connect two nearby local modes inside the
same parent cluster.

```shell
python scripts/embeddings/score_acquisition.py \
  xlocal_bridge \
  --input-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings" \
  --reference-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores \
  --output-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores" \
  --local-modes 5
```

How it works:
1. candidates are assigned to parent clusters and represented in residual space
2. inside each parent cluster, the reference residuals are grouped into a small number of local KMeans subclusters
3. for each candidate, distances to those local subcluster centroids are computed and the two closest centroids are selected
4. a balance term is high when the candidate is comparably close to both centroids rather than clearly owned by only one
5. a between-ness term is high when the candidate lies close to the line segment joining those two centroids
6. the perpendicular distance to that axis is normalized by a frozen reference-derived off-axis scale estimated separately for each local-mode-pair transition zone, using the same transition-band rule as `xglobal_bridge`
7. a mild segment-position gate downweights points projected well beyond either end of that segment
8. the final score is the product of the balance term and the between-ness term

Interpretation:
1. score is high when a candidate lies geometrically between two local residual-space KMeans subclusters
2. these samples can act as connective tissue rather than tails

### High-quality prototypes
Best when you want clean, representative points from dense local neighborhoods.

```shell
python scripts/embeddings/score_acquisition.py \
  prototypes \
  --input-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings" \
  --reference-dir /weka/dfive-default/hadriens/oe_inst_embeddings_ps8_shwp12_4s_s2_ixes/_scores \
  --output-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores" \
  --local-prototypes 50 \
  --radius-percentile 80 \
  --coverage-k 16
```

How it works:
1. candidates are assigned to parent clusters and evaluated in residual space against reference residuals from the same parent
2. inside each parent cluster, the reference residuals are partitioned into a fixed number of local KMeans prototype subclusters
3. each prototype subcluster contributes one centroid and a prototype radius estimated from the reference distances to that centroid
4. each candidate is matched to its nearest prototype centroid inside the assigned parent cluster
5. the score is high when the candidate lies close to that centroid relative to the centroid's reference radius
6. with `--coverage-k > 0`, a sparsity-ratio coverage penalty is computed: the candidate's mean `k`-NN distance to the reference points in its matched subcluster is compared to the mean self `k`-NN distance of the reference points in that subcluster. If the candidate sits in a denser region than the reference average, the coverage term drops below 1, softly penalizing redundant candidates
7. candidates far from all local prototype centroids score lower

Interpretation:
1. score is high for candidates that closely match a local reference prototype centroid
2. this favors representative candidates rather than novel or tail candidates
3. with `--coverage-k > 0`, candidates in already-dense prototype regions are penalized relative to candidates in sparser regions of the same prototype


<br/>

---
---

# SECTION E: Combine strategy scores and select candidates

---
---
<br/>
Use this when you want one final ranking that mixes several acquisition goals
instead of selecting from a single score.

By default this combines:
1. novelty: `0.4`
2. xglobal_bridge: `0.2`
3. sparse-infill: `0.2`
4. xlocal_bridge: `0.1`
5. prototypes: `0.1`

## X-strategy scoring
Each strategy score is normalized first, then combined with the chosen
weights. `rank` normalization is the safest default because the raw score
scales differ across strategies.

```shell
python olmoearth_pretrain/dataset_creation/candidate_embeddings/combine_acquisition.py \
  --scores-dir "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores" \
  --normalization rank \
  --weight-novelty 0.4 \
  --weight-xglobal-bridge 0.2 \
  --weight-sparse-infill 0.2 \
  --weight-xlocal-bridge 0.1 \
  --weight-prototypes 0.1 \
  --ablation
```

Outputs:
1. `combined_acquisition_scores.parquet`
2. `combined_ranked_sample_idx.npy`
3. `combined_acquisition_summary.json`

How it works:
1. the script loads the novelty and acquisition score parquet files already written under `_scores/` (legacy `.csv` files are still accepted)
2. each active strategy score is normalized onto a comparable scale
3. the normalized scores are combined with user-specified weights
4. the final `combined_score` is the weighted average across the selected strategies
5. the output parquet keeps both the raw and normalized component scores for inspection and debugging

### Ablation mode
Add `--ablation` to the same command to also measure the marginal value of
each strategy in your production weighting (leave-one-out).


## Build a top-X selection table across strategies and ablations
This step consumes the combined parquet from the previous step (must be the
ablation-mode file, so the `combined_score_drop_*` columns exist) and writes
a single deduplicated selection table per sample budget `X`.

For budget `X` it picks:
1. top `3 * X / 5` candidates for each of the 5 standalone strategies, ranked
   by that strategy's `*_normalized_score`
2. top `4 * X / 5` candidates for each of the 5 leave-one-out ablation scores
   (`combined_score_drop_<strategy>`)
3. top `X` candidates for the full weighted `combined_score`

```shell
python scripts/embeddings/select_top_samples.py \
  --combined-parquet "${RSLEARN_EAI_ROOT}/datasets/globe_land_grid/s50ix24_embeddings/_scores/combined_acquisition_scores.parquet" \
  --num-samples 250000
```

Outputs (written next to the input parquet, override with `--output-dir`):
1. `selection_top<X>.parquet`
2. `selected_sample_ids_top<X>.json` – flat JSON list of the selected `window_name` values

Columns in the parquet:
1. sample metadata: `sample_idx`, `window_name`, `lat`, `lon`, `parent_label`
2. `in_top_combined` -- 1 if the sample was in the top-X of `combined_score`
3. `in_top_solo_<strategy>` for each of the 5 strategies -- 1 if the sample was in the top-3X/5 of that strategy's standalone normalized score
4. `in_top_drop_<strategy>` for each of the 5 strategies -- 1 if the sample was in the top-4X/5 of that leave-one-out ablation score

Each row appears exactly once (duplicates across selections are merged); a
sample that satisfies several criteria simply has `1` in several indicator
columns. No scores are carried over -- only metadata plus the 0/1 indicators.
