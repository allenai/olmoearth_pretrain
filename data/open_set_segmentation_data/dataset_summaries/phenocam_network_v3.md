# PhenoCam Network v3 (`phenocam_network_v3`)

- **Status:** completed
- **Task type:** classification (sparse point segmentation, spec §2a)
- **Num samples:** 916 points
- **Output:** `datasets/phenocam_network_v3/points.geojson` + `metadata.json`

## Source

The PhenoCam network of near-surface (ground camera) phenology stations. The ORNL DAAC
"PhenoCam Images V3" product (https://daac.ornl.gov/VEGETATION/guides/Phenocam_Images_V3.html)
corresponds to this network. We do **not** need the (Earthdata-gated, multi-GB) image
archives — only the *labels*: each site's coordinate and its human-assigned
vegetation/land-cover type.

That label-only signal is published openly (no credentials) by the PhenoCam project API:
`https://phenocam.nau.edu/api/cameras/?format=json` — returns `Sitename`, `Lat`, `Lon`,
`date_first`/`date_last`, `active`, and `sitemetadata.primary_veg_type`. We cache the full
raw response to `raw/phenocam_network_v3/cameras.json` for reproducibility. (Note the live
API lists 1063 sites, a superset of the ~738 in the ORNL v3 release; we use the current
network as the label source.)

## Triage / accept reasoning

Each PhenoCam station monitors a dominant, relatively homogeneous vegetation stand, and the
PhenoCam team assigns the site's land-cover type by **human vegetation typing** (in-situ
reference, not derived-product). The vegetation type is a genuine ground-truth
characterization of the site ecosystem — not merely a property of the camera imagery — and
the distinctions kept (forest / grassland / cropland / wetland / shrub / tundra /
non-vegetated) are all resolvable at 10–30 m from S2/S1/Landsat. Post-2016 and georeferenced.
**Accepted** as a weak site-level land-cover reference point dataset, handled like other
point land-cover/habitat references (e.g. `olmoearth_lcmap_land_use`).

**Caveat (weak label):** the coordinate is the *camera* location and the field of view is
oblique/local, so the 10 m pixel at the coordinate is only an approximate stand-in for the
site's dominant land cover (the camera may sit at a tower/edge). Downstream assembly treats
these as weak labels. No live S2 overlay was rendered (point labels over an authoritative
global reference network; coordinates validated to WGS84 ranges).

## Class mapping (PhenoCam `primary_veg_type` → class id)

| id | name | code | count |
|----|------|------|-------|
| 0 | Deciduous broadleaf forest | DB | 208 |
| 1 | Evergreen needleleaf forest | EN | 157 |
| 2 | Grassland | GR | 162 |
| 3 | Agriculture | AG | 195 |
| 4 | Shrub | SH | 78 |
| 5 | Wetland | WL | 62 |
| 6 | Evergreen broadleaf forest | EB | 27 |
| 7 | Tundra | TN | 17 |
| 8 | Non-vegetated | NV | 7 |
| 9 | Deciduous needleleaf forest | DN | 3 |
| 10 | Mixed forest | MX | 0 (no sites in current network) |

Sparse classes (TN, NV, DN, MX) are retained per spec §5; downstream assembly drops
too-small classes. `MX` is kept in the class map for completeness though no current site
carries it.

## Dropped records

From 1063 raw sites: **96** with missing/empty `primary_veg_type`, **43** whose activity is
entirely pre-2016, **4** understory-only cameras (`UN`, no coherent overhead land-cover
meaning), **4** with unparseable dates. → 916 usable points.

## Time-range / change handling

Vegetation/land-cover type is a **persistent (static) label** (spec §5). Each point gets a
1-year window `[Jan 1 Y, Jan 1 Y+1)` where `Y = clamp(max(2016, first_year), last_year)` —
i.e. the first Sentinel-era year the site was operating. No `change_time` (not a change
label).

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.phenocam_network_v3
```

Idempotent: re-running re-fetches the API and overwrites `points.geojson`/`metadata.json`
atomically (no per-file tif tree). The `green-up/down dates` phenology transition product
listed in the manifest classes is a separate regression target and is **not** included here;
this dataset covers only the site land-cover typing.
