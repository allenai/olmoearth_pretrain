# xView3-SAR (dark vessel detection)

- **Slug:** `xview3_sar_dark_vessel_detection`
- **Status:** `rejected` — `needs-credential`
- **Family:** vessels · **label_type:** points (object detection) · **task_type (planned):** classification (detection encoding)
- **Source:** DIU / Global Fishing Watch — xView3-SAR challenge, https://iuu.xview.us/
- **License:** open (non-commercial research)

## What the source is
~1,000 large Sentinel-1 SAR scenes (GRD, on average ~29,400 x 24,400 px) over 11 EEZs,
with 220k+ georeferenced maritime-object annotations (dark / non-broadcasting vessels,
other vessels, fixed structures) produced by combining AIS tracks, automated SAR analysis,
and manual visual detection. Manifest classes: fishing vessel, non-fishing vessel, fixed
structure. Labels are CSVs (`GRD_train.csv`, `GRD_validation.csv`, and SLC equivalents)
whose rows carry WGS84 `detect_lat` / `detect_lon`, `is_vessel`, `is_fishing`,
`vessel_length_m`, `confidence`, `scene_id`.

This is a good fit for the corpus (specific-image detections, directly geolocated, at
Sentinel-1 resolution) — it is **not** rejected on suitability grounds. It is blocked
purely on access.

## Rejection reason (needs-credential)
The labels and imagery are distributed **only behind a registration/login wall** at
`https://iuu.xview.us/signup` (DrivenData-style account). Open mirrors were checked briefly
and none host the label CSVs unauthenticated:

| Source | Result |
|---|---|
| iuu.xview.us | "Register/Login to Download Data" — account required |
| allenai/sar_vessel_detect (AI2 xView3 model) | no data links; points back to iuu.xview.us |
| DIUx-xView/xview3-reference | points back to iuu.xview.us |
| DIUx-xView/SARFish | imagery gated on HF; labels still from xView3 site |
| ConnorLuckettDSTG/SARFishSample (public HF) | only 1 sample GRD + 1 SLC scene; **no label CSVs** |
| ai2-prior-sarfish S3 | only a model checkpoint public; bucket listing denied |
| weka `/weka/dfive-default/joer/agent-yawen/repos/xview3` | code checkout only; no CSVs |

Per the task SOP, a credential gate is a `rejected` with
`notes: "needs-credential: ..."` (not `temporary_failure` — the block is a permanent
access gate, not a transient outage). The user can supply the registered CSVs out of band.

## How to complete it later (retry is implemented and idempotent)
The processing script is written and ready. Drop the registered files into
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/raw/xview3_sar_dark_vessel_detection/`:

1. **Label CSVs:** any of `GRD_train.csv`, `GRD_validation.csv` (SLC_* also accepted — same
   schema). Columns used: `scene_id, detect_lat, detect_lon, is_vessel, is_fishing`.
2. **`scenes.csv`:** two columns `scene_id,acquisition_time` (ISO 8601). xView3 scene ids do
   **not** embed the timestamp, so this mapping is required to honor the specific-image
   ~1-hour time range (spec §5). The Sentinel-1 product names in the challenge file listing
   embed the datetime (`S1x_IW_GRDH_..._YYYYMMDDThhmmss_...`), so it is trivially derivable.

Then re-run:
```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.xview3_sar_dark_vessel_detection
```

## Planned processing (spec §4 detection; mirrors `olmoearth_sentinel_2_vessels`)
- **Class scheme (unified):** `0 background`, `1 fishing_vessel` (is_vessel & is_fishing),
  `2 non_fishing_vessel` (is_vessel & not is_fishing), `3 fixed_structure` (is_vessel=False).
  Rows with `is_vessel` unknown, or a vessel with `is_fishing` unknown, are dropped for class
  purity (noted here as a caveat — a meaningful fraction of xView3 rows have low/`NaN`
  confidence attributes).
- **Detection encoding:** one 64×64 UTM 10 m context tile per detection built **directly from
  lon/lat** (no SAR raster read needed — we do not touch the multi-TB imagery), 1 px positive
  of the class id, 10 px nodata (255) buffer ring (SAR detect coords are not pixel-exact),
  rest background (0). Co-located same-scene detections inside a tile are also marked.
- **Negatives:** background-only tiles sampled at ocean points ≥30 px from every detection,
  inside each scene's detection bounding box, so class 0 has spatially-meaningful negatives
  (detection exception, spec §5).
- **Sampling:** tiles-per-class balanced, up to 1000 per class, hard cap 25,000
  (`select_tiles_per_class` + `MAX_SAMPLES_PER_DATASET`). All splits used.
- **Time range:** each detection uses its scene's ~1-hour acquisition window
  (specific-image, spec §5).
- **Sensors_relevant:** `sentinel1` (SAR-native detections).

## Judgment calls
- Treated as `rejected`/needs-credential rather than `temporary_failure`: the barrier is a
  standing registration wall, not a transient server error.
- We only need label CSVs + per-scene acquisition times; the full SAR imagery is deliberately
  **not** downloaded (labels are self-geolocating via detect_lat/detect_lon).
- Vessels of unknown fishing status and objects of unknown is_vessel are dropped rather than
  forced into a class, to keep the 3-class scheme clean.
