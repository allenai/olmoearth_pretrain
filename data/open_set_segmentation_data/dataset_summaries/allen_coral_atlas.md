# Allen Coral Atlas

- **Slug:** `allen_coral_atlas`
- **Status:** completed
- **Task type:** classification (dense polygon rasterization)
- **Samples:** 4,728 label tiles (64×64, 10 m UTM, uint8, nodata=255)
- **License:** CC-BY-4.0 (attribution)

## Source & access

Allen Coral Atlas (Arizona State University / Planet / University of Queensland),
https://allencoralatlas.org/ — global 5 m maps of shallow tropical coral-reef
**geomorphic zonation** and **benthic habitat**, built from Planet Dove mosaics (~2018–2021)
trained/validated with field photo-quadrats + contextual editing.

The website's bulk download is behind a free login (no credential for it in
`.env`), and the Google Earth Engine mirror (`ACA/reef_habitat/v2_0`)
needs a GEE key that is **not present** in this environment (the `.env`
`TEST_GEE_SERVICE_ACCOUNT_CREDENTIALS` points at `/etc/credentials/gee_key.json`, which does
not exist). **No credential gate was hit, however**, because the identical vector maps are
served openly (CC-BY-4.0, `Fees NONE`) from the ACA **GeoServer WFS** at
`https://allencoralatlas.org/geoserver/ows`:

- `coral-atlas:benthic_data_verbose` — benthic cover polygons
- `coral-atlas:geomorphic_data_verbose` — geomorphic zone polygons

Both are EPSG:4326 `MultiPolygon` layers with attributes `{class_name, area_sqkm}`.
(urllib needs a browser `User-Agent`, else HTTP 403.) This is the access path used here.

## Bounded regional sampling (global derived product, spec §5)

ACA is a global derived-product map, so we did **bounded regional sampling** rather than
global coverage. We pulled WFS polygons for **21 reef regions** spanning the major reef
provinces and habitat types, then tiled/rasterized locally:

Great Barrier Reef (Lizard, Cairns, Capricorn), Maldives, Red Sea (Farasan, Egypt),
Persian Gulf (Qatar), Belize barrier reef, Bahamas (Exuma), Florida Keys, Hawaii
(Kāne‘ohe), Moorea, Tuamotu (Rangiroa), New Caledonia (barrier + lagoon), Fiji (Suva),
Philippines (Palawan), Indonesia (Wakatobi), Seychelles (Mahé), Gulf of Mannar (India),
Zanzibar. (A 22nd candidate box, Chagos, returned 0 features and was dropped.) Per-region
tile counts are in `metadata.json` (`region_tile_counts`); largest are New Caledonia (509),
Zanzibar (371), Maldives (359), Red Sea Farasan (351).

## Class scheme & 10 m suitability

The ACA benthic (6) and geomorphic (11) classes ARE the product's **top-level** legend —
broad cover / zonation categories occupying large contiguous reef areas, **not** sub-metre
zonation. They are resolvable at 10 m (native 5 m; we resample by polygon rasterization at
10 m). We therefore **kept the full top-level legend for both families and attempted no
finer sub-classes** (the manifest's "coral/seagrass fine zonation" concern does not arise —
the WFS only exposes these top-level classes).

Benthic and geomorphic are two **orthogonal segmentations of the same pixels**, so they
cannot share one per-pixel raster. We keep **one dataset with a unified legend (17 ids)**;
each output tile is rasterized from a **single family** (benthic OR geomorphic), so its
pixels carry only that family's ids, and the sample JSON `classes_present` records which.

| id | name | family | tiles |
|----|------|--------|------|
| 0 | Coral/Algae | benthic | 1371 |
| 1 | Seagrass | benthic | 998 |
| 2 | Sand | benthic | 1461 |
| 3 | Rubble | benthic | 1192 |
| 4 | Rock | benthic | 1139 |
| 5 | Microalgal Mats | benthic | 1074 |
| 6 | Reef Slope | geomorphic | 1101 |
| 7 | Sheltered Reef Slope | geomorphic | 1123 |
| 8 | Reef Crest | geomorphic | 1084 |
| 9 | Outer Reef Flat | geomorphic | 1982 |
| 10 | Inner Reef Flat | geomorphic | 1349 |
| 11 | Terrestrial Reef Flat | geomorphic | 1059 |
| 12 | Back Reef Slope | geomorphic | 1285 |
| 13 | Plateau | geomorphic | 1027 |
| 14 | Patch Reef | geomorphic | 0 (not present in the 21 sampled regions) |
| 15 | Deep Lagoon | geomorphic | 999 |
| 16 | Shallow Lagoon | geomorphic | 1053 |

(Tiles per family: benthic 1658, geomorphic 3070.) Counts are **tiles-per-class** (a tile
counts toward every class it contains), so common classes exceed 1000 as a side effect of
co-occurring in tiles selected to fill rarer classes; the per-class dedicated fill is capped
at 1000 and total (4,728) is far under the 25k cap. Class 14 **Patch Reef** was not present
in any sampled region (kept in the legend with 0 tiles). Microalgal Mats was reachable to
~1074 tiles thanks to the turbid Persian Gulf + Red Sea regions.

## Encoding decisions

- **Positive-only** reef maps (spec §5): non-reef / unmapped pixels are left as **nodata
  (255)** — no background class is fabricated; the assembly step supplies negatives from
  other datasets.
- Polygons reprojected WGS84→local UTM pixel space, clipped to each 64×64 (640 m) tile,
  rasterized with `all_touched=True` (larger polygons drawn first so slivers of small
  classes survive overlaps). Fill = 255.
- **Time range:** benthic cover / geomorphic zonation are persistent habitat/geological
  features and the maps are a 2018–2021 composite, so a **static 1-year window (2020)** with
  `change_time=null` (spec §5 static labels).

## Verification (spec §9)

- Sampled tifs: single band, UTM CRS (e.g. 32758/32737/32751/32760/32651), 10 m res,
  64×64, uint8, nodata 255, values are valid class ids; each has a matching JSON with a
  1-year `time_range`; declared class ids cover all observed values.
- Coordinate sanity: tile centroids land in the expected reef regions (e.g. sample
  Zanzibar tile → 39.505 E, −6.192 S; Fiji → 178.514 E, −18.126 S; Wakatobi → 123.614 E,
  −5.387 S). Full Sentinel-2 overlay was not rendered; centroid geolocation confirms the
  labels sit on the intended shallow-reef areas.
- Idempotent: re-running skips existing tiles and recovers metadata counts from sidecars.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.allen_coral_atlas
```

Outputs on weka under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/`:
`raw/allen_coral_atlas/` (per-region benthic+geomorphic WFS GeoJSON, ~689 MB),
`datasets/allen_coral_atlas/{metadata.json, registry_entry.json, locations/*.tif+*.json}`.

## Caveats

- Derived-product map (not in-situ reference); non-reef pixels are ignore(255).
- Benthic vs geomorphic tiles share one legend but each tile is single-family (see above).
- Patch Reef (id 14) has 0 samples in the sampled regions; some common classes exceed 1000
  tiles due to tiles-per-class co-occurrence. Both are expected and documented.
- Leap-year 2020 window is 366 days (calendar-year convention of the shared `io.year_range`
  helper; within the pretraining ~1-year cap).
