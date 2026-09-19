# FAO GSASmap (Salt-Affected Soils) — `fao_gsasmap_salt_affected_soils`

**Status: REJECTED (source temporarily unavailable — FAO server outage).**
No `datasets/{slug}/` outputs were written other than the required
`registry_entry.json`.

## What the dataset is

The FAO Global Map of Salt-Affected Soils (GSASmap v1.0), produced by the FAO Global
Soil Partnership. It is a country-driven global product harmonized from 257,419 measured
field observations across 118 countries (≈85% of global land area). Distributed as
30 arc-second (~1 km) GeoTIFFs, with layers at two depth intervals (0–30 cm and
30–100 cm): electrical conductivity (ECe), exchangeable sodium percentage (ESP), pH, and
a categorical **classes of salt-affected soils** layer.

- Product / landing page: https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/global-map-of-salt-affected-soils/en/
- GSP page: https://www.fao.org/global-soil-partnership/gsasmap/en/
- Data platform ("ACCESS THE GSASmap DATA PLATFORM"): http://54.229.242.119/GloSIS/ →
  redirects to https://data.apps.fao.org/glosis/ (a TerriaJS map viewer, "GloSIS").

## Task-type decision (documented, though not executed)

**Classification.** The manifest target classes are the categorical salt-affected-soil
types from the GSASmap "classes" layer:

| id | class | topsoil dist. | subsoil dist. |
|----|-------|---------------|---------------|
| 0  | saline       | ~85% | ~62% |
| 1  | sodic        | ~10% | ~24% |
| 2  | saline-sodic | ~5%  | ~14% |

Definitions (FAO GSASmap): salt-affected pixels are those with ECe > 2 dS/m,
ESP > 15%, and pH > 8.2; saline vs sodic vs saline-sodic is assigned from the ECe/ESP
thresholds. This is the categorical layer the manifest lists, hence classification.
(The product's continuous ECe / ESP / pH rasters could alternatively support a
*regression* target, but the manifest's declared classes are categorical, so
classification is the correct choice for this entry.)

Intended processing had the source been reachable (global derived-product raster →
bounded-tile sampling, per spec §4/§5, mirroring
`datasets/jrc_tropical_moist_forest_tmf.py`):
- Download the ~1 km global "classes" GeoTIFF (topsoil 0–30 cm).
- Bounded-tile sample representative regions; reproject native ~1 km to local UTM at
  10 m with **nearest/mode** resampling (categorical); ≤64×64 tiles.
- Prefer spatially-homogeneous windows; balance to ≤1000 tiles/class (3 classes → well
  under the 25k cap). uint8, class ids 0–2, nodata = 255.
- Time range: 1-year window in the mapped period (manifest time_range 2016–2021; the
  product reflects a static contemporary map — pick a representative Sentinel-era year,
  e.g. 2019/2020).

## Why rejected

The GSASmap rasters are **openly licensed (FAO open data) and require no credential**,
but the **only programmatic distribution endpoints are currently returning HTTP 502
(backend outage)** — this is a server-side FAO infrastructure problem, not an access/
credential wall or a permanent dead link. Access investigation performed on 2026-07-11:

- **FAO GloSIS / GSP GeoServer — `https://io.apps.fao.org/geoserver/...`**: HTTP 502 on
  every endpoint (root, `/geoserver/web/`, WMS & WCS GetCapabilities), 30+ consecutive
  retries. DNS resolves (35.227.205.77, Google LB); the 502 originates from FAO's backend
  behind the load balancer, i.e. the service is down, not blocked from our network. This
  is the host that serves FAO GSP global soil rasters (confirmed pattern via GLEAM3 on the
  same GeoServer).
- **FAO map GeoNetwork catalog — `https://data.apps.fao.org/map/catalog/...`**: HTTP 502
  (both the `srv/eng/q` and `srv/api/search/records/_search` endpoints).
- **FAO GIS Manager API — `https://io.apps.fao.org/gismgr/...`**: HTTP 502.
- **GloSIS Terria viewer — `https://data.apps.fao.org/glosis/`**: loads a JS/Terria SPA
  whose catalog config is not fetchable as a static file (`config.json`, `init/*.json` all
  404) and whose data layers are backed by the (down) io.apps.fao.org GeoServer.
- **Up-but-wrong FAO GeoServer — `https://data.apps.fao.org/map/gsrv/edit/ows`** (the
  Hand-in-Hand instance, 563 coverages / 817 WMS layers): scanned all layer/coverage
  names — **no salt / GSAS / salinity / sodic layer**. GSASmap is not published here.
- **Legacy GloSIS GeoServer — `http://54.229.242.119/geoserver/`**: responds but now
  publishes **zero layers** (data migrated off it).
- **Mirrors**: no Zenodo copy, no Google Earth Engine community asset, no direct
  `.tif`/`.zip` on FAO fileadmin or ISRIC found. The FAO Soils Portal and GSP pages link
  only to the (down) GloSIS platform.

Per the task spec ("try unauthenticated/mirror/alternate access briefly, then reject … so
it collects in the registry for the user to act on later"), this is recorded as rejected
with an actionable note. It is a **transient outage**, not a fundamental
non-observability or credential problem — the dataset is a good fit (global salinity-class
raster, cleanly classification, resolvable at 10 m as a coarse ~1 km product) and should
be **retried once FAO infrastructure recovers**.

## How to reproduce / retry (when FAO is back up)

1. Confirm the GeoServer is healthy:
   `curl -sI "https://io.apps.fao.org/geoserver/web/"` → expect HTTP 200.
2. List coverages and find the GSASmap "classes" layer:
   `curl -s "https://io.apps.fao.org/geoserver/wcs?service=WCS&version=2.0.1&request=GetCapabilities"`
   (WCS 2.0.1 uses `__` as the namespace/layer separator on this server, e.g.
   `WORKSPACE__layername`; grep the CoverageId list for salt/sas/salinity).
   Alternatively browse `https://data.apps.fao.org/glosis/` to identify the layer, or
   query the FAO map GeoNetwork `https://data.apps.fao.org/map/catalog/`.
3. Download the topsoil (0–30 cm) "classes" GeoTIFF (WCS GetCoverage, `format=image/tiff`),
   then run bounded-tile sampling as described above (mirror
   `olmoearth_pretrain/open_set_segmentation_data/datasets/jrc_tropical_moist_forest_tmf.py`).
4. Command once a `datasets/fao_gsasmap_salt_affected_soils.py` script exists:
   `python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.fao_gsasmap_salt_affected_soils`

## Caveats for eventual processing

- Native resolution is ~1 km; at S2/S1/Landsat scale each label pixel is very coarse.
  This is a low-precision derived-product map (measured points + expert interpolation),
  so prefer homogeneous windows and treat it as coarse/weak supervision.
- Consider whether the continuous ECe raster (0–30 cm) is a more useful *regression*
  target than the 3-class categorical layer; the manifest asks for the classes, but both
  are available in the same product.
