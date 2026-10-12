# MapBiomas Alerta

- **slug**: `mapbiomas_alerta`
- **status**: **rejected** — **needs-credential** (SOP §8.2). The validated deforestation-alert
  polygons are only served through the MapBiomas Alerta GraphQL API (and the login-gated
  website Shapefile/Excel exports), which require a **MapBiomas Alerta user account**
  (email + password → Bearer token via the `signIn` mutation, account email-confirmed). No
  such credential is present in `.env`, and there is **no anonymous /
  open-access path** to the alert polygons (verified below). This is a permanent access gate
  (a free registration portal), so it is `rejected` with a `needs-credential` note, **not**
  `temporary_failure`: the endpoint is up and functional, the block is purely the login gate.
- **task_type** (intended, had a token been available): **classification** — a dated CHANGE
  dataset (single foreground class `deforestation`, positive-only → nodata outside), polygons
  rasterized to ≤64×64 UTM 10 m tiles, `change_time` set + a 1-year window centered on it.
- **num_samples**: 0 (only `registry_entry.json` written to weka; no `metadata.json` / `locations/`).

## Source

- Manifest: name `MapBiomas Alerta`, source **MapBiomas**, url
  <https://alerta.mapbiomas.org/en/>, classes `["validated deforestation event"]`,
  time_range `[2019, 2025]`, family `deforestation`, region "All Brazilian biomes",
  label_type `polygons`, annotation_method "manual photointerpretation validation",
  license **CC-BY-SA-4.0**, have_locally: false.
- MapBiomas Alerta validates and refines deforestation alerts (from DETER/INPE, SAD/IMAZON,
  GLAD/GFW, PRODES, etc.) for every Brazilian biome since Jan 2019, using daily PlanetScope
  ~3.7 m imagery. Trained analysts select a **before** and **after** high-resolution image
  for each event and refine the polygon boundary — "an effect similar to a photo of a
  vehicle's license plate." Data is described as public/open with source attribution.

## Access investigation (why rejected)

Access mechanisms and the exhaustive open-access check performed:

1. **GraphQL API** — endpoint `https://plataforma.alerta.mapbiomas.org/api/v2/graphql`.
   The `alerts` query (paginated, filterable by `boundingBox`, `startDate`/`endDate`,
   `dateType`, size, source, etc.) returns polygon geometry directly as `geometryWkt`, plus
   all the date fields needed (see below). But an **unauthenticated** `alerts` request
   returns:
   ```json
   {"errors":[{"message":"Token de acesso inválido","path":["alerts"]}]}
   ```
   The API docs state plainly: *"A autenticação … é feita através da mutation `signIn` e
   necessária para todas as demais requisições a API"* (auth via `signIn` is required for all
   API requests). `signIn(email, password)` needs a real, email-confirmed account; there is
   **no service-account access** (docs: *"Atualmente, o acesso não possui acesso por conta de
   serviço"*).
2. **Website downloads** — the FAQ lists Shapefile (.shp), per-year Excel (with CAR), PDF
   reports, and a QGIS plugin, but these are an *"open service for registered users"* and the
   "download all alerts" path routes through the authenticated `relatoryAlert` query
   (`fileUrl`), i.e. same token gate.
3. **Frontend has no public token** — the React app builds its `authorization: Bearer …`
   header from `localStorage.getItem("token")`, which is only populated after a user logs in;
   there is **no embedded anonymous/guest token** in the JS bundles
   (`main.*.chunk.js`, `596.*.chunk.js`). Anonymous map browsing uses raster tile layers, not
   the alert-data query.
4. **No public GCS mirror** — `mapbiomas-public` GCS bucket has **no** objects under
   `alerta`/`initiatives/alerta` prefixes (Storage JSON API listing returned empty).
5. **No public GEE asset for the Alerta polygons** — MapBiomas publishes public GEE assets
   under `projects/mapbiomas-public/assets/brazil/lulc/...`, but those are the **annual**
   LULC / "Deforestation and Secondary Vegetation" / transition **rasters** (Collections
   9/10). That is a *different product*: year-resolved (annual), and already covered by the
   separate `mapbiomas_brasil_annual_lulc` dataset. It does **not** carry the per-event
   before/after image dates that make Alerta a valid ≤1–2-month dated change dataset, and
   using the annual raster as a "change" label would itself fail the §5 change-timing rule
   (year-resolved → reject). So substituting the public GEE raster is not an acceptable
   stand-in for this manifest entry.
6. **`.env`** holds no MapBiomas Alerta credential (it has NASA Earthdata, Copernicus,
   USGS/M2M, Planet, CDS, GEE service account, and internal S3 keys — none apply here).

Per SOP §8.2, a source gated behind a per-dataset registration portal with no credential in
`.env` and no working unauthenticated/mirror path is `rejected` with `needs-credential`.

## Change-timing assessment (would have been ACCEPTED on data grounds)

The dataset is otherwise an excellent fit and a strong **retry candidate** once a token is
supplied. Each alert record exposes (confirmed from the API schema) the fields that satisfy
the §5 change-timing requirement:

- `imageAcquiredBeforeAt` — acquisition date of the "before" (pre-deforestation) image.
- `imageAcquiredAfterAt` — acquisition date of the "after" (deforested) image.
- `detectedAt` — detection date; `publishedAt` — publication date.
- `geometryWkt` — the refined event polygon (WGS84); `areaHa`, `sources`, `alertCode`.

Because MapBiomas refines each event with **daily 3.7 m PlanetScope** imagery, the
before/after pair is typically only weeks–a couple of months apart — i.e. the change date is
knowable to within ~1–2 months for most alerts. That makes this a genuine dated CHANGE
dataset per §5: set `change_time` to the midpoint of `[imageAcquiredBeforeAt,
imageAcquiredAfterAt]`, and `time_range` = a 360-day window centered on it. Alerts whose
before/after span exceeds ~1–2 months (or lack a tight pair) would be dropped, not forced
into the yearly scheme.

## Intended processing recipe (for the retry, once credentialed)

1. `signIn(email, password)` with a MapBiomas Alerta account (from `.env` when added) → Bearer
   token; send it as `Authorization: Bearer <token>`.
2. Page the `alerts` query over a bounded set of `boundingBox` cells across the six Brazilian
   biomes (spatially-stratified, à la `mapbiomas_brasil_annual_lulc`), 2019–2025, requesting
   `geometryWkt`, `imageAcquiredBeforeAt`, `imageAcquiredAfterAt`, `detectedAt`, `areaHa`,
   `alertCode`, `sources`. Cache raw JSON pages to `raw/mapbiomas_alerta/`.
3. Keep alerts whose before/after span ≤ ~60 days; `change_time` = midpoint; drop the rest.
4. Rasterize each polygon into a ≤64×64 local-UTM 10 m tile (centroid- or representative-point
   centered; `rasterize.rasterize_shapes(..., all_touched=True)`), class **0 = deforestation**,
   outside-polygon = **255 nodata** (positive-only, §5 — no synthetic negatives). Large
   polygons captured as a central 640 m window.
5. `time_range` = ±180 d around `change_time`; `change_time` set on every sample JSON.
6. Geographically-stratified round-robin selection, honoring the **25,000**-sample hard cap
   (`sampling.MAX_SAMPLES_PER_DATASET`); use `multiprocessing.Pool(64)` for the write phase.
7. Write `metadata.json` (task_type=classification, one class `deforestation`), verify per §9.

## Reproduce the rejection check

From any shell (no credential needed — this demonstrates the gate):

```bash
curl -s -X POST https://plataforma.alerta.mapbiomas.org/api/v2/graphql \
  -H 'Content-Type: application/json' \
  -d '{"query":"query($b:[Float!]){alerts(limit:1,boundingBox:$b){collection{alertCode geometryWkt imageAcquiredBeforeAt imageAcquiredAfterAt detectedAt}}}","variables":{"b":[-53.0,-6.0,-52.8,-5.8]}}'
# -> {"errors":[{"message":"Token de acesso inválido","path":["alerts"]}]}
```

To process the dataset later, supply a MapBiomas Alerta account (register free at
<https://plataforma.alerta.mapbiomas.org/sign-up>, confirm the email) — ideally added to
`.env` as `MAPBIOMAS_ALERTA_EMAIL` / `MAPBIOMAS_ALERTA_PASSWORD` — then
run the recipe above. Nothing else blocks this dataset; it is a clean retry once credentialed.
