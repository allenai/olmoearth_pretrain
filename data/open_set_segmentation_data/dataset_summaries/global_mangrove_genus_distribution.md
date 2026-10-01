# Global Mangrove Genus Distribution

- **Slug:** `global_mangrove_genus_distribution`
- **Status:** completed
- **Task type:** classification (sparse points, label_type `points`)
- **Num samples:** 472 point labels
- **Source:** Twomey, A.J. & Lovelock, C.E. (2024), "Global spatial dataset of mangrove
  genus distribution in seaward and riverine margins", *Scientific Data* 11, 306.
  DOI [10.1038/s41597-024-03134-1](https://doi.org/10.1038/s41597-024-03134-1).
  Data on PANGAEA: [10.1594/PANGAEA.942481](https://doi.pangaea.de/10.1594/PANGAEA.942481).
- **License:** CC-BY-4.0.

## Access / download

No credentials needed. Two files pulled to `raw/global_mangrove_genus_distribution/`:
- `Genus_Shapefiles.zip` → `FrontalMangroveGenus.shp` (250 polygons) — **not used** (see below).
- `MangroveZonationData.xlsx` — the "Original Data" sheet is the source of the point labels.

Direct URLs:
`https://download.pangaea.de/dataset/942481/files/Genus_Shapefiles.zip`,
`https://download.pangaea.de/dataset/942481/files/MangroveZonationData.xlsx`.

## What the release contains and which product we used

The release ships two products:

1. **`FrontalMangroveGenus` shapefile** — 250 Marine-Ecoregions-of-the-World (MEOW)
   polygons, each tagged with the dominant *frontal* (seaward-margin) mangrove genus
   (91 have a genus; 159 are non-mangrove temperate/polar ecoregions). These polygons are
   **whole marine ecoregions** (median area ~625,000 km², mostly open ocean) and carry no
   mangrove extent. Rasterizing them would paint a single genus over vast non-mangrove
   areas and misrepresent genuinely mixed regions (e.g. the Floridian ecoregion is labeled
   solely *Rhizophora*, though Florida mangroves are mixed *Rhizophora*/*Avicennia*/
   *Laguncularia*). **Not usable as a per-pixel label; discarded.**

2. **`MangroveZonationData.xlsx` "Original Data" sheet** — 733 mangrove-zonation studies
   compiled from 195 publications. Each row has a `Frontal Mangrove Genus`, a
   `Frontal Mangrove Species`, country/location text, a per-record `Latitude`/`Longitude`,
   and a `Location Coordinates` precision flag (`Specific` = precise; `Estimated` =
   inferred from the location name). **This is the georeferenced point product we used**,
   matching the manifest `label_type: points` and description ("georeferenced points
   identifying mangrove genera").

## Processing

- Kept the 473 rows with valid lat/lon; dropped one out-of-band record (Punta Arenas,
  Chile, −53.17°S — no mangroves there; a mis-estimated coordinate). **472 points remain.**
- Latitude band filter: `−40°..+33°` (global mangrove range).
- Label = observed dominant frontal mangrove genus at each site. Genus names normalized
  (fixed source typos: `Aviennia`→`Avicennia`, `Brugueira`→`Bruguiera`,
  `Luminitzera`→`Lumnitzera`, trailing-whitespace variants merged).
- **22 genera**, ordered by frequency → class ids 0..21 (well under the 254 uint8 cap).
  No per-class truncation (max 222 < 1000/class; total 472 ≪ 25k cap).
- Species kept in each point's `properties` for reference but **genus is the class** (the
  manifest target; species is even less observable at 10 m).
- Output: one dataset-wide `points.geojson` (spec §2a); no per-point GeoTIFFs (1×1 sparse
  points). Each feature carries `label` (genus id), `genus`, `species`, `coord_precision`,
  `source_id` (`zonation_row_<n>`), and `time_range`.

## Class distribution (472 points)

| id | genus | n | | id | genus | n |
|----|-------|---|---|----|-------|---|
| 0 | Rhizophora | 222 | | 11 | Excoecaria | 1 |
| 1 | Avicennia | 137 | | 12 | Aegialitis | 1 |
| 2 | Sonneratia | 50 | | 13 | Nypa | 1 |
| 3 | Laguncularia | 28 | | 14 | Lumnitzera | 1 |
| 4 | Bruguiera | 9 | | 15 | Pelliciera | 1 |
| 5 | Ceriops | 5 | | 16 | Xylocarpus | 1 |
| 6 | Conocarpus | 3 | | 17 | Drepanocarpus | 1 |
| 7 | Cocos | 2 | | 18 | Camptostemon | 1 |
| 8 | Heritiera | 2 | | 19 | Aegiceras | 1 |
| 9 | Pemphis | 2 | | 20 | Acanthus | 1 |
| 10 | Acoelorraphe | 1 | | 21 | Osbornia | 1 |

Dominated by *Rhizophora* + *Avicennia* (76%); many genera are single-sample. Per spec §5
sparse classes are kept — downstream assembly drops classes below its minimum count.

## Time-range handling

Static literature compilation with no per-record observation date. Mangrove forests and
their genus composition are persistent, so a **representative Sentinel-era 1-year window
(2020-01-01..2021-01-01)** is assigned to every point (spec §5 static labels).
`change_time` is null.

## Caveats

- **Coordinate precision:** 68 points are `Specific` (precise); 404 are `Estimated`
  (inferred from location names, so possibly off by kilometres — may land off the exact
  mangrove stand or in adjacent water). Flag stored per-point as `coord_precision` for
  downstream filtering to the precise subset if desired.
- **Genus at 10 m is a weak label:** the point marks a real mangrove site with an observed
  dominant *frontal* genus, but per-pixel genus discrimination from S2/S1/Landsat is
  difficult, and the label describes the seaward-margin dominant (interior/riverine zones
  may differ). Treat as weakly-supervised genus signal.
- The ecoregion-polygon product was intentionally not used (too coarse; see above).
- No exhaustive S2 overlay performed (sparse 1×1 points, many with estimated coordinates);
  latitude-band and coordinate-range sanity checks applied instead.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_mangrove_genus_distribution
```
Idempotent: re-running re-reads the xlsx and atomically overwrites `points.geojson` /
`metadata.json`. Outputs under
`/weka/dfive-default/helios/dataset_creation/open_set_segmentation/datasets/global_mangrove_genus_distribution/`.
