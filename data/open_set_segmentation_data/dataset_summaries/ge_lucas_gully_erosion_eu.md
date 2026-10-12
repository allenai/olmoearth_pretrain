# GE-LUCAS Gully Erosion (EU)

- **Slug**: `ge_lucas_gully_erosion_eu`
- **Task type**: classification (sparse point segmentation)
- **Samples**: 3,446 points (`points.json`, spec 2a — pure 1x1 point dataset, no per-point GeoTIFFs)
- **Region**: European Union + UK. **Year**: 2022 (1-year time range anchored on 2022).

## Source and access

EC JRC / ESDAC "Gully erosion in the EU" (Borrelli, Matthews, Alewell, Kaffas, Poesen,
Saggau, Prăvălie, Vanmaercke, Panagos 2025, *Nature Scientific Data* 12, 755). The ESDAC
download portal (https://esdac.jrc.ec.europa.eu/content/gully-erosion-eu) is
registration-gated (free, but requires a login form we cannot complete). **However, the
identical dataset is deposited openly on Figshare** (DOI `10.6084/m9.figshare.27211473`),
so it was pulled from there and **no credential rejection was needed**.

The Figshare deposit has 8 files; we use **Data 1 - LUCAS2022 original** (the full LUCAS
2022 feature-detection survey of 399,591 grid locations). This CSV carries WGS84
`POINT_LAT`/`POINT_LONG` for every point plus the gully-erosion attributes directly, so a
single file yields both presence and erosion type without a spatial join.

## Label mapping

The survey visited 399,591 EU/UK monitoring-grid locations (49.8% in-situ, 50.2%
on-screen). Two source fields drive the label:
- `SURVEY_GULLY_SIGNS`: 1 = gully channel present (3,116 pts), 2 = absent (396,475 pts).
- `SURVEY_GULLY_TYPE` (present points only): 1 ephemeral, 2 permanent, 3 badlands.

We fold gully presence and erosion class into one unified 4-class scheme:

| id | name | source | raw count |
|----|------|--------|-----------|
| 0 | No gully channel | SIGNS=2 | 396,475 |
| 1 | Ephemeral gully (<0.5 m deep) | TYPE=1 | 656 |
| 2 | Permanent gully (0.5-30 m deep) | TYPE=2 | 1,670 |
| 3 | Badlands (gullied landscape) | TYPE=3 | 790 |

## Sampling

`balance_by_class(per_class=1000, total_cap=25000)`. Selected counts: class 0 = 1000
(down-sampled from 396,475), class 2 = 1000 (from 1,670), class 3 = 790 (all), class 1 =
656 (all). **Total 3,446**, well under the 25k cap. All source records are fair game (no
split filtering).

## Time range

Static 2022 survey -> 1-year window [2022-01-01, 2023-01-01). No change labels.

## Caveats

- The companion **gully-occurrence probability raster** (Data 4, ETRS89, 100 m,
  Random-Forest interpolation) is the manifest's second "class" but is a derived-product
  regression map, not point reference data; per the point-only instruction it is **not**
  encoded here. It could later be added as a separate dense-regression dataset if desired.
- Absent-class (0) points are abundant and were randomly down-sampled to 1000; they cover
  all of Europe, so they act as diverse negatives.
- Gully channels are typically a few metres wide and may be near/below S2's 10 m
  resolution at a single pixel; the point marks the surveyed location (badlands and
  permanent gullies are the more resolvable classes).

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ge_lucas_gully_erosion_eu
```
Idempotent: re-downloads the Figshare zip to `raw/` only if missing, then rewrites
`points.json` + `metadata.json`.
