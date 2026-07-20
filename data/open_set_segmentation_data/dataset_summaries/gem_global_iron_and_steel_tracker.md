# GEM Global Iron and Steel Tracker (GIST)

- **Slug:** `gem_global_iron_and_steel_tracker`
- **Status:** completed · classification (presence-only points) · **986 points**
- **Source:** Global Iron and Steel Tracker (GIST), Global Energy Monitor (GEM), June 2026 (V1)
  release. <https://globalenergymonitor.org/projects/global-iron-and-steel-tracker/download-data/>
- **License:** CC-BY-4.0
- **Annotation method:** authoritative/expert asset-level inventory (company filings, government
  data, satellite imagery, news); coordinates carry an exact/approximate accuracy flag.

## Source & access

Asset-level inventory of the world's crude iron/steel plants (≥ 500,000 t/yr), 1,293 plants in
the June-2026 release, each with a geocoded point, a `Coordinate accuracy` flag, a `Start date`
(commissioning year), furnace mix, and per-unit operating status. GIST is distributed behind a
lightweight web download form (name/email/use-case), **not** a credential gate; the script
reproduces the `mint_submission`→`presign` flow via
`download.download_gem_tracker(["iron-steel-plant-tracker"], …)`. Only the label spreadsheet
(~0.72 MB) is downloaded; no imagery. See `raw/{slug}/SOURCE.txt`.

## Label type — presence-only points

**Converted from the old positive-only object-detection tile encoding** (64×64 buffer+negative
tiles). Now emitted as **presence-only points** in a dataset-wide `points.geojson` (spec §2a):
each included plant is one point of the single foreground class. There is **no fabricated
GeoTIFF context, and no background / buffer / negative tiles** — this dataset carries **no
fabricated negatives**; negatives are supplied downstream by the assembly step.

## Classes / counts / inclusion

Single class `0 = steel/iron plant`. **986 points** (up to 1000/class, `balance_by_class`).

Kept a plant iff it has ≥ 1 unit with status ∈ {operating, operating pre-retirement, mothballed,
mothballed pre-retirement} **and** an `exact` (satellite-confirmed) coordinate → **986 of
1,293**. Dropped announced/cancelled (not built), construction-only, retired-only (may be
demolished / no time anchor), and `approximate` coordinates.

## Time handling

Presence/state, not change: `Start date` is year-granular only (coarser than the ~1–2 month
change-timing rule), so `change_time = null` and each point gets a 1-year window in
`[clamp(start, 2017, 2025), 2025]`, spread deterministically per plant (md5 of GEM id) for
imagery diversity. `start` < 2017 or unknown → `[2017, 2025]`.

## Output

- `datasets/gem_global_iron_and_steel_tracker/points.geojson`
- `datasets/gem_global_iron_and_steel_tracker/metadata.json`

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.gem_global_iron_and_steel_tracker \
  --contact-name "<your name>" --contact-email "<your email>" --contact-organization "<your org>"
```

Idempotent; re-downloads the xlsx only if missing.

## Caveats

- GIST omits < 500 ktpa mini-mills; precise negatives are now the assembly step's responsibility.
- 152 "approximate" plants and all retired/construction/announced plants excluded.
