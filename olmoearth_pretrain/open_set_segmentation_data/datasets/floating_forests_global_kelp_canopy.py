"""Triage Floating Forests (Global) Kelp Canopy for open-set segmentation.

Source: Zooniverse "Floating Forests" citizen-science project, served openly (no login,
CC-BY-4.0) via the IMAS geoserver as the WFS layer ``imas:TRB_FloatingForests``:

  metadata: https://metadata.imas.utas.edu.au/geonetwork/srv/metadata/554ef3f6-4f05-4e40-bbf5-1e6dd31d920c
  WFS:      https://geoserver.imas.utas.edu.au/geoserver/imas/wfs
            typeNames=imas:TRB_FloatingForests  (GeoJSON, EPSG:4326)

Each feature is a consensus outline of surface giant-kelp (Macrocystis pyrifera) canopy,
drawn by Zooniverse volunteers on 30 m Landsat scenes. Per scene there are nested
polygons at multiple ``threshold`` values (minimum number of users who marked a pixel as
kelp), plus a ``scene_timestamp`` giving the Landsat acquisition date.

TRIAGE OUTCOME: REJECT (temporal).
  The openly-downloadable IMAS layer at the manifest URL is California-only and every one
  of its 15,276 features comes from Landsat scenes acquired in 1999-2002 and 2013 -- i.e.
  entirely PRE-2016, outside the Sentinel era. The AGENT_SUMMARY spec lists "temporal
  coverage entirely pre-2016 (outside Sentinel era) with no usable window" as an explicit
  rejection reason.

  Surface kelp canopy is among the most temporally dynamic marine habitats: strong
  seasonal cycles (summer/autumn peak, winter storm loss) and dramatic interannual
  collapse/recovery (e.g. the 2014-2016 NE Pacific marine heatwave removed >90% of
  northern-California canopy). A canopy polygon mapped from a 1999-2013 Landsat scene
  therefore does NOT indicate kelp presence at that location in the Sentinel era, so the
  labels cannot be validly relocated onto a 2016+ imagery window. The labels are also
  intrinsically specific-image (per-scene, per-date) labels, so they can only be paired
  with imagery from their own acquisition date -- all of which is pre-Sentinel-2.

  The data is openly accessible, so this is NOT a credential rejection.

This script is idempotent: it fetches the WFS year distribution (cached to raw/), verifies
that all features are pre-2016, then records the rejection registry entry + summary. It
writes NO label outputs to datasets/{slug}/locations.

Run (from repo root):
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.floating_forests_global_kelp_canopy
"""

import json
import urllib.request
from collections import Counter

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "floating_forests_global_kelp_canopy"
NAME = "Floating Forests Global Kelp Canopy"

WFS_BASE = "https://geoserver.imas.utas.edu.au/geoserver/imas/wfs"
LAYER = "imas:TRB_FloatingForests"
METADATA_URL = (
    "https://metadata.imas.utas.edu.au/geonetwork/srv/metadata/"
    "554ef3f6-4f05-4e40-bbf5-1e6dd31d920c"
)

SUMMARY_PATH = (
    manifest.UPath(  # type: ignore[attr-defined]
        "data/open_set_segmentation_data/"
        "dataset_summaries"
    )
    / f"{SLUG}.md"
)


def _fetch_geojson() -> dict:
    """Download the full WFS GeoJSON to raw/ (idempotent) and return it parsed."""
    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    dst = raw / "TRB_FloatingForests.geojson"
    if not dst.exists():
        url = (
            f"{WFS_BASE}?service=WFS&version=2.0.0&request=GetFeature"
            f"&typeNames={LAYER}&count=20000&outputFormat=application/json"
            "&srsName=EPSG:4326"
        )
        tmp = raw / "TRB_FloatingForests.geojson.tmp"
        with urllib.request.urlopen(url, timeout=600) as r, tmp.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(dst)
    with dst.open() as f:
        return json.load(f)


def main() -> None:
    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    gj = _fetch_geojson()
    feats = gj["features"]
    years = Counter((f["properties"].get("scene_timestamp") or "")[:4] for f in feats)
    n = len(feats)
    max_year = max(int(y) for y in years if y)
    n_2016plus = sum(v for y, v in years.items() if y and int(y) >= 2016)

    print(f"features: {n}")
    print(f"year distribution: {dict(sorted(years.items()))}")
    print(f"max year: {max_year}; features >= 2016: {n_2016plus}")

    assert n_2016plus == 0, (
        "Unexpected: found Sentinel-era (>=2016) features; re-evaluate the triage."
    )

    year_line = ", ".join(f"{y}: {v}" for y, v in sorted(years.items()) if y)
    reason = (
        f"temporal: all {n} openly-available features (IMAS layer imas:TRB_FloatingForests, "
        f"California only) are pre-2016 (latest Landsat scene {max_year}); kelp canopy is highly "
        "dynamic so pre-2016 labels cannot be paired with Sentinel-era imagery -- no usable "
        "2016+ window"
    )

    _write_summary(n, year_line, max_year)

    manifest.write_registry_entry(SLUG, "rejected", notes=reason)
    print("REJECTED:", reason)


def _write_summary(n: int, year_line: str, max_year: int) -> None:
    text = f"""# Floating Forests Global Kelp Canopy -- REJECTED (temporal)

- **Slug**: `{SLUG}`
- **Source**: Zooniverse "Floating Forests" citizen-science project, served openly (no
  login, CC-BY-4.0) via the IMAS geoserver.
  - Metadata record: {METADATA_URL}
  - WFS: `{WFS_BASE}` , layer `{LAYER}` (GeoJSON, EPSG:4326)
- **Label type**: polygons (consensus outlines of surface giant-kelp *Macrocystis
  pyrifera* canopy, drawn by volunteers on 30 m Landsat scenes).
- **Access**: fully open, no account required. **Not a credential rejection.**

## Decision: REJECT

The openly-downloadable IMAS layer at the manifest URL is **California only** and all
**{n} features** are derived from Landsat scenes acquired **entirely pre-2016**:

    scene_timestamp year distribution -> {year_line}
    latest acquisition year = {max_year}; features in the Sentinel era (>=2016) = 0

The AGENT_SUMMARY spec (Section 8, triage) lists *"temporal coverage entirely pre-2016
(outside Sentinel era) with no usable window"* as an explicit rejection reason.

**Why there is no usable window:** surface kelp canopy is among the most temporally
dynamic marine habitats -- strong seasonal cycles (summer/autumn peak, winter storm loss)
and dramatic interannual collapse/recovery (e.g. the 2014-2016 NE Pacific marine heatwave
removed >90% of northern-California canopy). A canopy polygon mapped from a 1999-2013
Landsat scene does **not** indicate kelp presence at that location in 2016+, so the labels
cannot be relocated onto a Sentinel-era imagery window. The labels are also intrinsically
specific-image (per-scene, per-date) labels: each is tied to one Landsat acquisition and
could only be paired with imagery from that same pre-Sentinel-2 date.

## Data characteristics (for reference, had it been in the Sentinel era)

- 15,276 MultiPolygon features across 413 Landsat scenes.
- Per-scene nested polygons at multiple `threshold` levels (minimum number of volunteers
  who classified a pixel as kelp) -- a consensus/confidence axis; a single mid threshold
  per scene would be chosen to avoid nested duplicates.
- Fields: `global_fid, threshold, zooniverse_id, scene, classification_count, image_url,
  tile corner lon/lat, scene_timestamp, created_at, geom`.
- Binary target would have been: 0 = background, 1 = kelp_canopy; polygons rasterized to
  <=64x64 UTM 10 m tiles, plus background-only negative tiles.

## Note on the manifest entry

The manifest lists `time_range: [2016, 2019]` and `region: California, Tasmania,
Falklands, others`. The specific openly-available IMAS layer at the manifest URL does not
match this: it is California-only, 1999-2013 (30 m Landsat). The broader global Floating
Forests product (Tasmania, Falklands) is likewise Landsat-based (30 m, Landsat 5/7/8/9)
and is distributed via kelpwatch.org, not this record.

## QUESTION FOR USER

If OlmoEarth pretraining pairs labels with **same-date Landsat** imagery (Landsat 5/7 for
1999-2002, Landsat 8 for 2013) rather than strictly Sentinel-era imagery, this dataset is
recoverable: ~413 scenes could be processed (pick one consensus `threshold` per scene,
rasterize kelp=1 into <=64x64 UTM 10 m tiles with per-scene specific-date time ranges,
plus background negatives). Please confirm whether pre-2016 Landsat-dated labels are in
scope, and/or point to a Sentinel-era (2016+) Floating Forests / kelp-canopy release if
you prefer one; I can then process it.

## Reproduce

    python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.floating_forests_global_kelp_canopy
"""
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_PATH.open("w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
