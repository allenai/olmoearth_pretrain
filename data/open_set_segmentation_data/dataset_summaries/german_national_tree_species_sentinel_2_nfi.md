# German National Tree Species (Sentinel-2 + NFI)

- **Slug:** `german_national_tree_species_sentinel_2_nfi`
- **Manifest name:** German National Tree Species (Sentinel-2 + NFI)
- **Family / label_type:** tree_species / points
- **Source:** ESSD 17, 351–367, 2025 (doi:10.5194/essd-17-351-2025). Data DOI 10.3220/DATA20240402122351-0 (OpenAgrar, Thünen Institute). License CC-BY-4.0.
- **Status: REJECTED** — fundamental reason: **no recoverable geocoordinates (coordinate-fuzzed points).**

## What the source is

A machine-learning training dataset for tree-species classification in Germany. It is
built from the German National Forest Inventory (NFI, "Bundeswaldinventur") 2012 field
plots: 387,775 upper-canopy trees (~360k after filtering) plus ~70k non-tree locations.
For each tree, the authors extracted the matching Sentinel-2 L2A bottom-of-atmosphere
reflectance time series (10 bands, signed int16), FORCE-processed, spanning ~July 2015 to
end of October 2022, yielding ~83 million spectral data points. Labels cover 48 tree
species + 3 species groups (spruce, pine, fir, Douglas fir, larch, beech, oak, birch, …).

The task fit would otherwise be good: a genuine per-pixel species-classification label in
the Sentinel-2 era, expressible as a uint8 class map, in the points recipe (§2a/§4).

## Why it is rejected

The dataset **does not publish placeable geocoordinates.** Per the paper and the data
landing page, the exact NFI sampling-unit and individual-tree positions are **confidential**
(German NFI legal confidentiality). As the only geolocation reference, the authors publish
**the center coordinate of the closest 1 km INSPIRE grid cell** to each sampling unit — not
the plot/tree location. The true tree can lie anywhere within that 1 km cell (up to ~700 m
from the published point).

For this effort we must co-locate a label with imagery on the 10 m Sentinel-2 grid. A
single-pixel dominant-tree-species label at ~1 km positional uncertainty cannot be pinned
to any specific 10 m pixel, and German forests are far too heterogeneous at the 1 km /
64×64-tile (640 m) scale for an aggregate/mask representation to salvage it — the dominant
species at the plot is not the dominant species across the whole cell. This is the exact
situation the spec calls out as a rejection: "coordinate-fuzzed points like FIA ~1 mi …
and no aggregate/mask representation salvages it" (§8), i.e. **no recoverable
geocoordinates** (§1a fundamental reason).

The authors deliberately alleviate the confidentiality limitation by publishing the
**extracted Sentinel-2 spectra themselves** instead of coordinates. That makes the dataset
usable for pixel-classifier training directly on spectra, but gives us no georeferenced
geometry to attach a label raster/point to independent imagery — which is what OlmoEarth
pretraining assembly requires.

## Judgment calls

- Considered treating the 1 km INSPIRE cell center as a coarse location and emitting a
  large tile: rejected because the tile cap (64×64 = 640 m) is smaller than the 1 km
  uncertainty and forest species composition is not homogeneous at that scale, so the label
  would frequently not correspond to the pixels.
- Confirmed no alternate file with exact lat/lon exists (coordinates are legally
  confidential, not merely omitted), so this is not a transient/credential issue — it is a
  permanent property of the release. Hence `rejected`, not `temporary_failure` or
  `needs-credential`.

## Reproduce / verify the rejection

- Paper: https://essd.copernicus.org/articles/17/351/2025/ — see the "Data availability"
  and confidentiality discussion (INSPIRE 1 km grid as geolocation reference).
- Data: https://doi.org/10.3220/DATA20240402122351-0 (redirects to OpenAgrar record
  openagrar_mods_00094435). Each record's coordinate is the 1 km INSPIRE cell center, not
  the plot.

No outputs written to weka `datasets/` beyond the required `registry_entry.json`.
