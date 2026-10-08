# Seabird Colony Registers (Circumpolar & N. Pacific)

- **slug**: `seabird_colony_registers_circumpolar_n_pacific`
- **status**: **rejected** — phenomenon not observable at 10–30 m (fundamental; not a
  retry candidate)
- **task_type** (intended, had it been usable): classification (colony species)
- **num_samples**: 0

## Source

- Manifest name: `Seabird Colony Registers (Circumpolar & N. Pacific)`
- Source: CBird / CAFF ABDS (Circumpolar Seabird Expert Group / Arctic Biodiversity Data
  Service), portal <https://geo.abds.is/geonetwork/caff/search>
- Description: downloadable seabird breeding-colony locations and population counts across
  the Arctic and N. Pacific.
- Family: wildlife; region: Circumpolar Arctic + N. Pacific; label_type: `points`;
  annotation_method: manual census / expert; license: "open (CAFF)"; manifest time_range:
  2016–2024; have_locally: false.
- Manifest classes (8): murres, puffins, kittiwakes, auklets, fulmars, cormorants, gulls,
  terns.

## What the labels actually are

Each record is a **seabird breeding-colony census location**: a georeferenced point where
one or more seabird species breed, paired with **population counts** (pairs / individuals /
apparently-occupied sites) gathered by manual survey. The "class" attached to a point is
the **bird species (or species group)** that breeds there — murres vs. puffins vs.
kittiwakes, etc. It is a wildlife-census point, not a land-cover map.

## Why rejected — observability at 10–30 m (SOP §8, third bullet)

The labeled quantity — *which seabird species colonizes this point* — is **not a
land-cover phenomenon resolvable from Sentinel-2 / Sentinel-1 / Landsat at 10–30 m**:

- **It is a census location, not a spectrally-distinct surface.** The point marks where a
  survey counted birds. The birds themselves (tens of cm) and the distinction between a
  murre colony and a puffin/kittiwake/auklet colony are far below any 10 m pixel and have
  no separating spectral/backscatter signature. This is the same failure mode as the other
  wildlife-point rejections in this effort.
- **The manifest note ("colony islands/cliffs/guano-stained ground discernible at 10–30 m")
  does not rescue it.** Even where a colony coincides with a mappable feature — a specific
  cliff, islet, or guano-stained slope — **that feature is not the labeled quantity.** A
  cliff or island is generic land cover; guano staining is neither species-specific nor
  reliably present, and the dataset provides no delineated guano/cliff polygon to
  rasterize, only a colony census point. Mapping "island here" or "cliff here" would label
  something the dataset does not actually annotate (the 8 species classes) and would be
  indistinguishable across all 8 classes.
- **No salvageable aggregate/mask.** The strongest thing expressible at 10–30 m would be a
  single weak "seabird colony present here" presence point, which discards the entire
  species class scheme (the only labeled signal) and still points at a census location, not
  a segmentable surface. Per SOP §8 this does not salvage the dataset.

Because even with the data fully in hand the labels cannot be expressed as a meaningful
per-pixel classification or regression at 10–30 m, this is a **fundamental `rejected`**
(SOP §8: "phenomenon not observable at 10–30 m … and no aggregate/mask representation
salvages it"), not `temporary_failure` and not `needs-credential`.

## Access note (secondary, moot)

The CAFF ABDS GeoNetwork (<https://geo.abds.is/geonetwork/caff/search>) does host
open circumpolar seabird colony layers, so the data is plausibly obtainable without
credentials. Access was **not pursued** because the observability failure is fundamental
and independent of whether the points can be downloaded — no download would change the
rejection.

## Reproduce

No outputs are produced. To re-triage: read this file and the manifest entry for
`Seabird Colony Registers (Circumpolar & N. Pacific)`. The rejection is a judgment call
on observability (SOP §8), not on data access; revisit only if the effort's scope changes
to admit wildlife-census presence points as labels.
