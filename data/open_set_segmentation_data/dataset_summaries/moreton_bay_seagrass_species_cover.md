# Moreton Bay Seagrass Species & Cover

- **slug**: `moreton_bay_seagrass_species_cover`
- **status**: **rejected** — all labels are pre-2016 (outside the Sentinel-2 era);
  fundamental, not a retry candidate
- **task_type** (intended, had it been usable): classification (dominant seagrass species)
  and/or regression (seagrass percent cover) as sparse points
- **num_samples**: 0

## Source

- Manifest name: `Moreton Bay Seagrass Species & Cover`
- Source: PANGAEA / Scientific Data. Parent DOI
  <https://doi.org/10.1594/PANGAEA.846147> (Roelfsema, Kovacs, Lyons, Phinn 2015),
  supplement to *Scientific Data* 2, 150040, <https://doi.org/10.1038/sdata.2015.40>.
- Description: point-based seagrass species composition and percent cover derived from
  ~3,000 manually interpreted downward-looking photo-transect images per survey, over the
  Eastern Banks, Moreton Bay, Australia.
- Family: seagrass; region: Moreton Bay, Australia; label_type: `points`;
  annotation_method: manual photo-interpretation (Coral Point Count / CPCe); license:
  CC-BY-3.0; have_locally: false.
- Manifest classes: Halophila, Halodule, Zostera, Cymodocea, Syringodium, percent cover.
- **Manifest `time_range` is `[2016, 2016]`, which is incorrect** — see below.

## What the labels actually are

The parent DOI is a **dataset publication series of 9 child datasets**, one per survey
campaign. Each child is a tab-delimited point table where every row is one georeferenced
benthic photo (lon/lat + UTM easting/northing/zone, WGS84 UTM Zone 56S), carrying per-
species percent-cover columns (*Cymodocea serrulata*, *Halodule uninervis*, *Halophila
ovalis*, *Halophila spinulosa*, *Syringodium isoetifolium*, *Zostera muelleri*, plus
epiphyte/algae variants) and total seagrass % cover, plus many macroalgae/substrate
categories. Coordinates are recoverable and the label semantics (dominant species class /
% cover value at a 10 m pixel) would map cleanly to the sparse-point table format (§2a).

## Why rejected — pre-2016 rule (SOP §2 / §8.2)

The survey campaigns span **2004-07 → 2015-06** — **every** label is pre-2016:

| child DOI | survey |
|---|---|
| 846264 | 2004-07 |
| 846142 | 2007-08 |
| 846143 | 2011-06 |
| 846144 | 2012-02 |
| 846146 | 2012-06 |
| 846185 | 2013-02 |
| 846186 | 2013-05 |
| 846266 | 2014-07 |
| 867188 | 2015-06 |

Verified directly from PANGAEA: the parent record's coverage is `DATE/TIME START:
2004-07-28 … END: 2015-06-17`, and the most recent child (867188, "2015-06") downloaded as
tab-delimited text has `DATE/TIME START: 2015-06-15 … END: 2015-06-17`. There is **no
post-2016 survey** in the series (the abstract notes the 2015 campaign was the final
addition to the collection).

SOP §8.2: *"Pre-2016 labels: reject if ALL labels fall before 2016 (outside the Sentinel
era, with no usable post-2016 window). … Landsat-era-only labels are still rejected under
this rule — we anchor to the Sentinel era."* The latest survey (June 2015) precedes
routine Sentinel-2 availability, so no post-2016 window can be assigned to any sample.
This makes the dataset a **fundamental `rejected`** (entirely pre-Sentinel-era), not
`temporary_failure` (the PANGAEA source is fully accessible, open, CC-BY, no credential
needed) and not `needs-credential`.

## Secondary note — observability at 10–30 m

Even setting the date aside, this dataset would have been marginal under the observability
triage the task flags for submerged seagrass. The photos are taken at 0.5–2.5 m depth in
"shallow, clear water," so seagrass **presence/% cover** in the Eastern Banks is at least
partially observable to S2/Landsat blue–green bands (this site is in fact a canonical
satellite seagrass-mapping location). However, **species-level discrimination**
(Halophila vs. Halodule vs. Zostera vs. Cymodocea vs. Syringodium) at a 1 m photo footprint
is well below a 10 m pixel and is not reliably retrievable — the spec explicitly lists
"fine coral/seagrass zonation" as potentially unresolvable at 10 m (§4). The % cover
regression target would have been the more defensible signal. This is recorded as a caveat
only; the **decisive** rejection ground is pre-2016.

## Judgment calls

- Rejected on **pre-2016** (the primary, non-retryable ground) after verifying actual
  record dates from the authoritative PANGAEA source rather than trusting the manifest's
  erroneous `[2016, 2016]` time range.
- Did not download the remaining 8 child tables or write any label outputs: no post-2016
  data exists, so no sample could satisfy the ≤1-year Sentinel-era time-range rule.
- Not `temporary_failure`: PANGAEA served the data fine (HTTP 200, open CC-BY); the block
  is fundamental (temporal), not transient.

## Reproduce

No outputs were written to weka `datasets/` beyond `registry_entry.json`. To re-examine:

```bash
# parent series landing page
curl -sL "https://doi.pangaea.de/10.1594/PANGAEA.846147"
# any child as tab-delimited text (latest = 2015-06)
curl -sL "https://doi.pangaea.de/10.1594/PANGAEA.867188?format=textfile"
```

The dates (2004–2015, all pre-2016) are stable and the rejection stands regardless.
