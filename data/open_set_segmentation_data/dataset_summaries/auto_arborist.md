# Auto Arborist

- **Slug:** `auto_arborist`
- **Manifest name:** Auto Arborist
- **Family / label_type:** tree_species / points
- **Source:** Google / CVPR 2022 — https://google.github.io/auto-arborist/
- **Region / time:** 23 US/Canada cities, 2016–2020. ~2.6M trees, >320 tree genera.
- **License:** public (imagery terms apply).
- **Status: REJECTED** — fundamental reason: **phenomenon not observable at 10–30 m**
  (individual urban street trees), and no aggregate/mask representation salvages it.

## What the source is

Auto Arborist is a large-scale **multiview fine-grained urban tree-genus benchmark**
assembled from the public tree censuses (inventories) of 23 North American cities, paired
with **Google Street View (ground-level) + aerial** imagery. Each label is a single
**street/park tree** at a point location, tagged with its **genus** (>320 genera; a
long-tailed distribution). The task the benchmark is built for is: given ground + aerial
imagery of a specific tree, predict its genus.

On paper the label semantics fit the sparse-point classification recipe (§2a/§4): a
(lon, lat, genus, year) tuple with a Sentinel-era time range (2016–2020), expressible as a
uint8 class map after applying the top-254-by-frequency cap (§5). So the reject is **not**
about label semantics, time range, or georeferencing.

## Why it is rejected

**The label is not observable from the sensors OlmoEarth pretraining uses (Sentinel-2 /
Sentinel-1 / Landsat, 10–30 m).** This is the exact case the spec calls out as a
rejection: *"Phenomenon not observable at 10–30 m from S2/S1/Landsat (individual small
trees, …), and no aggregate/mask representation salvages it"* (§8), and the reason
individual street-tree genus needs the observability triage.

- **Sub-pixel, urban-embedded targets.** An individual street tree occupies far less than
  one 10 m Sentinel-2 pixel (typical crown diameter is a few metres), and that pixel is
  dominated by the surrounding **impervious urban matrix** — road, sidewalk, parking lot,
  rooftops. There is no spectral/temporal signal at 10–30 m from which the *genus* of one
  street tree could be recovered; often the tree is not even the majority land cover of its
  own pixel.
- **No natural-habitat proxy.** For scattered natural/forest occurrences a point can act as
  a *weak habitat* label because the surrounding vegetation correlates with the target (this
  is why `globalgeotree` was accepted as a weak label). Urban street trees have **no such
  proxy** — the context is pavement and buildings, not the tree's ecological setting — so
  the weak-label rationale does not transfer.
- **No aggregate/mask salvage.** Urban trees are sparse, scattered single points along
  streets, not a contiguous canopy, so we cannot map a "dominant genus" over a 64×64
  (640 m) tile or build a meaningful genus mask. Aggregating to genus density/richness
  would (a) discard the actual class label and (b) still not be robustly observable at
  10–30 m in dense urban pixels.
- **The benchmark itself confirms this.** Auto Arborist exists precisely because tree genus
  is *not* determinable from satellite views — it fuses ground-level Street View with aerial
  imagery to get the resolution/viewpoint needed. That design is direct evidence the label
  is a VHR/ground-level phenomenon, outside the 10–30 m regime.

## Judgment calls / alternatives considered

- **Weak-label acceptance (à la `globalgeotree`/`geolifeclef_geoplant`):** rejected — those
  precedents lean on a natural-vegetation habitat signal that urban street-tree points lack.
- **Aggregate genus-density / canopy tile:** rejected — sparse scattered points do not form
  a mappable canopy at 10 m, and it would abandon the genus label the dataset provides.
- **Coarser class set (e.g. broadleaf vs conifer):** still an individual-tree property that
  is sub-pixel and urban-embedded; not resolvable at 10–30 m. Not pursued.
- This is a **permanent property** of the phenomenon vs. our sensor resolution, so it is
  `rejected` — not `temporary_failure` (no transient/infra issue) and not
  `needs-credential`. Access was therefore not pursued (the Auto Arborist release also gates
  the multiview *imagery* behind terms, but we never reached that step because the label is
  unusable for our purpose regardless of access).

## Reproduce / verify the rejection

- Project + data terms: https://google.github.io/auto-arborist/
- Paper: Beery et al., "The Auto Arborist Dataset: A Large-Scale Benchmark for Multiview
  Urban Forest Monitoring Under Domain Shift," CVPR 2022. The multiview (Street View +
  aerial) design documents that genus recognition requires sub-metre / ground-level
  imagery, i.e. it is not observable at 10–30 m.

No outputs written to weka `datasets/` beyond the required `registry_entry.json`.
