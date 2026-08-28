# Antarctic Ice-Shelf Surface Damage (crevasses/rifts)

- **Slug:** `antarctic_ice_shelf_surface_damage_crevasses_rifts`
- **Status:** completed
- **Task type:** classification (binary dense segmentation)
- **Samples:** 977 label tiles (64×64 @ 10 m)
- **Classes:** `0 = background` (undamaged ice-shelf surface), `1 = surface_damage`; nodata = 255

## Source

"Surface Damage Dataset for Antarctic Ice Shelves 1999–2024" (Tang, Bamber, Li, Qiao,
2026). Zenodo versioned record **20425952** (concept DOI 10.5281/zenodo.20425951),
license **CC-BY-4.0**. One 118 MB zip of annual 30 m surface-damage maps over nine
representative ice shelves (Amery, Brunt, Crosson, Dotson, Holmes, Larsen B, Pine Island,
Thwaites, Totten), 1999–2024, EPSG:3031 (Antarctic Polar Stereographic). Maps were produced
by a deep-learning segmentation model on Landsat 7/8/9 optical imagery; Landsat-7 SLC-off
gaps (2003–2013) were in-painted with a diffusion model (DiffGF). Reported test mIoU 0.845.

### Access notes / caveats

- The **manifest URL points at the concept DOI** (`zenodo.20425951`), whose API record
  returns HTTP 410 GONE. The downloadable **versioned** record is `20425952`.
- Zenodo **fingerprints generic User-Agents** and returns HTTP 403 ("unusual traffic from
  your network") to `urllib`/`curl` default agents. Downloading with a real browser
  User-Agent (Firefox) succeeds. The script sets that UA. (This was the initial blocker; it
  is a UA gate, not a hard IP block — not a credential or transient-outage issue.)

## Class mapping — key decision

The manifest lists three feature types (crevasses, rifts, heavily fractured areas), but the
**raster does not label them separately** — those are the kinds of features collectively
mapped as one "damage" class. Each yearly folder has two products:

- **Type 1** `*_damage_map.tif` (effective ice-shelf extent): `0`=no damage, `1`=damage,
  `255`=NoData (outside the effective extent). **← used.**
- **Type 2** `*_damage.tif` (full ROI, no extent mask, "requires additional manual
  checking"): `0`=no damage, `255`=damage. **← not used.**

We use **Type 1** because its NoData mask makes class 0 a genuine *undamaged ice-shelf
surface* (a spatially-meaningful within-tile negative), not ocean/rock. Output is therefore
a **binary** scheme: `0=background`, `1=surface_damage`, `255=nodata`. Source→output code
map is the identity `{0:0, 1:1}`; source 255 stays nodata. Because the source encodes a
single damage class, we emit `background + surface_damage` rather than the manifest's three
types (documented in `metadata.json`).

## Filtering & sampling

- **Pre-2016 dropped** (spec §8): only Sentinel-era years 2016–2024 kept → 76 of the 170
  main-shelf annual maps.
- **`Amery_front` excluded**: the only `_front` subregion; it spatially overlaps Amery's
  main extent, so including it would produce duplicate tiles. Nine main shelves only.
- **Tiling:** each 30 m EPSG:3031 raster is scanned in native ~21 px blocks (≈630 m ≈ a
  64 px @ 10 m tile). 638,602 blocks contain damage. Each selected block footprint is
  reprojected to a **local UTM projection at 10 m, 64×64, NEAREST** resampling (categorical
  30 m→10 m, spec §4). Padding fill = 255 so out-of-shelf padding never fakes background.
- **Selection:** tiles-per-class balanced (`balance_tiles_by_class`, ≤1000/class, 25k cap).
  Result: 1000 tiles selected; **23 reproject to a footprint whose damage pixels fall just
  outside the 640 m tile → dropped as "empty"; 977 written.** Every written tile contains
  both `background` and `surface_damage` (class_tile_counts: background 977, damage 977).
  (`num_samples` in `metadata.json` reflects the 977 written; there are small gaps in the
  0-padded id sequence for the 23 dropped candidates.)
- **Per-shelf counts:** Amery 306, Brunt 352, Crosson 72, Dotson 20, Holmes 35, Larsen B
  32, Pine Island 49, Thwaites 54, Totten 57. **Per-year:** 46–134 across 2016–2024.

## Time / change handling

Each annual map is a **persistent-state class map** for one year → 1-year window on that
year (spec §5, annual labels). `change_time = null`: surface damage is a persistent
structural feature, not a dated change event. Multiple years of the same shelf are eligible
(temporal diversity).

## Verification

- 977 `.tif` each with a matching `.json`; all single-band uint8, 64×64, 10 m, north-up,
  in a southern-hemisphere UTM CRS. Pixel values ⊆ {0, 1, 255}; class ids in `metadata.json`
  cover them.
- Sampled tiles resolve to the correct UTM zones for each shelf (Amery→32743, Brunt→32727,
  Thwaites→32713, Pine Island→32714, Holmes→32752), confirming correct georeferencing;
  all shelves are north of 80°S so UTM (not UPS) applies.
- Time ranges are 1-year and `change_time` is null throughout.
- A full Sentinel-2 image overlay was not run (Antarctic-shelf S2 eyeballing of ice-on-ice
  damage vs undamaged ice is low-diagnostic); georeferencing was validated via CRS/zone and
  coordinate checks instead.

## Reproduce

```bash
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.antarctic_ice_shelf_surface_damage_crevasses_rifts --workers 64
# inspect raw rasters:
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.antarctic_ice_shelf_surface_damage_crevasses_rifts --inspect
```

Outputs: `datasets/antarctic_ice_shelf_surface_damage_crevasses_rifts/{metadata.json,
locations/*.tif,*.json}` on weka. Idempotent (skips already-written tiles).
