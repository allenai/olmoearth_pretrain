# Precomputed Embedding Coverage

Measured coverage of the downloaded precomputed embedding product
(AlphaEarth/GSE) across the AEF supplemental eval datasets. **Read this before
putting an AEF number in a table next to an OlmoEarth number** — a product that
covers only a fraction of a dataset's windows gives a score on that fraction,
which is not an estimate of its performance on the dataset.

Measured 2026-07-30 from each dataset's
`embedding_materializer_manifest_<product>.json`. Numbers will change if the
materializer is re-run (e.g. to retry failures) or if a product publishes new
years.

## Coverage

"Have" is windows carrying the layer (`written + skipped_existing`); the rest
are coverage gaps, where the product had no data for that window's target year.

| Dataset | Windows | AEF have | AEF |
|---|---:|---:|---:|
| africa_crop_mask | 2 556 | 2 556 | 100.0% |
| canada_crops_coarse | 16 079 | 16 001 | 99.5% ⚠️ |
| canada_crops_fine | 14 566 | 14 490 | 99.5% |
| descals | 17 477 | 17 477 | 100.0% |
| ethiopia_crops | 2 530 | 2 530 | 100.0% |
| glance | 34 885 | 34 885 | 100.0% |
| lcmap_lu | 26 513 | 26 510 | 100.0% |
| us_trees | 45 382 | *materializing* | |

⚠️ `canada_crops_coarse` AEF also has 26 windows whose fetch failed (as opposed
to having no coverage). Those are recoverable — re-run the materializer for
that dataset and product, which skips windows already written.

**AEF coverage is effectively complete everywhere.** It is a global product for
2018-2024, and every dataset's label years fall inside that range.

**Tessera is not in this table.** The published Tessera products are global for
2024 only (US/EU back to 2017), so the downloaded layers covered as little as 8%
of the non-US datasets; the release compares against Tessera v2 instead, which
we run ourselves (see below).

## Why partial coverage is not just a smaller sample

Two distinct consequences, and the second is the one that bites silently.

**The product cannot be scored on gap windows.** A window without the layer
either drops out of the dataset (if the model.yaml input is `required: true`) or
reaches `PrecomputedEmbedding.forward` without its modality and raises
(`required: false`). Either way the reported metric comes only from covered
windows.

**A required input shrinks the dataset for *every* model.** There is one
model.yaml per dataset, and `RslearnToOlmoEarthDataset.from_model_config` builds
the underlying rslearn `ModelDataset` from all of its `inputs` before
`input_modalities` selects which the model reads. rslearn then filters out
windows missing any *required* input's layers. So marking a sparsely covered
product required would cut the dataset for the OlmoEarth and AEF evals too,
silently invalidating comparison against previously recorded numbers on it.

This is why `scripts/tools/wire_embedding_modalities.py` gates going live on
`--min_coverage` (default 99%). At that threshold, AEF is live on every dataset
above except `canada_crops_coarse` (held for its 26 failures).

## Reporting guidance

- **Do not report a product's number for a dataset below the coverage
  threshold.** The covered subset is *geographically and temporally biased* —
  approximately "the windows whose label year the product publishes", not a
  random sample — so the comparison is not apples-to-apples.
- **Where a product is live, state the covered fraction** if it is not ~100%;
  a 99.5% fraction is worth a footnote, not an asterisk on the conclusion.
- **Watch the window set when comparing against older OlmoEarth runs.** Marking
  a product required drops its gap windows for every eval on the dataset, so an
  OlmoEarth number recorded before the wiring is not measured on the same
  windows as one recorded after. On `lcmap_lu` that is 3 of 26 513 windows for
  AEF — small, but re-run the OlmoEarth ws16 embedding evals if the comparison
  needs to be exact.
- **`tessera_v2` is not subject to this table.** No v2 product is published, so
  we run the released v2 students ourselves over whatever windows we choose
  (docs/TesseraV2Inference.md) — coverage is 100% by construction, minus any
  windows whose fetch or inference failed, which the run's manifest lists.

## Regenerating these numbers

The wiring script's dry run prints coverage per (dataset, product) without
writing anything:

```bash
python scripts/tools/wire_embedding_modalities.py --dry_run
```

Or read the manifests directly — they also list the specific gap window IDs
under `coverage_gaps`:

```bash
python - <<'EOF'
import json
from pathlib import Path
for d in sorted(Path("/weka/dfive-default/olmoearth/eval_datasets").iterdir()):
    for product in ("aef",):
        path = d / f"embedding_materializer_manifest_{product}.json"
        if not path.exists():
            continue
        m = json.load(path.open())
        have = m["num_windows_written"] + m["num_windows_skipped_existing"]
        total = have + m["num_coverage_gaps"] + m["num_windows_failed"]
        print(f"{d.name:22s} {product:8s} {have:6d}/{total:6d} = {have / total:6.1%}")
EOF
```
