# Precomputed Embedding Coverage

Measured coverage of the precomputed embedding products (AlphaEarth/GSE,
Tessera) across the AEF supplemental eval datasets. **Read this before putting
a Tessera or AEF number in a table next to an OlmoEarth number** — for several
datasets the products cover only a fraction of the windows, and a score on that
fraction is not an estimate of the product's performance on the dataset.

Measured 2026-07-30 from each dataset's
`embedding_materializer_manifest_<product>.json`. Numbers will change if the
materializer is re-run (e.g. to retry failures) or if a product publishes new
years.

## Coverage

"Have" is windows carrying the layer (`written + skipped_existing`); the rest
are coverage gaps, where the product had no data for that window's target year.

| Dataset | Windows | AEF have | AEF | Tessera have | Tessera |
|---|---:|---:|---:|---:|---:|
| africa_crop_mask | 2 556 | 2 556 | 100.0% | 1 529 | **59.8%** |
| canada_crops_coarse | 16 079 | 16 001 | 99.5% ⚠️ | 4 724 | **29.4%** |
| canada_crops_fine | 14 566 | 14 490 | 99.5% | 4 391 | **30.1%** |
| descals | 17 477 | 17 477 | 100.0% | 2 532 | **14.5%** |
| ethiopia_crops | 2 530 | 2 530 | 100.0% | 206 | **8.1%** |
| glance | 34 885 | 34 885 | 100.0% | 13 236 | **37.9%** |
| lcmap_lu | 26 513 | 26 510 | 100.0% | 26 409 | 99.6% |
| us_trees | 45 382 | *materializing* | | *materializing* | |

⚠️ `canada_crops_coarse` AEF also has 26 windows whose fetch failed (as opposed
to having no coverage). Those are recoverable — re-run the materializer for
that dataset and product, which skips windows already written.

**AEF coverage is effectively complete everywhere.** It is a global product for
2018-2024, and every dataset's label years fall inside that range.

**Tessera coverage is poor outside the US.** The published product is global for
2024 only, reaching back to 2017 for the US and EU. The measured fractions are
consistent with that footprint: the non-US datasets land close to whatever
share of their windows carry 2024 labels (Canada ~30%, glance ~38%, descals
~15%, ethiopia_crops ~8%), while `lcmap_lu` — US, so covered back to 2017 —
reaches 99.6%.

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
windows missing any *required* input's layers. So marking `tessera` required on
`ethiopia_crops` would cut that dataset to 8% of its windows for the OlmoEarth
and AEF evals too, silently invalidating comparison against previously recorded
numbers on it.

This is why `scripts/tools/wire_embedding_modalities.py` gates going live on
`--min_coverage` (default 99%). At that threshold, AEF is live on every dataset
above except `canada_crops_coarse` (held for its 26 failures), and Tessera is
live only on `lcmap_lu`.

## Reporting guidance

- **Do not report a Tessera number for a dataset below the coverage
  threshold.** At 8-38% coverage the comparison is not apples-to-apples, and
  the covered subset is *geographically and temporally biased* — it is
  approximately "the 2024-labelled windows", not a random sample. A Tessera
  score on 8% of `ethiopia_crops` says something about Tessera on 2024 Ethiopian
  cropland, not about Tessera on `ethiopia_crops`.
- **Where a product is live, state the covered fraction** if it is not ~100%.
  `lcmap_lu` Tessera covers 99.6%; that is worth a footnote, not an asterisk on
  the conclusion.
- **Watch the window set when comparing against older OlmoEarth runs.** Marking
  a product required drops its gap windows for every eval on the dataset, so an
  OlmoEarth number recorded before the wiring is not measured on the same
  windows as one recorded after. On `lcmap_lu` that is 104 of 26 513 windows
  (0.4%) for Tessera and 3 for AEF — small, but re-run the OlmoEarth ws16
  embedding evals if the comparison needs to be exact.
- **Tessera v1 vs v1.1 are separate products** (`tessera` / `tessera_v11`
  modalities) with their own coverage. The table above is v1; re-measure before
  reporting v1.1.

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
    for product in ("aef", "tessera"):
        path = d / f"embedding_materializer_manifest_{product}.json"
        if not path.exists():
            continue
        m = json.load(path.open())
        have = m["num_windows_written"] + m["num_windows_skipped_existing"]
        total = have + m["num_coverage_gaps"] + m["num_windows_failed"]
        print(f"{d.name:22s} {product:8s} {have:6d}/{total:6d} = {have / total:6.1%}")
EOF
```
