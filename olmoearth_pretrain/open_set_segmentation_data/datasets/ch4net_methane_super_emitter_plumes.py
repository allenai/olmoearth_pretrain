"""Triage CH4Net (methane super-emitter plumes) -> REJECTED (needs-credential: gated HF repo).

CH4Net (Vaughan et al., 2024; AMT 17, 2583-2593, https://doi.org/10.5194/amt-17-2583-2024)
is a hand-annotated methane-plume segmentation dataset built on Sentinel-2 imagery over 23
known super-emitter locations in Turkmenistan (2017-2021). It comprises 925 hand-annotated
plume masks plus 9,121 plume-free scenes (10,046 images total). The paper's data/code
availability statement points to a SINGLE distribution:

    "Code and hand-annotated masks are available at https://doi.org/10.57967/hf/2117"

which resolves to the Hugging Face dataset `av555/ch4net`. That repo is **gated**: every
file fetch (labels, s2, mbmp arrays, and even the quickstart notebook) returns
`GatedRepoError` / HTTP 401 ("Access to dataset av555/ch4net is restricted. You must have
access to it and be authenticated to access it."). Accessing it requires (a) a Hugging Face
account, (b) accepting the dataset's access terms to be granted access, and (c) an
`HF_TOKEN` in the environment. None of these are available in this environment, and the AMT
paper lists no Zenodo record, GitHub data mirror, or other ungated source.

Per the task spec (SOP step 2 / registry section 1a) a dataset blocked solely on missing
credentials / an access gate we cannot satisfy is a **`rejected`** with
`notes: "needs-credential: ..."` (NOT `temporary_failure`, which is reserved for transient
5xx/timeout/rate-limit errors on an otherwise-open source — this is a persistent access
gate, not a transient outage). The user can lift this later by requesting access on Hugging
Face and supplying an `HF_TOKEN` (or a pre-downloaded copy), after which this script's
download path can be filled in and re-run.

Secondary concerns to record (for the user, not the primary reason):

  * LICENSE MISMATCH. The manifest lists this dataset as `CC-BY-4.0`, but the Hugging Face
    repo is tagged **`cc-by-nc-nd-4.0`** (NonCommercial-NoDerivatives). The paper text
    itself is CC-BY-4.0, but the *dataset* carries the more restrictive NC-ND terms.
    NoDerivatives in particular is in tension with producing and redistributing derived
    label rasters for pretraining; the user should confirm license terms before use even
    once access is granted. (This alone could justify rejection under "license forbids
    use"; we lead with the access gate because it is the hard, verified blocker.)

  * GEOREFERENCING UNVERIFIED. Files are opaque `.npy` arrays named by running index
    (`{split}/{label,s2,mbmp}/{i}.npy`) with no accompanying coordinate table visible in
    the (gated) file listing. The paper describes patches as 0.01 deg x 0.01 deg / 200 x 200
    px centered on the 23 known super-emitter sites, so per-patch geolocation is *plausibly*
    recoverable (site centroids + fixed extent) IF the release includes a site/patch->site
    mapping. This could not be confirmed because the arrays are inaccessible. If access is
    later granted, verify a coordinate/site index exists before committing to processing;
    if the arrays are coordinate-free like Landslide4Sense/LoveDA, this would flip to a
    "no recoverable georeferencing" rejection instead.

If accepted in the future, the intended recipe is `dense_raster`: single foreground class
`methane plume` (id 0; positive-only per spec 5 -- do NOT fabricate negatives, leave
non-plume pixels as nodata/ignore, though the plume-free scenes provide real negatives),
resample the ~5.5 m label masks to 10 m (mode/nearest) and tile to <=64x64 UTM patches,
time_range = the scene's acquisition year (Sentinel era, 2017-2021, all post-2016).

Running this module re-verifies the gate and (re)writes the rejection summary. It writes
nothing under weka `datasets/` other than the per-dataset `registry_entry.json`, and never
touches the central `registry.json`.
"""

from pathlib import Path

from olmoearth_pretrain.open_set_segmentation_data import manifest

SLUG = "ch4net_methane_super_emitter_plumes"
NAME = "CH4Net (methane super-emitter plumes)"
HF_REPO = "av555/ch4net"
DOI = "https://doi.org/10.57967/hf/2117"
PAPER = "https://doi.org/10.5194/amt-17-2583-2024"  # AMT 17, 2583-2593 (2024)

SUMMARY_PATH = Path(
    "data/open_set_segmentation_data/"
    "dataset_summaries/ch4net_methane_super_emitter_plumes.md"
)

REJECT_NOTE = (
    "needs-credential: HF token + access approval. Sole distribution is the GATED HF "
    "dataset av555/ch4net (DOI 10.57967/hf/2117); all file fetches return GatedRepoError/"
    "401 and no HF_TOKEN is available. No Zenodo/GitHub/other mirror in the AMT paper. "
    "Also NOTE license mismatch: manifest says CC-BY-4.0 but HF repo is cc-by-nc-nd-4.0 "
    "(NoDerivatives). To retry: request access on HF, set HF_TOKEN, re-run; then verify a "
    "per-patch site/coordinate mapping exists (arrays are index-named .npy, georef "
    "unverified) before processing as dense_raster (single class 'methane plume')."
)


def _try_access() -> bool:
    """Attempt an unauthenticated fetch of one label file; return True if it succeeds."""
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(HF_REPO, "test/label/0.npy", repo_type="dataset")
        return True
    except Exception as e:  # GatedRepoError / 401 expected
        print(f"  access check failed as expected: {type(e).__name__}: {str(e)[:160]}")
        return False


SUMMARY = f"""# CH4Net (methane super-emitter plumes) — REJECTED (needs-credential: gated HF repo)

- **Slug**: `{SLUG}`
- **Name**: {NAME}
- **Source**: AMT paper (Vaughan et al. 2024, {PAPER}); data/code DOI {DOI}
  -> Hugging Face dataset `{HF_REPO}`.
- **Family / region**: plume / Turkmenistan (23 super-emitter sites), 2017-2021.
- **Label type (manifest)**: dense_raster, single class `methane plume`, manual annotation.
- **License**: manifest says CC-BY-4.0; **HF repo is tagged `cc-by-nc-nd-4.0`** (see below).
- **Status**: **rejected** — `needs-credential`.
- **Primary rejection reason**: the only distribution is a **gated** Hugging Face repo that
  requires authentication + granted access we do not have.

## What CH4Net is

A hand-annotated methane-plume segmentation dataset on Sentinel-2 imagery for automated
super-emitter monitoring. It covers 23 known super-emitter locations in Turkmenistan and
comprises **925 hand-annotated plume masks** plus **9,121 plume-free scenes** (10,046 images
total, 2017-2021). The HF repo is organized as `{{train,val,test}}/{{label,s2,mbmp}}/{{i}}.npy`
(8,255 train / 255 val / 2,473 test triples), where `label` is the plume mask, `s2` the
Sentinel-2 patch, and `mbmp` the multi-band multi-pass methane product. The paper describes
patches as 0.01 deg x 0.01 deg (200 x 200 px) centered on the emitter sites.

## Why it is rejected (needs-credential)

The AMT data/code-availability statement gives a **single** location — "Code and
hand-annotated masks are available at {DOI}" — which resolves to `{HF_REPO}`. That repo is
**gated**: every attempted file download (label/s2/mbmp `.npy` and `quickstart.ipynb`)
returns `GatedRepoError` / HTTP 401: *"Access to dataset av555/ch4net is restricted. You
must have access to it and be authenticated to access it."* Only the (empty) `README.md`
metadata is fetchable. Accessing the data requires a Hugging Face account, accepting the
dataset's access terms to be granted access, and an `HF_TOKEN` — none available here. The
paper lists **no Zenodo, GitHub, or other ungated mirror**. This is a persistent access
gate, not a transient outage, so it is `rejected` (needs-credential), not
`temporary_failure`.

## Secondary concerns (for the user)

1. **License mismatch.** The manifest records `CC-BY-4.0`, but the HF dataset is tagged
   **`cc-by-nc-nd-4.0`** (NonCommercial-**NoDerivatives**). The paper *text* is CC-BY-4.0;
   the *dataset* is NC-ND. NoDerivatives is in tension with generating/redistributing
   derived label rasters for pretraining — confirm terms before use even once access is
   granted. This could independently support a "license forbids use" rejection.

2. **Georeferencing unverified.** Files are opaque running-index `.npy` arrays with no
   coordinate table visible in the gated listing. Per-patch geolocation is *plausibly*
   recoverable (23 known site centroids + a fixed 0.01 deg extent) IF the release includes a
   patch->site mapping, but this could not be confirmed because the arrays are inaccessible.
   If access is later granted, check for a site/coordinate index first; if the arrays are
   coordinate-free (like Landslide4Sense/LoveDA), reject instead on "no recoverable
   georeferencing".

## Intended recipe if accepted later

`dense_raster`, single foreground class `methane plume` (id 0). Positive-only phenomenon —
per spec 5, do NOT fabricate negatives (the 9,121 plume-free scenes already provide real
negatives; leave non-plume pixels as nodata/ignore for the positive scenes). Resample the
~5.5 m masks to 10 m with mode/nearest and tile to <=64x64 local-UTM patches; time_range =
the scene acquisition year (all 2017-2021, post-2016 Sentinel era). All splits are usable as
pretraining labels.

## Reproduce

```
python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.ch4net_methane_super_emitter_plumes
```

This re-verifies the gate and re-writes this summary; it produces no dataset outputs. To
process once credentials exist: `huggingface-cli login` (or export `HF_TOKEN`), request
access to `{HF_REPO}`, then implement the dense_raster path above.
"""


def main() -> None:
    print(f"{NAME}: sole distribution is gated HF repo {HF_REPO} (DOI {DOI}).")
    accessible = _try_access()
    if accessible:
        print(
            "WARNING: label file unexpectedly downloaded — the gate may have been lifted. "
            "Re-triage this dataset (verify georeferencing + license) before rejecting."
        )
    else:
        print("Confirmed: repo is gated / requires credentials we do not have.")
    print(
        "License note: manifest=CC-BY-4.0 but HF repo tag=cc-by-nc-nd-4.0 (NoDerivatives)."
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(SUMMARY)
    print(f"Wrote rejection summary -> {SUMMARY_PATH}")

    manifest.write_registry_entry(SLUG, "rejected", notes=REJECT_NOTE)
    print("Wrote registry_entry.json (status=rejected).")
    print(
        "STATUS: rejected — needs-credential (gated HF repo av555/ch4net; no token/access; "
        "no ungated mirror). See summary for license mismatch + georef caveats."
    )


if __name__ == "__main__":
    main()
