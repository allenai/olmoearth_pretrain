"""Triage EGMS (European Ground Motion Service) -> REJECTED (needs-credential: EGMS
Copernicus Land registration).

EGMS (https://egms.land.copernicus.eu/, Copernicus Land Monitoring Service) is the
strongest continental ground-motion product: a PSI/DS InSAR analysis of the full Sentinel-1
archive over EEA39 (EU + Norway, UK, Switzerland, Iceland). It provides per-point line-of-
sight and (L3 Ortho) vertical + east-west displacement VELOCITIES in mm/yr, at ~20x5 m
native point density (L2) or on a 100 m grid (L3), 2016-present. This would be an excellent
regression signal for this pipeline (vertical velocity mm/yr; static multi-year-average
label, change_time=null). See the dataset summary for the full intended recipe.

WHY REJECTED (needs-credential):
EGMS data downloads ONLY through the EGMS Explorer web app, gated behind EU Login
(ecas.ec.europa.eu) plus a time-limited, interactively-generated session token. The
download URL is
    https://egms.land.copernicus.eu/insar-api/archive/download/{FILE}.zip?id={TOKEN}
where {TOKEN} is a ~32-char user-session id obtained ONLY by authenticating in a browser on
the Explorer and copying it from a generated download link (this is exactly how the
community EGMStoolkit works: you paste the token via -t; there is no username/password login
path). There is no unauthenticated/anonymous download: a GET of a download URL WITHOUT a
token returns HTTP 401 (verified during triage). No open bulk mirror exists.

The credential in .env does NOT apply: COPERNICUS_USERNAME/PASSWORD
are Copernicus Data Space Ecosystem (CDSE) credentials -- a different identity system from
EU Login. Verified during triage that those creds return a valid access_token from the CDSE
Keycloak endpoint (identity.dataspace.copernicus.eu/.../CDSE/.../token), i.e. they are
CDSE-realm creds, not EU Login (ECAS) creds and not an EGMS session token. Copernicus LAND
(land.copernicus.eu / EGMS) authenticates against EU Login, for which .env has no
credential; and even a valid EU Login would still require replicating the Explorer's
interactive session to mint the download token.

Per the task spec (SOP step 2 / registry section 1a), a dataset blocked solely on a missing
credential / registration portal we cannot satisfy is a `rejected` with
`notes: "needs-credential: ..."` -- NOT `temporary_failure` (the 401 is an intended auth
gate, not a transient outage). The user can lift this later by supplying an EGMS Explorer
token (EU Login) or a pre-downloaded copy of the bounded L3 Ortho tiles.

Running this module re-verifies the 401 gate and (re)writes the per-dataset
registry_entry.json (status=rejected). It writes nothing else under weka `datasets/` and
never touches the central `registry.json`. The hand-authored summary lives at
data/open_set_segmentation_data/dataset_summaries/egms_european_ground_motion_service.md.
"""

import urllib.error
import urllib.request

from olmoearth_pretrain.open_set_segmentation_data import manifest

SLUG = "egms_european_ground_motion_service"
NAME = "EGMS (European Ground Motion Service)"
PORTAL = "https://egms.land.copernicus.eu/"
# A representative L3 Ortho tile download URL (no token) -> expected HTTP 401.
PROBE_URL = (
    "https://egms.land.copernicus.eu/insar-api/archive/download/"
    "EGMS_L3_E30N30_100km_U_2018_2022_1.zip"
)

REJECT_NOTE = (
    "needs-credential: EGMS Copernicus Land registration. Download is gated behind EU Login "
    "(ecas.ec.europa.eu) + a time-limited session token from the EGMS Explorer web UI "
    "(URL .../insar-api/archive/download/{FILE}.zip?id={TOKEN}); no-token GET returns HTTP "
    "401 and there is no anonymous/API access or open mirror. .env COPERNICUS_* creds are "
    "Copernicus Data Space Ecosystem (CDSE) creds, a DIFFERENT identity system from EU Login "
    "(verified: they return a valid CDSE token), so they do not authenticate EGMS. To retry: "
    "supply an EGMS Explorer token (EU Login) or a pre-downloaded copy of the bounded L3 "
    "Ortho tiles, then process as regression on vertical velocity mm/yr (static label, "
    "change_time=null) per the dataset summary."
)


def _probe_gate() -> int | None:
    """GET a token-less EGMS download URL; return the HTTP status (expect 401)."""
    try:
        with urllib.request.urlopen(PROBE_URL, timeout=30) as r:
            print(
                f"  WARNING: probe returned HTTP {r.status} (expected 401 auth gate)."
            )
            return r.status
    except urllib.error.HTTPError as e:
        print(f"  probe returned HTTP {e.code} as expected (auth gate).")
        return e.code
    except Exception as e:  # noqa: BLE001 - network hiccup; gate is still in place
        print(f"  probe network error: {type(e).__name__}: {str(e)[:160]}")
        return None


def main() -> None:
    print(
        f"{NAME}: download is gated (EU Login + interactive session token) at {PORTAL}."
    )
    status = _probe_gate()
    if status == 401:
        print("Confirmed: EGMS download requires a credential/token we do not have.")
    elif status == 200:
        print(
            "WARNING: an unauthenticated download unexpectedly succeeded -- re-triage this "
            "dataset (the gate may have changed) before rejecting."
        )
    else:
        print(
            "Could not confirm the 401 gate over the network right now, but the access model "
            "is unchanged (EU Login + interactive token); rejecting on needs-credential."
        )
    print(
        "Credential note: .env COPERNICUS_* are CDSE (dataspace.copernicus.eu) creds, NOT "
        "EU Login (ecas.ec.europa.eu) creds -- they do not authenticate EGMS/Copernicus Land."
    )
    manifest.write_registry_entry(SLUG, "rejected", notes=REJECT_NOTE)
    print("Wrote registry_entry.json (status=rejected).")
    print(
        "STATUS: rejected -- needs-credential (EGMS Copernicus Land registration / EU Login "
        "+ session token; no anonymous access; .env has CDSE creds only). Intended framing "
        "if accepted: regression on vertical displacement velocity (mm/yr). See summary."
    )


if __name__ == "__main__":
    main()
