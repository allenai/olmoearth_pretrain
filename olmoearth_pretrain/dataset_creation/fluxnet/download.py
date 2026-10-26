"""Stage 1: list and download FLUXNET (ONEFlux v1.3+) archives from the data hubs.

Does what the FLUXNET Shuttle (https://github.com/fluxnet/shuttle) does, with
plain ``requests`` calls against the same three hubs:

* AmeriFlux: ``amfcdn.lbl.gov/api/v2`` site list + download links + citations.
  Downloads are logged with the Shuttle's anonymous ``fluxnetshuttle`` user id
  and no personal information.
* ICOS Carbon Portal: SPARQL query for ``miscFluxnetArchiveProduct`` objects
  (ICOS, European Fluxes Database, JapanFlux, ChinaFlux, KoFlux, SAEON and the
  FLUXNET legacy collection). The ``licence_accept`` link accepts the CC-BY-4.0
  licence per object and sets the cookie the object download needs.
* TERN (OzFlux): the THREDDS product catalogue.

All data are CC-BY-4.0; ``catalog.csv`` keeps every product's citation.

Usage::

    python -m olmoearth_pretrain.dataset_creation.fluxnet.download all --workers 8
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .common import DEFAULT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AMERIFLUX_API = "https://amfcdn.lbl.gov/api/v2/"
AMERIFLUX_USER_ID = "fluxnetshuttle"
ICOS_SPARQL_URL = "https://meta.icos-cp.eu/sparql"
ICOS_LICENCE_URL = "https://data.icos-cp.eu/licence_accept?ids=%5B%22{}%22%5D"
TERN_BASE_URL = "https://dap.tern.org.au/thredds/fileServer/ecosystem_process/fluxnet/"
HTTP_TIMEOUT = (60, 300)
MAX_ATTEMPTS = 5

ICOS_SPARQL_QUERY = """
prefix cpmeta: <http://meta.icos-cp.eu/ontologies/cpmeta/>
prefix prov: <http://www.w3.org/ns/prov#>
select ?dobj ?fileName ?size ?lat ?lon ?ecosystemType ?citationString
where {
    VALUES ?spec {<http://meta.icos-cp.eu/resources/cpmeta/miscFluxnetArchiveProduct>}
    ?dobj cpmeta:hasObjectSpec ?spec .
    ?dobj cpmeta:hasSizeInBytes ?size .
    ?dobj cpmeta:hasName ?fileName .
    ?dobj cpmeta:wasAcquiredBy/prov:wasAssociatedWith ?station .
    OPTIONAL { ?station cpmeta:hasLatitude ?lat . ?station cpmeta:hasLongitude ?lon . }
    OPTIONAL { ?station cpmeta:hasEcosystemType ?ecosystemType . }
    OPTIONAL { ?dobj cpmeta:hasCitationString ?citationString . }
    FILTER NOT EXISTS {[] cpmeta:isNextVersionOf ?dobj}
}
"""

CATALOG_COLUMNS = [
    "hub",
    "site_id",
    "filename",
    "url",
    "size",
    "md5",
    "lat",
    "lon",
    "igbp",
    "citation",
]
# Daily-file markers: ONEFlux v1.3+ archives and the older FLUXNET2015-style
# FULLSET archives some sites are still published as.
DAILY_MARKERS = ("_FLUXMET_DD_", "_FULLSET_DD_")


# --- catalogue ----------------------------------------------------------------
def _site_id_from_filename(filename: str) -> str:
    """Site id from a ONEFlux archive name.

    Args:
        filename: e.g. ``AMF_US-Ha1_FLUXNET_1991-2020_v1.3_r1.zip``.

    Returns:
        The site id (``US-Ha1``).
    """
    return filename.split("_")[1]


def list_ameriflux(session: requests.Session) -> list[dict[str, Any]]:
    """AmeriFlux sites that publish a FLUXNET product, with links and citations.

    Args:
        session: HTTP session.

    Returns:
        Catalogue rows (see ``CATALOG_COLUMNS``).
    """
    resp = session.get(
        f"{AMERIFLUX_API}site_info_display/AmeriFlux", timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    sites = {
        s["site_id"]: s
        for s in resp.json().get("values", [])
        if s.get("grp_publish_fluxnet")
    }
    site_ids = sorted(sites)
    resp = session.post(
        f"{AMERIFLUX_API}amf_shuttle_data_files_and_manifest",
        json={
            "user_id": AMERIFLUX_USER_ID,
            "data_product": "FLUXNET",
            "data_variant": "FULLSET",
            "site_ids": site_ids,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    links = resp.json().get("data_urls", [])
    resp = session.post(
        f"{AMERIFLUX_API}citations/FLUXNET",
        json={"site_ids": site_ids},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    citations = {
        c["site_id"]: c.get("citation", "") for c in resp.json().get("values", [])
    }

    rows = []
    for link in links:
        site_id = link["site_id"]
        location = sites.get(site_id, {}).get("grp_location", {})
        igbp = sites.get(site_id, {}).get("grp_igbp", {}).get("igbp", "")
        rows.append(
            dict(
                hub="ameriflux",
                site_id=site_id,
                filename=link["url"].split("/")[-1].split("?")[0],
                url=link["url"],
                size=int(link.get("download_size") or 0),
                md5=link.get("download_checksum", ""),
                lat=float(location.get("location_lat") or "nan"),
                lon=float(location.get("location_long") or "nan"),
                igbp=igbp or "",
                citation=citations.get(site_id, ""),
            )
        )
    return rows


def list_icos(session: requests.Session) -> list[dict[str, Any]]:
    """Latest-version FLUXNET archive objects on the ICOS Carbon Portal.

    Args:
        session: HTTP session.

    Returns:
        Catalogue rows (see ``CATALOG_COLUMNS``).
    """
    resp = session.post(
        ICOS_SPARQL_URL,
        data={"query": ICOS_SPARQL_QUERY},
        headers={"Accept": "application/json"},
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    rows = []
    for binding in resp.json()["results"]["bindings"]:
        value = {key: item["value"] for key, item in binding.items()}
        object_id = value["dobj"].split("/")[-1]
        rows.append(
            dict(
                hub="icos",
                site_id=_site_id_from_filename(value["fileName"]),
                filename=value["fileName"],
                url=ICOS_LICENCE_URL.format(object_id),
                size=int(value["size"]),
                md5="",
                lat=float(value.get("lat", "nan")),
                lon=float(value.get("lon", "nan")),
                # .../cpmeta/igbp_ENF -> ENF
                igbp=value.get("ecosystemType", "").rsplit("igbp_", 1)[-1]
                if "igbp_" in value.get("ecosystemType", "")
                else "",
                citation=value.get("citationString", ""),
            )
        )
    return rows


def list_tern(session: requests.Session) -> list[dict[str, Any]]:
    """TERN (OzFlux) FLUXNET products from the THREDDS catalogue.

    Args:
        session: HTTP session.

    Returns:
        Catalogue rows (see ``CATALOG_COLUMNS``); coordinates come from the
        archives' BIF files later.
    """
    resp = session.get(
        f"{TERN_BASE_URL}TERN_THREDDS_catalogue.csv", timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    rows = []
    for record in csv.DictReader(io.StringIO(resp.text)):
        url = record["PRODUCT_URL"].strip()
        if not url:
            continue
        rows.append(
            dict(
                hub="tern",
                site_id=record["SITE_ID"].strip(),
                filename=url.split("/")[-1],
                url=url,
                size=0,
                md5="",
                lat=float("nan"),
                lon=float("nan"),
                igbp="",
                citation=record.get("PRODUCT_CITATION", "").strip(),
            )
        )
    return rows


def build_catalog(root: Path) -> pd.DataFrame:
    """List all three hubs and write ``raw/catalog.csv``.

    Args:
        root: Dataset root.

    Returns:
        The catalogue, one row per archive.
    """
    session = requests.Session()
    rows: list[dict[str, Any]] = []
    for name, lister in (
        ("ameriflux", list_ameriflux),
        ("icos", list_icos),
        ("tern", list_tern),
    ):
        hub_rows = lister(session)
        logger.info("%s: %d archives", name, len(hub_rows))
        rows.extend(hub_rows)
    catalog = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    duplicated = catalog["site_id"].duplicated(keep=False)
    if duplicated.any():
        logger.warning(
            "site ids listed by more than one hub (all kept, deduplicated later): %s",
            sorted(catalog.loc[duplicated, "site_id"].unique()),
        )
    out = root / "raw" / "catalog.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out, index=False)
    logger.info("catalogue: %d archives, %.1f GB listed -> %s", len(catalog),
                catalog["size"].sum() / 1e9, out)  # fmt: skip
    return catalog


# --- download -----------------------------------------------------------------
def archive_path(root: Path, row: dict[str, Any]) -> Path:
    """Where one catalogue row's archive is stored.

    Args:
        root: Dataset root.
        row: Catalogue row.

    Returns:
        ``<root>/raw/<hub>/<filename>``.
    """
    return root / "raw" / row["hub"] / row["filename"]


def _is_valid_archive(path: Path, row: dict[str, Any]) -> bool:
    """Whether a downloaded archive is complete.

    Args:
        path: Local archive.
        row: Its catalogue row (expected size / md5, when the hub gives them).

    Returns:
        True if the size and checksum match (when known) and the zip opens.
    """
    if not path.exists():
        return False
    if row["size"] and path.stat().st_size != int(row["size"]):
        return False
    if isinstance(row["md5"], str) and row["md5"]:
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                digest.update(block)
        if digest.hexdigest() != row["md5"]:
            return False
    try:
        with zipfile.ZipFile(path) as archive:
            return any(
                marker in name
                for name in archive.namelist()
                for marker in DAILY_MARKERS
            )
    except zipfile.BadZipFile:
        return False


def _log_ameriflux_download(session: requests.Session, filename: str) -> None:
    """Record the download with AmeriFlux's tracker, as the Shuttle does (no user info).

    Args:
        session: HTTP session.
        filename: Archive name.
    """
    try:
        session.post(
            f"{AMERIFLUX_API}log_shuttle_data_request",
            json={"user_id": AMERIFLUX_USER_ID, "zip_filenames": [filename]},
            headers={"Referer": "local_shuttle"},
            timeout=(30, 30),
        )
    except requests.RequestException as exc:
        logger.warning("AmeriFlux download log failed for %s: %s", filename, exc)


def download_one(root: Path, row: dict[str, Any]) -> tuple[str, str]:
    """Download one archive (atomically, with retries) unless it is already complete.

    Args:
        root: Dataset root.
        row: Catalogue row.

    Returns:
        ``(filename, status)`` with status ``exists``, ``downloaded`` or ``failed: ...``.
    """
    path = archive_path(root, row)
    if _is_valid_archive(path, row):
        return row["filename"], "exists"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # A fresh session per attempt: ICOS keys the licence cookie per object.
        session = requests.Session()
        try:
            if row["hub"] == "ameriflux":
                _log_ameriflux_download(session, row["filename"])
            with session.get(row["url"], stream=True, timeout=HTTP_TIMEOUT) as resp:
                resp.raise_for_status()
                # ICOS answers a refused licence with its HTML licence page.
                if "text/html" in resp.headers.get("Content-Type", ""):
                    raise ValueError(
                        f"not an archive: {resp.headers.get('Content-Type')}"
                    )
                with tmp.open("wb") as f:
                    for block in resp.iter_content(1 << 22):
                        f.write(block)
            tmp.replace(path)
            if _is_valid_archive(path, row):
                return row["filename"], "downloaded"
            error = "invalid archive after download"
        except (requests.RequestException, ValueError, OSError) as exc:
            error = str(exc)
        logger.warning("%s attempt %d failed: %s", row["filename"], attempt, error)
        time.sleep(min(300, 15 * 2**attempt))
    return row["filename"], f"failed: {error}"


def download_all(root: Path, catalog: pd.DataFrame, workers: int) -> None:
    """Download every catalogue archive and write ``raw/download_status.csv``.

    Args:
        root: Dataset root.
        catalog: Catalogue from :func:`build_catalog`.
        workers: Concurrent downloads.
    """
    rows = catalog.to_dict("records")
    statuses: dict[str, str] = {}
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(download_one, root, row) for row in rows]
        for k, future in enumerate(as_completed(futures), start=1):
            filename, status = future.result()
            statuses[filename] = status
            if k % 25 == 0 or status.startswith("failed"):
                logger.info("[%d/%d] %s: %s", k, len(rows), filename, status)
    catalog = catalog.assign(status=catalog["filename"].map(statuses))
    catalog.to_csv(root / "raw" / "download_status.csv", index=False)
    logger.info(
        "download status:\n%s", catalog["status"].str.split(":").str[0].value_counts()
    )


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=["catalog", "download", "all"])
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--limit", type=int, default=None, help="first N archives only (smoke test)"
    )
    args = ap.parse_args()
    root = Path(args.root)
    if args.step in ("catalog", "all"):
        catalog = build_catalog(root)
    else:
        catalog = pd.read_csv(root / "raw" / "catalog.csv", keep_default_na=False)
    if args.step in ("download", "all"):
        if args.limit is not None:
            catalog = catalog.head(args.limit)
        download_all(root, catalog, args.workers)


if __name__ == "__main__":
    main()
