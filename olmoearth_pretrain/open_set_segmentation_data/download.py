"""Download helpers with atomic writes. Extend as new source types are needed.

Always write to raw_dir(slug). Re-check disk with io.check_disk() before large pulls.
"""

import io as _io
import struct
import time as _time
import urllib.parse
import urllib.request
import zipfile
import zlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
from upath import UPath

if TYPE_CHECKING:
    from _typeshed import WriteableBuffer


def _http_range(
    url: str,
    start: int,
    end_inclusive: int,
    headers: dict[str, str] | None = None,
    retries: int = 4,
    timeout: float = 300.0,
) -> bytes:
    """Fetch a byte range [start, end_inclusive] from a URL (with simple retry)."""
    last = ""
    for attempt in range(retries):
        try:
            h = dict(headers or {})
            h["Range"] = f"bytes={start}-{end_inclusive}"
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
                return r.read()
        except Exception as ex:  # noqa: BLE001 - retry transient network errors
            last = repr(ex)
            _time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(
        f"range request failed for {url} [{start}-{end_inclusive}]: {last}"
    )


def remote_zip_index(
    url: str, total_size: int | None = None
) -> dict[str, tuple[int, int, int, int]]:
    """Read a remote zip's central directory via HTTP Range requests (no full download).

    Returns ``{member_name: (local_header_offset, compressed_size, uncompressed_size,
    method)}``. Supports Zip64 (large archives / >65535 entries). The server must accept
    ``Range`` requests. Use with :func:`extract_remote_zip_member` to pull only the members
    (e.g. a thin label layer) needed out of a large bulk archive without downloading it all.
    """
    if total_size is None:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=120) as r:  # nosec B310
            total_size = int(r.headers["Content-Length"])
    tail_len = min(1 << 16, total_size)
    tail = _http_range(url, total_size - tail_len, total_size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no EOCD found in zip tail")
    (_, _, _, _, _, cd_size, cd_off, _) = struct.unpack("<IHHHHIIH", tail[i : i + 22])
    loc = tail.rfind(b"PK\x06\x07")
    if loc >= 0:
        (_, _, z64_off, _) = struct.unpack("<IIQI", tail[loc : loc + 20])
        hdr = _http_range(url, z64_off, z64_off + 56)
        if hdr[:4] == b"PK\x06\x06":
            (_, _, _, _, _, _, _, _, cd_size64, cd_off64) = struct.unpack(
                "<IQHHIIQQQQ", hdr[:56]
            )
            cd_size, cd_off = cd_size64, cd_off64
    cd = _http_range(url, cd_off, cd_off + cd_size - 1)
    entries: dict[str, tuple[int, int, int, int]] = {}
    p = 0
    while p + 46 <= len(cd) and cd[p : p + 4] == b"PK\x01\x02":
        (_, _, _, _, method, _, _, _, csize, usize, nlen, elen, clen, _, _, _, lho) = (
            struct.unpack("<IHHHHHHIIIHHHHHII", cd[p : p + 46])
        )
        name = cd[p + 46 : p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen : p + 46 + nlen + elen]
        if lho == 0xFFFFFFFF or csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
            ep = 0
            while ep + 4 <= len(extra):
                tag, dsz = struct.unpack("<HH", extra[ep : ep + 4])
                vp = ep + 4
                if tag == 1:
                    if usize == 0xFFFFFFFF:
                        usize = struct.unpack("<Q", extra[vp : vp + 8])[0]
                        vp += 8
                    if csize == 0xFFFFFFFF:
                        csize = struct.unpack("<Q", extra[vp : vp + 8])[0]
                        vp += 8
                    if lho == 0xFFFFFFFF:
                        lho = struct.unpack("<Q", extra[vp : vp + 8])[0]
                        vp += 8
                ep += 4 + dsz
        entries[name] = (lho, csize, usize, method)
        p += 46 + nlen + elen + clen
    return entries


def extract_remote_zip_member(
    url: str, entry: tuple[int, int, int, int], max_uncompressed: int | None = None
) -> bytes:
    """Fetch (and decompress) a single member of a remote zip via Range requests.

    ``entry`` is the ``(local_header_offset, compressed_size, uncompressed_size, method)``
    tuple from :func:`remote_zip_index`. Supports stored (0) and deflate (8) members. If
    ``max_uncompressed`` is set, only that many leading uncompressed bytes are produced
    (fetching just enough compressed bytes) — handy for reading a file *header* (e.g. a
    GeoTIFF's georeferencing tags) without pulling the whole member.
    """
    lho, csize, usize, method = entry
    lh = _http_range(url, lho, lho + 29)
    nlen, elen = struct.unpack("<HH", lh[26:30])
    data_off = lho + 30 + nlen + elen
    if method == 0:
        want = csize if max_uncompressed is None else min(csize, max_uncompressed)
        return _http_range(url, data_off, data_off + want - 1)
    if method != 8:
        raise RuntimeError(f"unsupported zip compression method {method}")
    d = zlib.decompressobj(-15)
    out = b""
    pos = data_off
    step = 1 << 20
    end_all = data_off + csize
    while pos < end_all:
        chunk = _http_range(url, pos, min(pos + step, end_all) - 1)
        pos += len(chunk)
        if max_uncompressed is not None:
            out += d.decompress(chunk, max_uncompressed - len(out))
            if len(out) >= max_uncompressed or d.eof:
                break
        else:
            out += d.decompress(chunk)
            if d.eof:
                break
    return out


def extract_zip(
    zip_path: str | UPath, dst_dir: str | UPath, skip_existing: bool = True
) -> UPath:
    """Extract a .zip into dst_dir (idempotent). Returns dst_dir.

    If ``skip_existing`` and dst_dir already exists with contents, do nothing.
    """
    dst_dir = UPath(dst_dir)
    if skip_existing and dst_dir.exists() and any(dst_dir.iterdir()):
        return dst_dir
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path)) as zf:
        zf.extractall(str(dst_dir))
    return dst_dir


def _atomic(dst: UPath, write_fn: Callable[[UPath], None]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.parent / (dst.name + ".tmp")
    write_fn(tmp)
    tmp.rename(dst)


def download_http(
    url: str,
    dst: str | UPath,
    skip_existing: bool = True,
    headers: dict[str, str] | None = None,
    timeout: float | None = None,
) -> UPath:
    """Download a URL to dst (atomic).

    ``headers`` lets callers set request headers (e.g. a User-Agent) for hosts that
    reject the default urllib agent (Mendeley Data, some CDNs return HTTP 403).
    ``timeout`` (seconds) guards against a hung connection stalling indefinitely.
    """
    dst = UPath(dst)
    if skip_existing and dst.exists():
        return dst

    req = urllib.request.Request(url, headers=headers or {})

    def _w(tmp: UPath) -> None:
        with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as f:  # nosec B310
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)

    _atomic(dst, _w)
    return dst


def wms_getmap_geotiff(
    base_url: str,
    layer: str,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    srs: str = "EPSG:5070",
    time: str | None = None,
    fmt: str = "image/geotiff",
    version: str = "1.1.1",
    timeout: float = 120.0,
    retries: int = 4,
    headers: dict[str, str] | None = None,
    extra_params: dict[str, str] | None = None,
) -> bytes:
    """Fetch a raw single-band GeoTIFF window from a WMS GetMap endpoint.

    GeoServer's ``image/geotiff`` GetMap output returns the *raw* coverage values
    (not a styled RGB), so this is a convenient way to do bounded reads of a large
    raster/ImageMosaic coverage over HTTP without downloading the whole mosaic — the
    open-set-segmentation analogue of a COG range read for servers that only expose
    OGC services. ``bbox`` is ``(minx, miny, maxx, maxy)`` in ``srs`` axis order
    (for WMS 1.1.1 projected CRS this is easting, northing). ``time`` is an optional
    ISO8601 instant for coverages with a TIME dimension. Retries with backoff on
    transient errors; raises RuntimeError if the server returns a ServiceException /
    non-GeoTIFF payload after all retries. Returns the GeoTIFF bytes.
    """
    import time as _time

    params = {
        "service": "WMS",
        "version": version,
        "request": "GetMap",
        "layers": layer,
        "styles": "",
        "srs" if version.startswith("1.1") else "crs": srs,
        "bbox": ",".join(f"{v:.6f}" for v in bbox),
        "width": str(width),
        "height": str(height),
        "format": fmt,
    }
    if time is not None:
        params["TIME"] = time
    if extra_params:
        params.update(extra_params)
    url = base_url + ("&" if "?" in base_url else "?") + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "Mozilla/5.0"})

    last_err = ""
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
                data = r.read()
            # GeoTIFF magic: little-endian "II*\0" or big-endian "MM\0*".
            if data[:2] in (b"II", b"MM") and len(data) > 8:
                return data
            last_err = data[:300].decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001 - retry any transient network error
            last_err = repr(e)
        _time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"WMS GetMap failed for {layer} bbox={bbox}: {last_err}")


def download_earthdata(url: str, dst: str | UPath, skip_existing: bool = True) -> UPath:
    """Download an Earthdata-protected file (NASA URS OAuth) to dst (atomic).

    Uses a ``requests.Session`` which reads ``~/.netrc`` for
    ``machine urs.earthdata.nasa.gov`` credentials and follows the URS OAuth redirect
    chain (keeping cookies), so ORNL DAAC / LP DAAC ``/protected/`` URLs authenticate.
    Write your ~/.netrc (chmod 600) with those credentials before calling.
    """
    import requests

    dst = UPath(dst)
    if skip_existing and dst.exists():
        return dst

    def _w(tmp: UPath) -> None:
        with requests.Session() as s:
            r = s.get(url, timeout=300, stream=True)
            r.raise_for_status()
            if "text/html" in r.headers.get("Content-Type", ""):
                raise RuntimeError(
                    f"Earthdata auth failed for {url}: got HTML (login page?). "
                    "Check ~/.netrc for urs.earthdata.nasa.gov."
                )
            with tmp.open("wb") as f:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

    _atomic(dst, _w)
    return dst


def download_s3_unsigned(
    bucket: str,
    key: str,
    dst: str | UPath,
    skip_existing: bool = True,
    endpoint_url: str | None = None,
) -> UPath:
    """Download an object from a public (unsigned) S3 bucket to dst (atomic).

    ``endpoint_url`` targets an S3-compatible host other than AWS (e.g. Source
    Cooperative's ``https://data.source.coop`` data proxy, where ``bucket`` is the account
    name and ``key`` the repo-relative path).
    """
    import boto3
    import botocore

    dst = UPath(dst)
    if skip_existing and dst.exists():
        return dst
    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )

    def _w(tmp: UPath) -> None:
        s3.download_file(Bucket=bucket, Key=key, Filename=tmp.path)

    _atomic(dst, _w)
    return dst


def download_zenodo(
    record_id: str, dst_dir: str | UPath, filenames: list[str] | None = None
) -> list[UPath]:
    """Download files from a Zenodo record. If filenames is None, download all."""
    import json

    dst_dir = UPath(dst_dir)
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record_id}") as r:  # nosec B310
        meta: dict[str, Any] = json.loads(r.read())
    out = []
    for f in meta.get("files", []):
        name = f.get("key") or f.get("filename")
        if filenames and name not in filenames:
            continue
        link = f["links"].get("self") or f["links"].get("download")
        out.append(download_http(link, dst_dir / name))
    return out


def download_arcgis_layer(
    base_url: str,
    layer_id: int,
    dst: str | UPath,
    where: str = "1=1",
    out_sr: int = 4326,
    page: int = 2000,
    order_field: str = "OBJECTID",
    skip_existing: bool = True,
    headers: dict[str, str] | None = None,
) -> UPath:
    """Download all features of an ArcGIS REST Map/FeatureServer layer as GeoJSON (atomic).

    Pages through the layer with ``resultOffset``/``resultRecordCount`` (respecting the
    server's ``maxRecordCount``) and concatenates every page into one GeoJSON
    FeatureCollection written to ``dst``. ``base_url`` is the service endpoint (``.../
    MapServer`` or ``.../FeatureServer``); ``layer_id`` the numeric sub-layer. Geometries are
    requested in ``out_sr`` (default WGS84 4326). Label-only extraction: no imagery pulled.
    """
    import json

    dst = UPath(dst)
    if skip_existing and dst.exists():
        return dst

    features: list[dict[str, Any]] = []
    offset = 0
    while True:
        params = {
            "where": where,
            "outFields": "*",
            "outSR": str(out_sr),
            "returnGeometry": "true",
            "orderByFields": order_field,
            "resultOffset": str(offset),
            "resultRecordCount": str(page),
            "f": "geojson",
        }
        url = f"{base_url}/{layer_id}/query?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=300) as r:  # nosec B310
            fc = json.loads(r.read())
        batch = fc.get("features", [])
        features.extend(batch)
        if len(batch) < page:
            break
        offset += len(batch)

    def _w(tmp: UPath) -> None:
        with tmp.open("w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    _atomic(dst, _w)
    return dst


def download_postgrest_json(
    base_url: str,
    dst: str | UPath,
    select: str = "*",
    order: str | None = None,
    page: int = 20000,
    skip_existing: bool = True,
    headers: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> UPath:
    """Download all rows of a PostgREST table endpoint as one JSON array (atomic).

    PostgREST (e.g. the USGS EERSC APIs like the US Wind Turbine Database at
    ``https://energy.usgs.gov/api/uswtdb/v1/turbines``) serves table rows as a JSON array,
    with ``limit``/``offset`` pagination and ``select``/``order`` query params. This pages
    through with ``limit=page`` until a short page and concatenates all rows into ``dst``.
    Label-only extraction: no imagery pulled. ``base_url`` is the full table endpoint.
    """
    import json

    dst = UPath(dst)
    if skip_existing and dst.exists():
        return dst

    rows: list[Any] = []
    offset = 0
    while True:
        params = {"select": select, "limit": str(page), "offset": str(offset)}
        if order:
            params["order"] = order
        url = (
            base_url
            + ("&" if "?" in base_url else "?")
            + urllib.parse.urlencode(params)
        )
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
            batch = json.loads(r.read())
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += len(batch)

    def _w(tmp: UPath) -> None:
        with tmp.open("w") as f:
            json.dump(rows, f)

    _atomic(dst, _w)
    return dst


class HttpRangeFile(_io.RawIOBase):
    """Seekable, read-only file-like object backed by HTTP Range requests.

    Lets libraries that expect a local seekable file (e.g. ``h5py``) read only the
    bytes they need from a large remote file without downloading the whole thing.
    The server must support ``Range`` requests (HTTP 206 / ``Content-Range``). Pass
    ``auth=(user, pass)`` for HTTP Basic auth. Useful for extracting a thin label
    layer out of a multi-GB *uncompressed* ML-ready archive (gzip has no random
    access, so this only helps on uncompressed files).
    """

    def __init__(self, url: str, auth: tuple[str, str] | None = None) -> None:
        """Open a range-backed file at ``url`` (optionally with HTTP Basic ``auth``)."""
        import requests

        self.url = url
        self.auth = auth
        self.pos = 0
        self.n_requests = 0
        self.n_bytes = 0
        self._sess = requests.Session()
        # Stream the probe (do NOT read the body): some servers (e.g. mediaTUM/Nextcloud
        # WebDAV) mishandle a degenerate ``bytes=0-0`` and stream the WHOLE file back, which
        # a non-streamed requests.get would eagerly buffer (hang on multi-GB files). With
        # stream=True we only read the headers (Content-Range's total) and close.
        r = self._sess.get(
            url, auth=auth, headers={"Range": "bytes=0-0"}, timeout=120, stream=True
        )
        try:
            r.raise_for_status()
            cr = r.headers.get("Content-Range")
            if not cr:
                raise RuntimeError(f"server does not support Range requests for {url}")
            self.size = int(cr.split("/")[-1])
        finally:
            r.close()

    def readable(self) -> bool:
        """Return True: the stream supports reading."""
        return True

    def seekable(self) -> bool:
        """Return True: the stream supports random access via Range requests."""
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        """Move the read position and return the new absolute offset."""
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        """Return the current read position."""
        return self.pos

    def _range(self, start: int, end_inclusive: int) -> bytes:
        r = self._sess.get(
            self.url,
            auth=self.auth,
            headers={"Range": f"bytes={start}-{end_inclusive}"},
            timeout=300,
        )
        r.raise_for_status()
        self.n_requests += 1
        self.n_bytes += len(r.content)
        return r.content

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes (all remaining if ``size`` < 0) via a Range request."""
        if size is None or size < 0:
            size = self.size - self.pos
        if size == 0:
            return b""
        data = self._range(self.pos, min(self.pos + size, self.size) - 1)
        self.pos += len(data)
        return data

    def readinto(self, b: "WriteableBuffer") -> int:
        """Read bytes into the pre-allocated buffer ``b`` and return the count read."""
        view = memoryview(b)
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        try:
            self._sess.close()
        finally:
            super().close()


def read_remote_h5_dataset(
    url: str, dataset: str, auth: tuple[str, str] | None = None
) -> np.ndarray:
    """Read one dataset out of a remote *uncompressed* HDF5 file via Range requests.

    Fetches only that dataset's bytes (plus a little HDF5 metadata), never the whole
    file. For a contiguous dataset this is a single big range read; otherwise it falls
    back to h5py's own (chunked) reads over the range file. Returns a numpy array.
    """
    import h5py

    rf = HttpRangeFile(url, auth=auth)
    try:
        f = h5py.File(rf, "r")
        dset = f[dataset]
        offset = dset.id.get_offset()
        if offset is not None:
            # Contiguous storage: read the whole dataset in one range request.
            nbytes = dset.id.get_storage_size()
            shape, dtype = dset.shape, dset.dtype
            f.close()
            raw = rf._range(offset, offset + nbytes - 1)
            return np.frombuffer(raw, dtype=dtype).reshape(shape)
        arr = dset[:]
        f.close()
        return arr
    finally:
        rf.close()


def list_gdrive_folder(folder_id: str) -> list[dict[str, str]]:
    """List a public Google Drive folder (recursively) without downloading.

    Returns a list of ``{"id": file_id, "path": relative_path}`` dicts. Uses gdown's
    folder walker in ``skip_download`` mode, so it only enumerates metadata. Reproducible:
    call this at runtime instead of hardcoding file ids.
    """
    import gdown

    res = gdown.download_folder(
        id=folder_id, skip_download=True, quiet=True, use_cookies=False
    )
    return [{"id": f.id, "path": f.path} for f in (res or [])]


def download_gdrive_file(
    file_id: str, dst: str | UPath, skip_existing: bool = True, timeout: float = 600.0
) -> UPath:
    """Download a single public Google Drive file by id (atomic).

    Uses the ``drive.usercontent.google.com/download`` endpoint with ``confirm=t``, which
    serves public files directly and is far less aggressively rate-limited than gdown's
    ``uc?id=`` path (gdown's endpoint returns "Cannot retrieve the public link ... have had
    many accesses" after a burst of anonymous hits, even for world-readable files). Follows
    redirects and validates that the payload is not the HTML virus-scan interstitial.
    """
    import requests

    dst = UPath(dst)
    if skip_existing and dst.exists():
        return dst
    url = "https://drive.usercontent.google.com/download"
    params = {"id": file_id, "export": "download", "confirm": "t"}

    def _w(tmp: UPath) -> None:
        with requests.get(url, params=params, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            first = next(r.iter_content(1 << 20), b"")
            if "text/html" in ctype and b"<html" in first[:2048].lower():
                raise RuntimeError(
                    f"Google Drive returned an HTML page for {file_id} (quota/scan "
                    "interstitial); retry later."
                )
            with tmp.open("wb") as f:
                if first:
                    f.write(first)
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        f.write(chunk)

    _atomic(dst, _w)
    return dst


def download_gem_tracker(
    requested_slugs: list[str],
    dst_dir: str | UPath,
    contact: dict[str, str],
    supabase_key: str = "sb_publishable_8mQAV8B2HhveNc5T8VGqPQ_1lgsFAvz",
    mint_url: str = "https://auxunjnrktkmeqyoyngm.supabase.co/rest/v1/rpc/mint_submission",
    presign_url: str = "https://auxunjnrktkmeqyoyngm.supabase.co/functions/v1/presign",
    skip_existing: bool = True,
    timeout: float = 300.0,
) -> list[UPath]:
    """Download Global Energy Monitor (GEM) tracker data files (atomic).

    GEM distributes its trackers (Global Iron and Steel Tracker, Global Cement Tracker,
    Global Iron Ore Mines Tracker, etc.) as CC-BY-4.0 Excel files behind a lightweight
    web download *form* (name/email/use-case). That form is NOT an authenticated
    credential gate: the page ships a **public** Supabase "publishable" key and the flow is
    (1) POST the form fields to the ``mint_submission`` RPC (with the public key) to get a
    short-lived ``capability_token``, (2) POST that token to the ``presign`` function to get
    presigned object URLs, (3) GET each URL. There is no email verification, so the whole
    flow is automatable with the embedded public key -- i.e. an open, if form-wrapped,
    download. ``requested_slugs`` are the GEM download slugs (e.g.
    ``["iron-steel-plant-tracker"]``, read from the ``<gem-download-form slugs=...>`` element
    on the project's download page). ``contact`` supplies the download-form fields; it must
    include at least ``name`` and ``email`` (the caller should collect these from the user,
    e.g. via CLI args); ``organization``, ``sector``, ``country`` and ``use_case`` default to
    generic values when omitted. Returns the list of downloaded file paths.
    """
    import json

    dst_dir = UPath(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    contact = {
        "organization": "",
        "sector": "Academic / Research",
        "country": "",
        "use_case": (
            "Academic research building a georeferenced label corpus of industrial "
            "facilities for self-supervised pretraining of Earth observation foundation "
            "models on Sentinel-2/Sentinel-1/Landsat imagery; only facility locations are "
            "used, with CC-BY attribution."
        ),
        **contact,
    }
    if not contact.get("name") or not contact.get("email"):
        raise ValueError(
            "download_gem_tracker requires contact 'name' and 'email' "
            "(collect these from the user, e.g. via CLI args)."
        )
    payload = {
        "name": contact["name"],
        "email": contact["email"],
        "organization": contact["organization"],
        "sector": contact["sector"],
        "country": contact["country"],
        "use_case": contact["use_case"],
        "license_text": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "email_optin": False,
        "request_mode": "slugs",
        "useragent": "Mozilla/5.0",
        "page_url": "https://globalenergymonitor.org/",
        "requested_slugs": list(requested_slugs),
    }
    req = urllib.request.Request(
        mint_url,
        data=json.dumps(payload).encode(),
        headers={
            "content-type": "application/json",
            "apikey": supabase_key,
            "authorization": f"Bearer {supabase_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310
        mint = json.loads(r.read())
    token = mint["capability_token"]
    req2 = urllib.request.Request(
        presign_url,
        data=b"{}",
        headers={
            "content-type": "application/json",
            "authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req2, timeout=timeout) as r:  # nosec B310
        urls = json.loads(r.read())["urls"]

    out: list[UPath] = []
    for u in urls:
        fname = u.get("filename") or (u.get("slug", "gem") + ".xlsx")
        dst = dst_dir / fname
        out.append(
            download_http(u["url"], dst, skip_existing=skip_existing, timeout=timeout)
        )
    return out


def hf_download(
    repo_id: str, filename: str, dst_dir: str | UPath, repo_type: str = "dataset"
) -> UPath:
    """Download a file from the Hugging Face hub (public repos)."""
    from huggingface_hub import hf_hub_download

    dst_dir = UPath(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(  # nosec B615
        repo_id=repo_id, filename=filename, repo_type=repo_type, local_dir=dst_dir.path
    )
    return UPath(path)
