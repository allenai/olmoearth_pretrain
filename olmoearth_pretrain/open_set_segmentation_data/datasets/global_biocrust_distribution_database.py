"""Triage the Global Biocrust Distribution Database (SOIL/Copernicus 2024).

Manifest describes this as global georeferenced biocrust occurrence/cover point records
(3848 entries, drylands) with dominant-group composition, which would be a sparse point
classification/presence dataset (-> points.json, spec 2a).

REALITY (verified by downloading and inspecting the only publicly accessible copy): the
article supplement ``biocrust_database.xlsx`` (Wang et al., SOIL 10, 763-2024) has exactly
4 columns -- ID, biocrust_cover, ai (aridity index), arid_gradient -- and 3848 data rows.
There are NO latitude/longitude coordinates and NO biocrust-type composition columns. The
underlying georeferenced point locations are "unpublished data (Ning Chen et al.)" and are
not distributed. Without lon/lat the records cannot be placed on the Sentinel-2 grid, so
this is a "no recoverable geocoordinates" rejection (spec section 8).

This script downloads the supplement to raw/, verifies the absence of coordinates, and
records a ``rejected`` registry entry. It is idempotent.

Reproduce: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.global_biocrust_distribution_database
"""

import io as _io
import re
import xml.etree.ElementTree as ET
import zipfile
from urllib.request import urlopen

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "global_biocrust_distribution_database"
SUPPLEMENT_URL = (
    "https://soil.copernicus.org/articles/10/763/2024/soil-10-763-2024-supplement.zip"
)
_A = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _inspect_xlsx(xlsx_bytes: bytes) -> tuple[list[str], int]:
    """Return (header column names, number of data rows) from the first worksheet."""
    z = zipfile.ZipFile(_io.BytesIO(xlsx_bytes))
    ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    shared = [
        "".join(t.text or "" for t in si.iter(f"{_A}t"))
        for si in ss_root.findall(f"{_A}si")
    ]
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = sheet.find(f"{_A}sheetData").findall(f"{_A}row")

    def cell(c: ET.Element) -> str | None:
        v = c.find(f"{_A}v")
        if v is None:
            return None
        return shared[int(v.text)] if c.get("t") == "s" else v.text

    header = [cell(c) for c in rows[0].findall(f"{_A}c")] if rows else []
    return [h for h in header if h is not None], max(len(rows) - 1, 0)


def main() -> None:
    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = raw / "soil-10-763-2024-supplement.zip"
    if not zip_path.exists():
        data = urlopen(SUPPLEMENT_URL, timeout=120).read()
        with zip_path.open("wb") as f:
            f.write(data)
    with zip_path.open("rb") as f:
        supp = zipfile.ZipFile(_io.BytesIO(f.read()))
    xlsx_name = next(n for n in supp.namelist() if n.endswith(".xlsx"))
    header, n_rows = _inspect_xlsx(supp.read(xlsx_name))
    print(f"supplement {xlsx_name}: {n_rows} data rows, columns={header}")

    coord_cols = [
        h for h in header if re.search(r"lat|lon|coord|geo|x|y", h, re.IGNORECASE)
    ]
    has_coords = any(
        re.search(r"lat|^y$|latitude", h, re.IGNORECASE) for h in header
    ) and any(re.search(r"lon|^x$|longitude", h, re.IGNORECASE) for h in header)

    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Global Biocrust Distribution Database (Wang et al., SOIL 10, 763-2024).\n"
            f"Supplement: {SUPPLEMENT_URL}\n"
            f"biocrust_database.xlsx columns: {header}\n"
            f"data rows: {n_rows}\n"
            f"coordinate columns detected: {coord_cols or 'NONE'}\n"
        )

    if has_coords:
        raise RuntimeError(
            "Unexpected: coordinate columns present; revisit -- this dataset was "
            "triaged as reject on the assumption of no coordinates."
        )

    reason = (
        "needs-georeferencing: released supplement biocrust_database.xlsx has only "
        "[ID, biocrust_cover, ai, arid_gradient] with NO lon/lat (point coordinates are "
        "unpublished data, Ning Chen et al.); records cannot be placed on the S2 grid."
    )
    print("REJECT:", reason)
    manifest.write_registry_entry(SLUG, "rejected", notes=reason)


if __name__ == "__main__":
    main()
