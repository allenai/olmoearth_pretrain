"""Process the LUCAS Land Use/Cover Survey (Eurostat / JRC) into an open-set-segmentation
point table.

LUCAS is the EU-wide in-situ ground survey of land cover (LC) and land use (LU). Each
record is a georeferenced field point with a manually observed LC1 land-cover class. This
is a pure sparse-point classification dataset (spec 2a/4 "points"), so we write ONE
dataset-wide points.geojson, not per-point GeoTIFFs.

Post-2016 (Sentinel era) surveys only:
- 2018: harmonised LUCAS DB (d'Andrimont et al. 2020), file ``lucas_harmo_uf_2018.csv``
  inside ``lucas_harmo_uf_2018.zip``. Columns ``lc1``/``lc1_label``, ``gps_lat``/``gps_long``
  (field GPS) and ``th_lat``/``th_long`` (theoretical grid point).
- 2022: LUCAS 2022 Copernicus survey table ``l2022_survey_cop_radpoly_attr.csv``. Columns
  ``survey_lc1`` ("CODE - Label"), ``survey_gps_lat``/``survey_gps_long`` and
  ``point_lat``/``point_long`` (theoretical).

Coordinate choice (per task): use the observed field **GPS** point where it is valid;
otherwise fall back to the theoretical grid coordinate (many "In office PI"
photo-interpreted points have no field GPS and carry the 88.888.. sentinel).

Classes: LC1 land-cover level (detailed 3-char codes, e.g. B11 = common wheat). Class ids
are assigned 0..N-1 in descending combined frequency; the uint8 254-class cap is honored
(LUCAS LC1 has ~76 codes, well under the cap). Balanced to <=1000/class subject to the
25k per-dataset cap (spec 5).

Run: ``python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.lucas_land_use_cover_survey``
"""

import argparse
import csv
import io as _io
import re
import zipfile
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest
from olmoearth_pretrain.open_set_segmentation_data.sampling import balance_by_class

SLUG = "lucas_land_use_cover_survey"
PER_CLASS = 1000
FILE_2018 = "lucas_harmo_uf_2018.zip"
FILE_2022 = "l2022_survey_cop_radpoly_attr.csv"

# Valid LC1 land-cover code: letter A-H + two chars (digit or X), e.g. A11, B11, BX1, C10,
# H23. Filters out the "8 - Not relevant" / blank placeholders.
LC1_RE = re.compile(r"^[A-H][0-9X][0-9X]$")


def _to_float(s: str | None) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _valid_coord(lat: float | None, lon: float | None) -> bool:
    """Plausible EU land coordinate; rejects None and the 88.888.. no-GPS sentinel."""
    if lat is None or lon is None:
        return False
    if not (-90.0 < lat < 84.0 and -180.0 < lon < 180.0):
        return False
    if abs(lat - 88.888) < 0.5 or abs(lon - 88.888) < 0.5:
        return False
    if lat == 0.0 and lon == 0.0:
        return False
    return True


def _pick_coord(
    gps_lat: float | None,
    gps_lon: float | None,
    th_lat: float | None,
    th_lon: float | None,
) -> tuple[float, float, str] | None:
    """Prefer the field GPS point; fall back to the theoretical grid point."""
    if _valid_coord(gps_lat, gps_lon):
        return gps_lat, gps_lon, "gps"  # type: ignore[return-value]
    if _valid_coord(th_lat, th_lon):
        return th_lat, th_lon, "theoretical"  # type: ignore[return-value]
    return None


def scan_2018(raw_dir) -> tuple[list[dict[str, Any]], dict[str, str]]:
    path = raw_dir / FILE_2018
    recs: list[dict[str, Any]] = []
    code2label: dict[str, str] = {}
    with zipfile.ZipFile(path.path) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            r = csv.reader(_io.TextIOWrapper(fh, encoding="utf-8", errors="replace"))
            hdr = next(r)
            idx = {h: i for i, h in enumerate(hdr)}
            for row in r:
                if len(row) < len(hdr):
                    continue
                code = row[idx["lc1"]].strip()
                if not LC1_RE.match(code):
                    continue
                code2label.setdefault(code, row[idx["lc1_label"]].strip())
                coord = _pick_coord(
                    _to_float(row[idx["gps_lat"]]),
                    _to_float(row[idx["gps_long"]]),
                    _to_float(row[idx["th_lat"]]),
                    _to_float(row[idx["th_long"]]),
                )
                if coord is None:
                    continue
                lat, lon, csrc = coord
                recs.append(
                    {
                        "lon": lon,
                        "lat": lat,
                        "code": code,
                        "year": 2018,
                        "coord_src": csrc,
                        "point_id": row[idx["point_id"]],
                    }
                )
    return recs, code2label


def scan_2022(raw_dir) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Parse the 2022 Copernicus survey table.

    The file carries a trailing multi-line, unquoted ``radpoly`` polygon-geometry blob that
    breaks whole-file CSV parsing, so we parse per record: every field we need (point_id,
    coords, survey_lc1, survey_date) lives on a record's FIRST physical line. A record-start
    line has an all-digit ``point_id`` before its first comma; geometry continuation lines
    start with a decimal, so ``token.isdigit()`` cleanly discriminates them.
    """
    path = raw_dir / FILE_2022
    recs: list[dict[str, Any]] = []
    code2label: dict[str, str] = {}
    with open(path.path, encoding="utf-8", errors="replace") as fh:
        hdr = next(csv.reader([fh.readline()]))
        idx = {h: i for i, h in enumerate(hdr)}
        i_lc, i_gla, i_glo = (
            idx["survey_lc1"],
            idx["survey_gps_lat"],
            idx["survey_gps_long"],
        )
        i_pla, i_plo, i_pid, i_dt = (
            idx["point_lat"],
            idx["point_long"],
            idx["point_id"],
            idx["survey_date"],
        )
        for line in fh:
            if not line.split(",", 1)[0].isdigit():
                continue  # geometry continuation line, not a record start
            row = next(csv.reader([line]))
            if len(row) <= i_lc:
                continue
            raw_lc = row[i_lc].strip()
            code = raw_lc.split(" - ", 1)[0].strip()
            if not LC1_RE.match(code):
                continue
            if " - " in raw_lc:
                code2label.setdefault(code, raw_lc.split(" - ", 1)[1].strip())
            coord = _pick_coord(
                _to_float(row[i_gla]),
                _to_float(row[i_glo]),
                _to_float(row[i_pla]),
                _to_float(row[i_plo]),
            )
            if coord is None:
                continue
            lat, lon, csrc = coord
            dt = row[i_dt][:4] if len(row) > i_dt else ""
            year = int(dt) if dt.isdigit() and dt.startswith("202") else 2022
            recs.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "code": code,
                    "year": year,
                    "coord_src": csrc,
                    "point_id": row[i_pid],
                }
            )
    return recs, code2label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    recs_18, lab_18 = scan_2018(raw)
    recs_22, lab_22 = scan_2022(raw)
    recs = recs_18 + recs_22
    # Prefer harmonised (2018) labels; fill any 2022-only codes.
    code2label = dict(lab_22)
    code2label.update(lab_18)
    print(f"2018 recs={len(recs_18)}  2022 recs={len(recs_22)}  total={len(recs)}")

    # Assign class ids 0..N-1 in descending combined frequency (uint8; <=254 cap).
    freq = Counter(r["code"] for r in recs)
    ordered_codes = sorted(freq, key=lambda c: (-freq[c], c))
    if len(ordered_codes) > 254:
        dropped = ordered_codes[254:]
        ordered_codes = ordered_codes[:254]
        keep = set(ordered_codes)
        recs = [r for r in recs if r["code"] in keep]
        print(f"254-class cap: kept top 254, dropped {len(dropped)} codes")
    code2id = {c: i for i, c in enumerate(ordered_codes)}
    for r in recs:
        r["label"] = code2id[r["code"]]

    selected = balance_by_class(recs, "label", per_class=PER_CLASS)
    print(f"selected {len(selected)} points (<= {PER_CLASS}/class, 25k cap)")

    points = []
    for i, r in enumerate(selected):
        points.append(
            {
                "id": f"{i:06d}",
                "lon": r["lon"],
                "lat": r["lat"],
                "label": r["label"],
                "time_range": io.year_range(r["year"]),
                "source_id": f"{r['year']}/{r['point_id']}/{r['coord_src']}",
            }
        )
    io.write_points_table(SLUG, "classification", points)

    sel_counts = Counter(r["code"] for r in selected)
    coord_counts = Counter(r["coord_src"] for r in selected)
    year_counts = Counter(r["year"] for r in selected)
    classes = [
        {
            "id": code2id[c],
            "name": f"{c} - {code2label.get(c, c)}",
            "description": None,
        }
        for c in ordered_codes
    ]
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "LUCAS Land Use/Cover Survey",
            "task_type": "classification",
            "source": "Eurostat / JRC",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://ec.europa.eu/eurostat/web/lucas ; https://essd.copernicus.org/articles/13/1119/2021/",
                "have_locally": False,
                "annotation_method": "manual in-situ field survey (LC1 land cover)",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": classes,
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(selected),
            "class_counts": {c: sel_counts.get(c, 0) for c in ordered_codes},
            "notes": (
                "LUCAS LC1 land-cover level; class name = 'CODE - label'. Post-2016 surveys "
                "only: 2018 (harmonised DB) + 2022 (Copernicus survey table). Coordinate is "
                "the field GPS point where valid, else the theoretical grid point "
                f"(coord_src in source_id). Selected coord sources: {dict(coord_counts)}. "
                f"Selected by year: {dict(year_counts)}. 1-year time range per survey year. "
                "Sparse single-pixel points -> points.geojson (spec 2a)."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(selected)
    )
    print("done")


if __name__ == "__main__":
    main()
