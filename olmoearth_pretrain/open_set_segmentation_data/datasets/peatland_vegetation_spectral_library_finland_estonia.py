"""Peatland Vegetation Spectral Library (Finland & Estonia) -> point-table labels.

Source: Mendeley Data 3866tj3w8v v1 (Salko et al. 2024, CC-BY-4.0),
https://data.mendeley.com/datasets/3866tj3w8v/1 . 446 georeferenced 1 m x 1 m field
plots in 13 hemiboreal/boreal/sub-Arctic/Arctic peatland sites in Finland (323 plots)
and Estonia (123 plots), measured in the 2022 and 2023 growing seasons. Each plot has a
WGS84 lon/lat, a survey date, a Finnish peatland-classification type, per-plot plant
functional type (PFT) fractional cover (%), and tree basal areas, plus a full
350-2500 nm reflectance spectrum (not needed for the label signal).

This is a pure sparse in-situ POINT dataset -> one dataset-wide points.geojson (spec 2a),
NOT per-point GeoTIFFs.

Primary label (classification): coarse peatland ecohydrological class derived from the
Finnish peatland type -- {bog, fen, palsa_mire}. This is the manifest's stated goal
("bog-vs-fen peatland vegetation classes") and, being coarse, keeps every one of the 446
plots contributing (downstream min-count filtering would drop most of the 39 fine Finnish
types). The fine Finnish type, the mire structural group, and the PFT cover fractions /
tree basal area are carried as AUXILIARY per-point properties (as coastbench/gloria do),
so a finer classification or a cover-fraction regression can still be built downstream.

Time range: quasi-static peatland vegetation -> a 1-year window anchored on each plot's
survey year (2022 or 2023). change_time = null.

Run: python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.peatland_vegetation_spectral_library_finland_estonia
"""

import argparse
import csv
from collections import Counter
from typing import Any

from olmoearth_pretrain.open_set_segmentation_data import io, manifest

SLUG = "peatland_vegetation_spectral_library_finland_estonia"
RAW_CSV = "raw.csv"  # Reflectance_spectra..._raw.csv (has all metadata + PFT columns)

# --- Primary class scheme: coarse peatland ecohydrological type. ---
CLASSES = [
    (
        "bog",
        "Ombrotrophic (rain-fed, nutrient-poor, Sphagnum-dominated) peatland: Finnish "
        "rahkaneva/rahkarame (Sphagnum fuscum bog), isovarpurame (dwarf-shrub pine bog), "
        "tupasvillaneva/-rame (cottongrass bog), lyhytkorsineva & kalvakkaneva (short-sedge "
        "/ Sphagnum papillosum bog); plus Estonian pine-covered 'Rame' and treeless 'Neva' "
        "plots (raised-bog vegetation, typed by tree cover on the Estonian sites).",
    ),
    (
        "fen",
        "Minerotrophic (groundwater/surface-water-fed) peatland: Finnish sedge fens "
        "(saraneva, rimpineva), rich fens (letto, rimpiletto), flood/riparian fens "
        "(luhta, luhtaneva) and spruce/hardwood mires (korpi types: sarakorpi, ruohokorpi, "
        "lehtokorpi, etc.); plus Estonian flood-influenced treeless 'Neva_luhtainen' plots.",
    ),
    (
        "palsa_mire",
        "Palsa mire: ombrotrophic peat mounds/plateaus with a perennially frozen "
        "(permafrost) core, sub-Arctic/Arctic (Finnish 'Kumpupalsa'). Kept separate from "
        "bog as a distinct permafrost landform.",
    ),
]
NAME_TO_ID = {name: i for i, (name, _d) in enumerate(CLASSES)}

# Exact Finnish_peatland_type string -> coarse class. Covers all 39 observed types.
TYPE_TO_CLASS = {
    # --- bog (ombrotrophic) ---
    "RaN rahkaneva": "bog",
    "RaN rahkaneva kulju": "bog",
    "RaR rahkarame": "bog",
    "IR isovarpurame": "bog",
    "TvN tupasvillaneva": "bog",
    "TvR tupasvillarame": "bog",
    "LkN lyhytkorsineva": "bog",
    "LhN lyhytkorsineva": "bog",
    "LhkN lyhytkorsikalvakkaneva": "bog",
    "Rame": "bog",  # Estonian pine-covered peatland (raised-bog vegetation)
    "Neva": "bog",  # Estonian treeless bog expanse
    # --- fen (minerotrophic) ---
    "VSN varsinainen saraneva": "fen",
    "VSN varsinainen saraneva tupasvillainen": "fen",
    "VRiN varsinainen rimpineva": "fen",
    "VRiN varsinainen rimpineva_valipinta": "fen",
    "RhRiN ruohoinen rimpineva": "fen",
    "RiL rimpiletto": "fen",
    "VL varsinainen letto": "fen",
    "LuN luhtaneva": "fen",
    "Lu luhta": "fen",
    "LuSN luhtainen saraneva": "fen",
    "RhLu ruoholuhta": "fen",
    "RhSN ruohoinen saraneva": "fen",
    "RhSR ruohoinen sararame": "fen",
    "VSR varsinainen sararame": "fen",
    "VSK varsinainen sarakorpi": "fen",
    "SK sarakorpi": "fen",
    "RhSK ruohoinen sarakorpi": "fen",
    "RhSK ruohoinen sarakorpi rimpi": "fen",
    "RhK ruohokorpi": "fen",
    "LK lehtokorpi": "fen",
    "MkK metsakortekorpi": "fen",
    "KsK karhunsammalkorpi": "fen",
    "TvK tupasvillakorpi": "fen",
    "KR korpirame": "fen",
    "Neva_luhtainen": "fen",  # Estonian flood-influenced treeless mire
    # --- palsa ---
    "Kumpupalsa": "palsa_mire",
    "Kumpupalsa_pieni": "palsa_mire",
}


def _mire_structure(ftype: str) -> str:
    """Coarse structural mire group from the Finnish type name (aux field)."""
    t = ftype.lower()
    if "palsa" in t:
        return "palsa"
    if "luhta" in t or "luhtainen" in t:
        return "luhta"  # flood/riparian mire
    if "letto" in t:
        return "letto"  # rich fen
    if "korpi" in t or ftype.strip().endswith("K") or " sarakorpi" in t:
        return "korpi"  # spruce/hardwood mire
    if "rame" in t:
        return "rame"  # pine mire
    if "neva" in t:
        return "neva"  # open/treeless mire
    return "other"


def _num(s: str) -> float | None:
    s = s.replace(" ", "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def read_records(raw_path: str) -> list[dict[str, Any]]:
    with open(raw_path, encoding="utf-8") as f:
        lines = f.readlines()
    # First 3 rows are citation/reading info; row index 3 is the column header.
    rows = list(csv.reader(lines[3:]))
    hdr = rows[0]
    idx = {h: i for i, h in enumerate(hdr)}
    data = rows[1:]
    recs: list[dict[str, Any]] = []
    for r in data:
        ftype = r[idx["Finnish_peatland_type"]].strip()
        lat = _num(r[idx["Coordinate_y"]])
        lon = _num(r[idx["Coordinate_x"]])
        if lat is None or lon is None:
            continue
        cls = TYPE_TO_CLASS.get(ftype)
        if cls is None:
            raise ValueError(f"unmapped peatland type: {ftype!r}")
        date = r[idx["Date"]].strip()  # DD/MM/YYYY
        year = int(date.split("/")[-1])
        ba_living = sum(
            (_num(r[idx[c]]) or 0.0)
            for c in ("BA_Pine_living", "BA_Spruce_living", "BA_Deciduous_living")
        )
        recs.append(
            {
                "lon": lon,
                "lat": lat,
                "label_name": cls,
                "year": year,
                "source_id": r[idx["Plot_ID"]].strip(),
                # auxiliary properties
                "country": r[idx["Country"]].strip(),
                "site": r[idx["Site"]].strip(),
                "finnish_peatland_type": ftype,
                "mire_structure": _mire_structure(ftype),
                "pft_sphagnum": _num(r[idx["PFT_sphagnum_mosses"]]),
                "pft_graminoids": _num(r[idx["PFT_graminoids"]]),
                "pft_woody_stemmed": _num(r[idx["PFT_woody_stemmed"]]),
                "pft_brown_mosses": _num(r[idx["PFT_brown_mosses"]]),
                "pft_herbaceous": _num(r[idx["PFT_herbaceous"]]),
                "pft_lichen": _num(r[idx["PFT_lichen"]]),
                "pft_bare_peat": _num(r[idx["PFT_bare_peat"]]),
                "pft_water": _num(r[idx["PFT_water"]]),
                "tree_basal_area_living": ba_living,
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw_path = str(raw / RAW_CSV)
    recs = read_records(raw_path)
    print(f"read {len(recs)} plots with valid coords")

    # Tiny dataset (446): well under 1000/class and the 25k cap -> keep every plot,
    # no balancing/truncation needed.
    points = []
    for i, r in enumerate(recs):
        p = {
            "id": f"{i:06d}",
            "lon": r["lon"],
            "lat": r["lat"],
            "label": NAME_TO_ID[r["label_name"]],
            "time_range": io.year_range(r["year"]),
            "change_time": None,
            "source_id": r["source_id"],
        }
        for k in (
            "country",
            "site",
            "finnish_peatland_type",
            "mire_structure",
            "pft_sphagnum",
            "pft_graminoids",
            "pft_woody_stemmed",
            "pft_brown_mosses",
            "pft_herbaceous",
            "pft_lichen",
            "pft_bare_peat",
            "pft_water",
            "tree_basal_area_living",
        ):
            p[k] = r[k]
        points.append(p)
    io.write_points_table(SLUG, "classification", points)

    counts = Counter(r["label_name"] for r in recs)
    type_counts = Counter(r["finnish_peatland_type"] for r in recs)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": "Peatland Vegetation Spectral Library (Finland & Estonia)",
            "task_type": "classification",
            "source": "Mendeley Data (10.17632/3866tj3w8v.1)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "https://data.mendeley.com/datasets/3866tj3w8v/1",
                "have_locally": False,
                "annotation_method": "field survey + ASD FieldSpec 4 spectroradiometer; "
                "peatland type per Finnish classification (Laine et al. 2012), "
                "PFT cover from near-nadir photo grid.",
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "classes": [
                {"id": i, "name": name, "description": desc}
                for i, (name, desc) in enumerate(CLASSES)
            ],
            "nodata_value": io.CLASS_NODATA,
            "num_samples": len(points),
            "class_counts": {name: counts.get(name, 0) for name, _ in CLASSES},
            "finnish_type_counts": dict(sorted(type_counts.items())),
            "auxiliary_properties": [
                "country",
                "site",
                "finnish_peatland_type",
                "mire_structure",
                "pft_sphagnum",
                "pft_graminoids",
                "pft_woody_stemmed",
                "pft_brown_mosses",
                "pft_herbaceous",
                "pft_lichen",
                "pft_bare_peat",
                "pft_water",
                "tree_basal_area_living",
            ],
            "notes": (
                "446 in-situ 1x1 m peatland plots (Finland 323, Estonia 123), 2022-2023. "
                "Primary label = coarse ecohydrological class {bog, fen, palsa_mire} mapped "
                "from the 39 Finnish peatland types. Fine type + mire_structure + PFT cover "
                "fractions + living tree basal area carried as auxiliary point properties. "
                "Estonian sites are typed only by tree cover (neva=treeless, rame=pine, "
                "korpi=spruce) with no explicit trophic status: 'Rame'/'Neva' mapped to bog "
                "(Estonian raised-bog vegetation), 'Neva_luhtainen' to fen; these ~123 "
                "Estonian assignments carry more uncertainty than the Finnish trophic types. "
                "1-year time range on each plot's survey year; change_time=null."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="classification", num_samples=len(points)
    )
    print(f"wrote {len(points)} points; class counts: {dict(counts)}")


if __name__ == "__main__":
    main()
