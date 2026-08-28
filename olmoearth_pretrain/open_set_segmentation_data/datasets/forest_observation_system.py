"""Process the Forest Observation System (FOS) into a point-table regression dataset (AGB).

Source: the Forest Observation System (FOS; http://forest-observation-system.net/), an
international initiative that compiles a global in-situ reference of forest aboveground
biomass (AGB) and canopy height from permanent research plots, to calibrate/validate EO
biomass products (Schepaschenko et al. 2019, Sci Data 6:198, doi:10.1038/s41597-019-0196-1).

The live portal requires free registration for the extended, accurately-geolocated data,
but the peer-reviewed Sci Data data package is OPEN (CC-BY-4.0) and hosted by IIASA:
  https://pure.iiasa.ac.at/id/eprint/17619/1/FOS_data_v2019.04.10.zip
  (DOI 10.22022/ESM/03-2019.38). No credential required.
It ships one Excel workbook (FOS_data_v2019.04.10.xlsx) with a "Plots" sheet of 1,645
sub-plots across 274 plots. Each sub-plot row carries the plot center lon/lat, census
year, plot area, AGB estimates (Mg/ha) under three allometric schemes (local / Chave 2014
eq.7 / Feldpausch), Lorey's and max canopy height (m), wood density, basal area, stem
density, etc. In this OPEN package coordinates are rounded to 2 decimal places (~1 km at
the equator); the accurately-geolocated version is portal-only (registration).

REGRESSION TARGET -- plot-mean aboveground biomass (AGB), Mg/ha (= t/ha).
We use AGB_local (local allometric equations / Chave 2014 eq.4 with local H-D curves),
the dataset's most direct estimate. AGB_Chave, AGB_Feldpausch and both canopy-height
metrics are carried per point as auxiliary properties (canopy height is the secondary
quantity noted in the manifest; the regression block holds AGB only).

Plot-level aggregation (not sub-plot): the OPEN package rounds coordinates to 2 dp, so all
sub-plots of a plot collapse onto the same ~1 km cell (240/274 plots share a single 2dp
coord; up to 136 sub-plots at one coord). Emitting sub-plots would put contradictory AGB
values on the same 10 m pixel, so we aggregate each plot to one point at its center with
the plot-mean AGB (FOS itself recommends "use a plot average" to avoid within-plot spatial
autocorrelation). This yields 247 plots with a valid plot-mean AGB and coordinates.

Each label is a single point with a continuous value -> REGRESSION written to a
dataset-wide point table (points.geojson, spec 2a), NOT per-point GeoTIFFs. 247 samples is
well under the 5000-sample regression cap, so all valid plots are kept (no downsampling,
no bucket-balancing needed); the AGB distribution is only mildly right-skewed.

Time range: plots censused 2016+ get a 1-year window on their census year; plots censused
before 2016 (median census ~2010) get a representative Sentinel-era 1-year window (2016,
the earliest full Sentinel-2 year, minimizing the gap to the field measurement). Per the
task, AGB at established permanent research plots is slowly-varying, so anchoring pre-2016
plots to an early Sentinel window is acceptable -- this is the main caveat (documented in
the summary), alongside the ~1 km coordinate rounding of the open package.

Run:
  python3 -m olmoearth_pretrain.open_set_segmentation_data.datasets.forest_observation_system
"""

import argparse

import numpy as np
import pandas as pd

from olmoearth_pretrain.open_set_segmentation_data import download, io, manifest

SLUG = "forest_observation_system"
NAME = "Forest Observation System"

DOWNLOAD_URL = "https://pure.iiasa.ac.at/id/eprint/17619/1/FOS_data_v2019.04.10.zip"
ZIP_NAME = "FOS_data_v2019.04.10.zip"
XLSX_REL = "FOS_data_v2019.04.10/FOS_data_v2019.04.10.xlsx"

# Plots censused in 2016+ use their census year; earlier plots (biomass slowly-varying at
# permanent plots) anchor to the earliest full Sentinel-2 year to minimize the temporal gap.
SENTINEL_START_YEAR = 2016
MAX_REGRESSION = 5000

NUM_COLS = [
    "AGB_local",
    "AGB_Chave",
    "AGB_Feldpausch",
    "H_Lorey_local",
    "H_max_Local",
    "Lat_cnt",
    "Lon_cnt",
    "Year_Census",
    "Plot_Area",
]


def load_plots() -> pd.DataFrame:
    """Return one plot-aggregated AGB record per FOS plot (quality filtered)."""
    xlsx = io.raw_dir(SLUG) / XLSX_REL
    df = pd.read_excel(xlsx.path, "Plots")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _mode_year(s: pd.Series) -> float:
        s = s.dropna()
        if s.empty:
            return np.nan
        m = s.mode()
        return float(m.iloc[0]) if len(m) else float(s.median())

    agg = df.groupby("Plot_ID").agg(
        lat=("Lat_cnt", "mean"),
        lon=("Lon_cnt", "mean"),
        agb=("AGB_local", "mean"),
        agb_chave=("AGB_Chave", "mean"),
        agb_feldpausch=("AGB_Feldpausch", "mean"),
        h_lorey=("H_Lorey_local", "mean"),
        h_max=("H_max_Local", "mean"),
        year=("Year_Census", _mode_year),
        area_ha=("Plot_Area", "sum"),
        n_subplots=("Sub-plot_ID", "size"),
        country=("Country_Name", "first"),
        network=("Network", "first"),
    )
    agg = agg.reset_index()
    # Valid plot-mean AGB (Mg/ha, positive) with real coordinates.
    agg = agg.dropna(subset=["agb", "lat", "lon", "year"])
    agg = agg[(agg["agb"] > 0)]
    agg = agg[agg["lat"].between(-90, 90) & agg["lon"].between(-180, 180)]
    return agg.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=MAX_REGRESSION)
    args = parser.parse_args()

    io.check_disk()
    manifest.write_registry_entry(SLUG, "in_progress")

    raw = io.raw_dir(SLUG)
    raw.mkdir(parents=True, exist_ok=True)
    zip_path = download.download_http(DOWNLOAD_URL, raw / ZIP_NAME, timeout=600)
    download.extract_zip(zip_path, raw / "FOS_data_v2019.04.10")
    with (raw / "SOURCE.txt").open("w") as f:
        f.write(
            "Forest Observation System (FOS).\n"
            f"OPEN Sci Data package (CC-BY-4.0): {DOWNLOAD_URL}\n"
            "DOI: https://doi.org/10.22022/ESM/03-2019.38 (IIASA eprint 17619)\n"
            "paper: Schepaschenko et al. 2019, Sci Data 6:198, "
            "https://doi.org/10.1038/s41597-019-0196-1\n"
            "portal (extended, accurately-geolocated; free registration): "
            "http://forest-observation-system.net/\n"
            "regression target: plot-mean aboveground biomass AGB_local (Mg/ha), "
            f"from {XLSX_REL} sheet 'Plots'\n"
            "NOTE: coordinates in this open package are rounded to 2 dp (~1 km).\n"
        )

    df = load_plots()
    print(f"{len(df)} valid plot-mean AGB records (of 274 FOS plots)")

    # 247 << 5000; keep all valid plots (no downsampling / bucket-balancing needed).
    if len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
        print(f"downsampled to {len(df)}")

    points = []
    for i, r in df.iterrows():
        year = int(round(float(r["year"])))
        win_year = year if year >= SENTINEL_START_YEAR else SENTINEL_START_YEAR
        points.append(
            {
                "id": f"{i:06d}",
                "lon": float(r["lon"]),
                "lat": float(r["lat"]),
                "label": float(r["agb"]),
                "time_range": io.year_range(win_year),
                "change_time": None,
                "source_id": f"FOS_plot_{r['Plot_ID']}",
                # auxiliary plot properties (spec 2a extra props).
                "census_year": year,
                "agb_chave_mg_ha": (
                    float(r["agb_chave"]) if pd.notna(r["agb_chave"]) else None
                ),
                "agb_feldpausch_mg_ha": (
                    float(r["agb_feldpausch"])
                    if pd.notna(r["agb_feldpausch"])
                    else None
                ),
                "canopy_height_lorey_m": (
                    float(r["h_lorey"]) if pd.notna(r["h_lorey"]) else None
                ),
                "canopy_height_max_m": (
                    float(r["h_max"]) if pd.notna(r["h_max"]) else None
                ),
                "plot_area_ha": float(r["area_ha"]) if pd.notna(r["area_ha"]) else None,
                "n_subplots": int(r["n_subplots"]),
                "country": str(r["country"]),
                "network": str(r["network"]),
            }
        )
    io.write_points_table(SLUG, "regression", points)

    vals = np.array([p["label"] for p in points], dtype=float)
    hist_counts, hist_edges = np.histogram(vals, bins=np.arange(0, 900, 100))
    n_recent = sum(1 for p in points if p["census_year"] >= SENTINEL_START_YEAR)
    io.write_dataset_metadata(
        SLUG,
        {
            "dataset": SLUG,
            "name": NAME,
            "task_type": "regression",
            "source": "Forest Observation System (FOS); IIASA Sci Data package (DOI 10.22022/ESM/03-2019.38)",
            "license": "CC-BY-4.0",
            "provenance": {
                "url": "http://forest-observation-system.net/",
                "have_locally": False,
                "annotation_method": (
                    "in-situ permanent forest research plots (0.25 ha sub-plots); tree "
                    "DBH/height/wood-density field measurements converted to AGB via "
                    "allometric equations (BIOMASS R-package; Chave et al. 2014)"
                ),
            },
            "sensors_relevant": ["sentinel2", "sentinel1", "landsat"],
            "regression": {
                "name": "aboveground_biomass",
                "description": (
                    "Plot-mean aboveground biomass (AGB) of live trees, Mg/ha (= t/ha), "
                    "at 0.25 ha permanent research plots. AGB_local: estimated using "
                    "local allometric equations or Chave et al. (2014) eq.4 with wood "
                    "density, DBH and tree height from local height-diameter "
                    "relationships (FOS / BIOMASS R-package). Aggregated to one plot-mean "
                    "value per plot. Secondary quantity available in the source but not "
                    "regressed here: canopy height (Lorey's / max, m), carried per point "
                    "as canopy_height_lorey_m / canopy_height_max_m."
                ),
                "unit": "Mg/ha (t/ha)",
                "dtype": "float32",
                "value_range": [float(vals.min()), float(vals.max())],
                "nodata_value": io.REGRESSION_NODATA,
            },
            "num_samples": len(points),
            "value_histogram": {
                "bin_edges": [float(e) for e in hist_edges],
                "counts": [int(c) for c in hist_counts],
            },
            "notes": (
                "Point-table regression (spec 2a); label = plot-mean AGB_local (Mg/ha). "
                "Source: OPEN Sci Data package of the Forest Observation System "
                "(Schepaschenko et al. 2019; IIASA DOI 10.22022/ESM/03-2019.38, "
                "CC-BY-4.0) -- the live FOS portal's accurately-geolocated extended set "
                "needs free registration (no credential in .env), but this peer-reviewed "
                "package is fully open. 274 plots / 1,645 sub-plots; aggregated to plot "
                f"level ({len(points)} plots with valid plot-mean AGB>0 and coords) "
                "because the open package rounds coordinates to 2 dp (~1 km) so all "
                "sub-plots of a plot fall on one ~1 km cell (240/274 plots share a single "
                "coord); FOS recommends using a plot average. All valid plots kept (247 "
                "<< 5000 cap; no downsampling/bucket-balancing; distribution only mildly "
                "right-skewed). AGB via three allometries in source (local/Chave/"
                "Feldpausch); we regress AGB_local and attach agb_chave_mg_ha, "
                "agb_feldpausch_mg_ha, and canopy heights as auxiliary properties. "
                f"Time range: {n_recent} plots censused >=2016 use their census-year "
                "window; earlier plots (median census ~2010) anchor to a 1-year 2016 "
                "window (earliest full Sentinel-2 year), justified because AGB at "
                "established permanent plots is slowly-varying. CAVEATS: (1) ~1 km "
                "coordinate rounding in the open package -- points are approximately "
                "located relative to a 10 m pixel (the portal has exact geolocation); "
                "(2) most census dates predate Sentinel-2, so pre-2016 plots rely on the "
                "quasi-static-AGB assumption."
            ),
        },
    )
    manifest.write_registry_entry(
        SLUG, "completed", task_type="regression", num_samples=len(points)
    )
    print(f"done num_samples={len(points)} task_type=regression")


if __name__ == "__main__":
    main()
