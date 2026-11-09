"""Assemble viz/index.html from page_template.html, viz/data.json and the probe results."""

import datetime
import json
from pathlib import Path

HERE = Path(
    "/weka/dfive-default/agent/piperbots/2026-10-29-piperw-bathymetry-direct-supervision/bluetopo_eval"
)
results = json.loads((HERE / "results_holdout.json").read_text())
all_cells = json.loads((HERE / "results_all.json").read_text())
summary = (HERE.parent / "downstream" / "summary.json").read_text()
page = (HERE / "page_template.html").read_text()
page = page.replace("/*DATA*/null", (HERE / "viz/data.json").read_text())
page = page.replace("/*SCALING*/null", json.dumps(results["probe"]))
page = page.replace("/*PAIRED*/null", json.dumps(results["paired"]))
page = page.replace(
    "/*ALLCELLS*/null",
    json.dumps({k: all_cells[k] for k in ("n_cells", "probe", "paired")}),
)
page = page.replace("/*SUMMARY*/null", summary)
page = page.replace("/*BUILT*/", datetime.date.today().isoformat())
(HERE / "viz/index.html").write_text(page)
print("wrote viz/index.html")
