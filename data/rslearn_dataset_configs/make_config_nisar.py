"""Write config_nisar.json: monthly NISAR L2 GCOV mosaics, ascending and descending.

Layers nisar_asc_mo01..12 and nisar_desc_mo01..12 use the same monthly offsets as
config_sentinel1.json (month m covers time_range[0] + (m - 7) * 30 days for 30 days).
Data comes from the olmoearth_run NisarL2Gcov source (OEDatasets search + ranged reads
from ASF, needs OEDATASETS_API_URL, DATASETS_API_TOKEN and EARTHDATA_CREDENTIALS).
Only HH+HV scenes are used (skip_items_missing_assets drops VV-only and HH-only ones).

Run from this directory: python make_config_nisar.py
"""

import json

LAYERS = {}
# jsonargparse parses the OrbitDirection enum by member name, not value.
for direction, prefix in [("ASCENDING", "nisar_asc"), ("DESCENDING", "nisar_desc")]:
    for month in range(1, 13):
        LAYERS[f"{prefix}_mo{month:02d}"] = {
            "type": "raster",
            "alias": prefix,
            "band_sets": [{"bands": ["HHHH", "HVHV"], "dtype": "float32"}],
            "data_source": {
                "class_path": "olmoearth_run.runner.tools.rslearn_data_sources.olmoearth_datasets.nisar.l2_gcov.NisarL2Gcov",
                "ingest": False,
                "duration": "30d",
                "time_offset": f"{(month - 7) * 30}d",
                "init_args": {
                    "query": {
                        "instrument": {"eq": "L"},
                        "orbit_direction": {"eq": direction},
                    },
                    "timeout": "0:2:0",
                    # Node-local cache of granule blocks (HDF5 metadata is re-read
                    # for every window otherwise).
                    "cache_dir": "/tmp/nisar_gcov_cache",
                },
            },
        }

CONFIG = {
    "layers": LAYERS,
    "window_data_storage": {
        "class_path": "rslearn.dataset.window_data_storage.per_layer.PerLayerStorageFactory"
    },
}

if __name__ == "__main__":
    with open("config_nisar.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
        f.write("\n")
