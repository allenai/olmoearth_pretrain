"""Dataset paths configured via environment variables."""

import os

from upath import UPath

# Only available to internal users
_DEFAULTS = {
    "GEOBENCH_DIR": "/weka/dfive-default/presto-geobench/dataset/geobench",
    "CROPHARVEST_DIR": "/weka/dfive-default/presto_eval_sets/cropharvest",
    "BREIZHCROPS_DIR": "/weka/dfive-default/skylight/presto_eval_sets/breizhcrops",
    "MADOS_DIR": "/weka/dfive-default/presto_eval_sets/mados",
    "FLOODS_DIR": "/weka/dfive-default/presto_eval_sets/floods",
    "PASTIS_DIR": "/weka/dfive-default/presto_eval_sets/pastis_r",
    "PASTIS_DIR_ORIG": "/weka/dfive-default/presto_eval_sets/pastis_r_origsize",
    "PASTIS_DIR_PARTITION": "/weka/dfive-default/presto_eval_sets/pastis",
    "SICKLE_DIR": "/weka/dfive-default/presto_eval_sets/sickle",
    "NANDI_DIR": "/weka/dfive-default/rslearn-eai/datasets/crop/kenya_nandi/20250625",
    "AWF_DIR": "/weka/dfive-default/rslearn-eai/datasets/crop/awf_2023",
}

GEOBENCH_DIR = UPath(os.getenv("GEOBENCH_DIR", _DEFAULTS["GEOBENCH_DIR"]))
CROPHARVEST_DIR = UPath(os.getenv("CROPHARVEST_DIR", _DEFAULTS["CROPHARVEST_DIR"]))
BREIZHCROPS_DIR = UPath(os.getenv("BREIZHCROPS_DIR", _DEFAULTS["BREIZHCROPS_DIR"]))
MADOS_DIR = UPath(os.getenv("MADOS_DIR", _DEFAULTS["MADOS_DIR"]))
FLOODS_DIR = UPath(os.getenv("FLOODS_DIR", _DEFAULTS["FLOODS_DIR"]))
PASTIS_DIR = UPath(os.getenv("PASTIS_DIR", _DEFAULTS["PASTIS_DIR"]))
PASTIS_DIR_ORIG = UPath(os.getenv("PASTIS_DIR_ORIG", _DEFAULTS["PASTIS_DIR_ORIG"]))
PASTIS_DIR_PARTITION = UPath(
    os.getenv("PASTIS_DIR_PARTITION", _DEFAULTS["PASTIS_DIR_PARTITION"])
)
SICKLE_DIR = UPath(os.getenv("SICKLE_DIR", _DEFAULTS["SICKLE_DIR"]))
NANDI_DIR = UPath(os.getenv("NANDI_DIR", _DEFAULTS["NANDI_DIR"]))
AWF_DIR = UPath(os.getenv("AWF_DIR", _DEFAULTS["AWF_DIR"]))
