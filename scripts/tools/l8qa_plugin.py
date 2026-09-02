"""Landsat data source with the QA_PIXEL band, for the landsat_qa layer set.

rslearn's LandsatOliTirs hardcodes BANDS to the eleven spectral bands, but the
usgs-landsat bucket ships `_QA_PIXEL.TIF` (CFMask cloud/shadow/cirrus/dilated
bit flags) beside them under the same blob prefix, and the asset filename
construction is generic -- so exposing the band is just an extended BANDS
list. Scene selection is untouched: the landsat_qa layers CLONE the prepared
items of their landsat_moNN siblings (setup_extra_layers.py), so the QA
raster always describes exactly the scene the imagery came from.

This module must be importable inside the materialize Beaker job, which runs
the plain rslp image without this repo installed. setup_extra_layers.py
copies it to PLUGIN_WEKA_DIR during ``apply`` and injects
``PYTHONPATH=PLUGIN_WEKA_DIR`` into landsat_qa jobs; the layer config's
data_source.class_path is ``l8qa_plugin.LandsatOliTirsQA``.
"""

from rslearn.data_sources.aws_landsat import LandsatOliTirs


class LandsatOliTirsQA(LandsatOliTirs):
    """LandsatOliTirs with QA_PIXEL available as an asset band."""

    BANDS = LandsatOliTirs.BANDS + ["QA_PIXEL"]
