"""NISAR L2 GCOV rslearn data source that skips pre-calibration (BETA) granules.

OEDatasets indexes both the BETA (Oct 2025 - Jan 2026, product v1.0.2) and PROVISIONAL
GCOV collections without a property telling them apart, so BETA items are recognized
by "GCOV_BETA" in their asset URLs. BETA reads ~0.5 dB (HH) / ~1.3 dB (HV) low relative
to PROVISIONAL reprocessings of the same acquisitions, and some track-frames have
multi-dB range ramps, so they are left out of pretraining data.

Requires olmoearth_run (only imported by rslearn when a dataset config names this class).
"""

from olmoearth_run.runner.tools.rslearn_data_sources.olmoearth_datasets.nisar.l2_gcov import (
    NisarL2Gcov,
)
from olmoearth_shared.models.datasets.item import Item as ApiItem


def is_beta(api_item: ApiItem) -> bool:
    """Whether an OEDatasets NISAR item is a BETA (pre-calibration) granule."""
    return any(
        "GCOV_BETA" in provider.model_dump_json()
        for provider in api_item.data_providers.values()
    )


class NisarL2GcovNoBeta(NisarL2Gcov):
    """NisarL2Gcov that treats BETA granules as missing their assets.

    The filter runs through skip_items_missing_assets, which must stay True.
    """

    def _item_has_all_bands(self, api_item: ApiItem) -> bool:
        if is_beta(api_item):
            return False
        return super()._item_has_all_bands(api_item)
