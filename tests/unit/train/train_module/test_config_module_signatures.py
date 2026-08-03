"""Every train-module config field must be accepted by the module it builds.

``OlmoEarthTrainModuleConfig.prepare_kwargs`` passes the config's fields to the
module's ``__init__`` as keyword arguments, but each subclass restates its whole
signature and forwards explicitly to ``super()``. So adding a field to the BASE
config silently breaks every subclass that does not also restate it -- and because
``prepare_kwargs`` uses ``exclude_none=True``, the break only surfaces at runtime,
on the first run that actually sets the new field. That is exactly how
``scheduler_overrides`` reached a live job before failing with
``__init__() got an unexpected keyword argument``.

This test compares the two signatures directly, so the mismatch is caught at
lint-speed instead of on a GPU.
"""

import dataclasses
import inspect

import pytest

from olmoearth_pretrain.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModule,
    ContrastiveLatentMIMTrainModuleConfig,
)
from olmoearth_pretrain.train.train_module.galileo import (
    GalileoTrainModule,
    GalileoTrainModuleConfig,
)
from olmoearth_pretrain.train.train_module.latent_mim import (
    LatentMIMTrainModule,
    LatentMIMTrainModuleConfig,
)
from olmoearth_pretrain.train.train_module.mae import (
    MAETrainModule,
    MAETrainModuleConfig,
)
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)

#: Pre-existing mismatches, recorded rather than hidden. ``regularizer_config`` is
#: declared on the BASE config but only consumed by the subclasses, so building a
#: plain ``OlmoEarthTrainModule`` with it set would raise. Unreachable in practice
#: (every experiment uses a subclass) and not fixed here because the base module has
#: no regularizer to wire it to -- but it is the same latent defect, so it is listed
#: explicitly and will start failing the moment someone gives the base class one.
KNOWN_GAPS = {"OlmoEarthTrainModuleConfig": {"regularizer_config"}}

CONFIG_MODULE_PAIRS = [
    (OlmoEarthTrainModuleConfig, OlmoEarthTrainModule),
    (LatentMIMTrainModuleConfig, LatentMIMTrainModule),
    (ContrastiveLatentMIMTrainModuleConfig, ContrastiveLatentMIMTrainModule),
    (MAETrainModuleConfig, MAETrainModule),
    (GalileoTrainModuleConfig, GalileoTrainModule),
]


@pytest.mark.parametrize(
    "config_cls,module_cls",
    CONFIG_MODULE_PAIRS,
    ids=[c.__name__ for c, _ in CONFIG_MODULE_PAIRS],
)
def test_every_config_field_is_accepted_by_its_module(
    config_cls: type, module_cls: type
) -> None:
    """No config field may be missing from the module's __init__ signature."""
    params = inspect.signature(module_cls).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        pytest.skip(f"{module_cls.__name__} accepts **kwargs")
    fields = {f.name for f in dataclasses.fields(config_cls)}
    known = KNOWN_GAPS.get(config_cls.__name__, set())
    missing = sorted(fields - set(params) - known)
    assert not missing, (
        f"{config_cls.__name__} declares {missing}, which "
        f"{module_cls.__name__}.__init__ does not accept. prepare_kwargs() passes "
        "config fields as kwargs, so this raises TypeError at construction -- and "
        "only on runs that set the field, since exclude_none=True hides it "
        "otherwise."
    )
