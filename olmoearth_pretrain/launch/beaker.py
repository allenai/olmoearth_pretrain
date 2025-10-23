"""Launching entrypoint for OlmoEarth Pretrain Beaker experiments."""

from dataclasses import dataclass

from beaker import ExperimentSpec
from olmo_core.launch.beaker import BeakerLaunchConfig

from olmoearth_pretrain._compat import deprecated_class_alias as _deprecated_class_alias


@dataclass
class OlmoEarthBeakerLaunchConfig(BeakerLaunchConfig):
    """Extend BeakerLaunchConfig with hostnames option.

    This enables targeting specific Beaker hosts.
    """

    hostnames: list[str] | None = None

    def build_experiment_spec(
        self, torchrun: bool = True, entrypoint: str | None = None
    ) -> ExperimentSpec:
        """Build the experiment spec."""
        # We simply call the superclass build_experiment_spec, but just replace cluster
        # setting in the Constraints with hostname setting if user provided hostname
        # list.
        spec = super().build_experiment_spec(torchrun, entrypoint)
        if self.hostnames:
            constraints = spec.tasks[0].constraints
            constraints.cluster = None
            constraints.hostname = self.hostnames
        return spec


HeliosBeakerLaunchConfig = _deprecated_class_alias(
    OlmoEarthBeakerLaunchConfig,
    "helios.internal.experiment.HeliosBeakerLaunchConfig",
)
