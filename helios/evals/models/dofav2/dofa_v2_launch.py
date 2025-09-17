"""DOFA v2 launch script."""

from helios.evals.models import DOFAv2Config
from helios.evals.models.utils import build_train_module_config, build_dataloader_config, build_dataset_config, build_visualize_config
from helios.internal.experiment import CommonComponents
from helios.nn.latent_mim import LatentMIMConfig

def build_model_config(common: CommonComponents) -> LatentMIMConfig:
    """Build the model config for an experiment."""
    model_config = DOFAv2Config()
    return model_config