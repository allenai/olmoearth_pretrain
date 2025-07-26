"""Embeddings from models."""

import logging
import math
import torch

from torch.utils.data import DataLoader

from helios.evals.datasets.configs import TaskType
from helios.nn.flexihelios import Encoder, PoolingType, TokensAndMasks
from helios.train.masking import MaskedHeliosSample
from helios.data.constants import Modality
from einops import rearrange, repeat
import torch.nn.functional as F
from helios.evals.panopticon.panopticon import PanopticonWrapper
from helios.evals.dinov2.dinov2 import DINOv2Wrapper
from torchvision import transforms

logger = logging.getLogger(__name__)

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean = IMAGENET_DEFAULT_MEAN,
    std = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


def get_embeddings(
    data_loader: DataLoader,
    task_type: TaskType,
    model: Encoder,
    patch_size: int,
    pooling_type: PoolingType = PoolingType.MAX,
    concat_features: bool = False,
    apply_imagenet_normalization: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []
    torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
    # torchhub_id = "dinov2_vitb14"
    # model = torch.hub.load("facebookresearch/dinov2", torchhub_id)
    model = DINOv2Wrapper(apply_imagenet_normalization=apply_imagenet_normalization, use_cls_token=True)

    model = model.eval()
    # move model to GPU
    model = model.to(device="cuda")
    device = next(model.parameters()).device
    total_samples = len(data_loader)
    with torch.no_grad():
        for i, (masked_helios_sample, label) in enumerate(data_loader):
            masked_helios_sample_dict = masked_helios_sample.as_dict(return_none=False)
            for key, val in masked_helios_sample_dict.items():
                if key == "timestamps":
                    masked_helios_sample_dict[key] = val.to(device=device)
                else:
                    masked_helios_sample_dict[key] = val.to(
                        device=device, dtype=torch.bfloat16
                    )

            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )
            spatial_pool = task_type == TaskType.SEGMENTATION
            is_multi_timestep = masked_helios_sample.timestamps.shape[1] > 1
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                if spatial_pool:
                    # Intermediate features are not yet working because of some bug internal to the model
                    batch_embeddings = model.forward_features(masked_helios_sample)
                    print(f"shape of batch_embeddings: {batch_embeddings.shape}")
                else:
                    batch_embeddings = model(masked_helios_sample)
                    print(f"shape of batch_embeddings: {batch_embeddings.shape}")
            averaged_embeddings = batch_embeddings
            embeddings.append(averaged_embeddings.cpu())
            labels.append(label)
            logger.debug(f"Processed {i} / {total_samples}")

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels
