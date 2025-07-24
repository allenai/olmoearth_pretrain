"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from helios.evals.datasets.configs import TaskType
from helios.nn.flexihelios import Encoder, PoolingType, TokensAndMasks
from helios.train.masking import MaskedHeliosSample
from einops import rearrange
import torch.nn.functional as F
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
    torchhub_id = "dinov2_vitb14"
    model = torch.hub.load("facebookresearch/dinov2", torchhub_id)
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
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # TODO: Model expects masked helios sample we need to pass empty masks
                # Likely we want to have a flag that checks for eval mode and passes empty masks
                # batch_embeddings: TokensAndMasks = model(
                #     masked_helios_sample, patch_size=patch_size
                # )[0]  # (bsz, dim)
                # create rgb only s2 data
                s2_data = masked_helios_sample.sentinel2_l2a
                # DinoV2 exbects B, C , H , W
                # channels first

                s2_data = rearrange(s2_data, "b h w t c -> b (c t) h w")
                s2_data = s2_data[:, [3,2,1], :, :]
                # logger.info(f"s2_data: {s2_data.shape}")                # probably need to apply the dinov2 style normalization
                # Resize the image to 224x224
                original_height = s2_data.shape[2]
                if original_height < 224:
                    image_size = 224
                    s2_data = F.interpolate(s2_data, size=(image_size, image_size), mode="bilinear", align_corners=False)
                else:
                    # move to nearest multiple of 14
                    image_size = ((original_height // 14) - 1) * 14
                    s2_data = F.interpolate(s2_data, size=(image_size, image_size), mode="bilinear", align_corners=False)
                if apply_imagenet_normalization:
                    print("Applying imagenet normalization")
                    # normalize the data
                    normalize_transform = make_normalize_transform()
                    s2_data = normalize_transform(s2_data)
                batch_embeddings_dict = model.forward_features(s2_data)
                # cls_token = batch_embeddings_dict['x_norm_clstoken']
                # batch_embeddings = (cls_token + batch_embeddings) / 2

                # dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks'])
            spatial_pool = task_type == TaskType.SEGMENTATION
            if spatial_pool:
                batch_embeddings = batch_embeddings_dict['x_norm_patchtokens']
                patch_size = 14
                height = image_size // patch_size # Number of patches in the height dimension
                batch_embeddings = rearrange(batch_embeddings, "b (h w) d -> b h w d", h=height, w=height)
                logger.info(f"batch_embeddings: {batch_embeddings.shape}")
            else:
                batch_embeddings = batch_embeddings_dict['x_norm_patchtokens'].mean(dim=1) #["x_norm_clstoken"]
            # Concat features across modalities in space averaged across time
            # averaged_embeddings = batch_embeddings.pool_unmasked_tokens(
            #     pooling_type,
            #     spatial_pooling=spatial_pool,
            #     concat_features=concat_features,
            # )
            averaged_embeddings = batch_embeddings
            embeddings.append(averaged_embeddings.cpu())
            labels.append(label)
            logger.debug(f"Processed {i} / {total_samples}")

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels
