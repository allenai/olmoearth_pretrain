"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from helios.evals.datasets.configs import TaskType
from helios.nn.flexihelios import Encoder, PoolingType, TokensAndMasks
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader,
    task_type: TaskType,
    model: Encoder,
    patch_size: int,
    pooling_type: PoolingType = PoolingType.MAX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []

    model = model.eval()
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
            # log the predictor and encoder
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # TODO: Model expects masked helios sample we need to pass empty masks
                # Likely we want to have a flag that checks for eval mode and passes empty masks
                # I want to be able to pass in extra tokens to the decoder only here
                batch_embeddings: TokensAndMasks = model.encoder(
                    masked_helios_sample, patch_size=patch_size
                )[0]  # (bsz, dim)
                # unshard the decoder because we are not in the hooked fsdp path
                model.decoder.unshard()
                batch_embeddings = model.decoder.predictor_pooling(
                    batch_embeddings, masked_helios_sample.timestamps, patch_size
                )
            # get the shape of the batcc embeddings
            averaged_embeddings = batch_embeddings.mean(dim=tuple(range(1, batch_embeddings.ndim -1)))
            logger.warning(f"averaged_embeddings shape: {averaged_embeddings.shape}")

            # spatial_pool = True if task_type == TaskType.SEGMENTATION else False
            # averaged_embeddings = batch_embeddings.pool_unmasked_tokens(
            #     pooling_type, spatial_pooling=spatial_pool
            # )
            embeddings.append(averaged_embeddings.cpu())
            labels.append(label)
            logger.debug(f"Processed {i} / {total_samples}")

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels
