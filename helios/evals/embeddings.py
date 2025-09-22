"""Embeddings from models."""

import logging

import torch
from einops import rearrange
from torch.utils.data import DataLoader

from helios.evals.datasets.configs import TaskType
from helios.evals.eval_wrapper import AnySatEvalWrapper, EvalWrapper
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader, model: EvalWrapper
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []
    model.eval()
    device = model.device
    total_samples = len(data_loader)
    with torch.no_grad():
        for i, (masked_helios_sample, label) in enumerate(data_loader):
            masked_helios_sample_dict = masked_helios_sample.as_dict(return_none=False)
            for key, val in masked_helios_sample_dict.items():
                if key == "timestamps":
                    masked_helios_sample_dict[key] = val.to(device=device)
                else:
                    masked_helios_sample_dict[key] = val.to(
                        device=device,
                    )

            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                batch_embeddings = model(masked_helios_sample)

            if isinstance(model, AnySatEvalWrapper):
                if model.task_type == TaskType.SEGMENTATION:
                    # this is a special case for AnySat. Since it outputs per-pixel embeddings,
                    # we subsample training pixels to keep the memory requirements reasonable.
                    # From https://arxiv.org/abs/2502.09356:
                    # """
                    # for semantic segmentation, the AnySat features are per-pixel
                    # instead of per-patch. For comparable training cost, we sam-
                    # ple 6.25% of its pixel features per image when training, but
                    # evaluate with all pixel features when testing. We confirmed
                    # the fairness of this evaluation with the the AnySat authors
                    # by personal communication.
                    # """
                    subsample_by = 1 / 16
                    batch_embeddings = rearrange(
                        batch_embeddings, "b h w d -> b (h w) d"
                    )
                    label = rearrange(label, "b h w -> b (h w)")

                    assert batch_embeddings.shape[1] == label.shape[1]
                    num_tokens = batch_embeddings.shape[1]
                    num_tokens_to_keep = int(num_tokens * subsample_by)
                    sampled_indices = torch.randperm(num_tokens)[:num_tokens_to_keep]
                    batch_embeddings = batch_embeddings[:, sampled_indices]
                    label = label[:, sampled_indices]

                    new_hw = int(num_tokens_to_keep**0.5)
                    # reshape to h w
                    batch_embeddings = rearrange(
                        batch_embeddings, "b (h w) d -> b h w d", h=new_hw, w=new_hw
                    )
                    label = rearrange(label, "b (h w) -> b h w", h=new_hw, w=new_hw)
            embeddings.append(batch_embeddings.cpu())
            labels.append(label)
            logger.info(f"Processed {i} / {total_samples}")

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels
