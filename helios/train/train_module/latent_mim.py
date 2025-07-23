"""Training and optimizer abstraction for Helios."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Any

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd
import torch.nn.functional as F
from olmo_core.distributed.parallel import DataParallelConfig
from olmo_core.distributed.utils import get_local_rank, get_local_tensor
from olmo_core.optim import OptimConfig
from olmo_core.optim.scheduler import Scheduler
from olmo_core.train.common import Duration, ReduceType

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.data.transform import TransformConfig
from helios.nn.flexihelios import TokensAndMasks
from helios.nn.latent_mim import LatentMIM
from helios.train.loss import LossConfig
from helios.train.masking import MaskedHeliosSample, MaskingConfig
from helios.train.train_module.train_module import (
    HeliosTrainModule,
    HeliosTrainModuleConfig,
)
import itertools
from helios.train.utils import split_batch
from einops import rearrange
import time

logger = getLogger(__name__)


@dataclass
class LatentMIMTrainModuleConfig(HeliosTrainModuleConfig):
    """A configuration class for building :class:`LatentMIMTrainModule` instances.

    Args:
        loss_config: The loss configuration for the model.
        masking_config: The masking configuration for the model.
        ema_decay: EMA decay rate for target encoder (default: 0.99).
    """

    loss_config: LossConfig = field(
        default_factory=lambda: LossConfig(loss_config={"type": "patch_discrimination"})
    )
    mae_loss_config: LossConfig | None = None
    masking_config: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(strategy_config={"type": "random"})
    )
    token_exit_cfg: dict[str, int] = field(
        default_factory=lambda: {modality: 0 for modality in Modality.names()}
    )
    warmup_duration: Duration = field(default_factory=lambda: Duration.epochs(2))
    ema_decay: tuple[float, float] = (0.996, 1.0)
    max_grad_norm: float = 1.0

    def build(
        self,
        model: LatentMIM,
        device: torch.device | None = None,
    ) -> "LatentMIMTrainModule":
        """Build the corresponding :class:`LatentMIMTrainModule`.

        Args:
            model: The model to train.
            device: The device to train on.
        """
        kwargs = self.prepare_kwargs()
        return LatentMIMTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class LatentMIMTrainModule(HeliosTrainModule):
    """A :class:`TrainModule`.

    Initialize the training module.

    Args:
        model: The transformer model to train.
        optim: The corresponding optimizer config.
        transform_config: The transform configuration for the model.
        masking_config: The masking configuration for the model.
        loss_config: The loss configuration for the model.
        mae_loss_config: Optional loss config for masked auto-encoding.
        rank_microbatch_size: The rank microbatch size in instances.
        compile_model: Whether to compile to the model.
        dp_config: Data parallel configuration for the model.
        loss_fn: Loss function to use.
        compile_loss: Whether to compile the loss function.
        autocast_precision: Enable AMP with this data type.
        max_grad_norm: Clip gradient norms to this value.
        scheduler: Optional learning rate scheduler.
        device: The device to train on.
        state_dict_save_opts: Override state dict options for saving.
        state_dict_load_opts: Override state dict options for loading.
        token_exit_cfg: The token exit configuration for the model.
        warmup_duration: The warmup duration for the model.
    """

    def __init__(
        self,
        model: LatentMIM,
        optim_config: OptimConfig,
        transform_config: TransformConfig,
        masking_config: MaskingConfig,
        loss_config: LossConfig,
        rank_microbatch_size: int,
        token_exit_cfg: dict[str, int],
        mae_loss_config: LossConfig | None = None,
        compile_model: bool = False,
        dp_config: DataParallelConfig | None = None,
        compile_loss: bool = False,
        autocast_precision: torch.dtype | None = None,
        max_grad_norm: float | None = None,
        scheduler: Scheduler | None = None,
        device: torch.device | None = None,
        state_dict_save_opts: dist_cp_sd.StateDictOptions | None = None,
        state_dict_load_opts: dist_cp_sd.StateDictOptions | None = None,
        ema_decay: tuple[float, float] = (0.996, 1.0),
        warmup_duration: Duration = Duration.epochs(2),
        regularizer_config: LossConfig | None = None,
        find_unused_parameters: bool = True,
    ):
        """Initialize the training module.

        Args:
            model: The transformer model to train.
            optim_config: The corresponding optimizer config.
            transform_config: The transform configuration for the model.
            masking_config: The masking configuration for the model.
            loss_config: The loss configuration for the model.
            rank_microbatch_size: The rank microbatch size in instances.
            compile_model: Whether to compile to the model.
            dp_config: Data parallel configuration for the model.
            loss_fn: Loss function to use.
            compile_loss: Whether to compile the loss function.
            autocast_precision: Enable AMP with this data type.
            max_grad_norm: Clip gradient norms to this value.
            scheduler: Optional learning rate scheduler.
            device: The device to train on.
            state_dict_save_opts: Override state dict options for saving.
            state_dict_load_opts: Override state dict options for loading.
            ema_decay: EMA decay rate for target encoder, as a tuple of (start_ema_decay, end_ema_decay)
            token_exit_cfg: The token exit configuration for the model.
            mae_loss_config: Optional loss config for masked auto-encoding.
            warmup_duration: The warmup duration for the model.
            regularizer_config: An optional regularizer configuration for the model.
            find_unused_parameters: Whether to find unused parameters in the model, only used for DDP.
        """
        super().__init__(
            model=model,
            optim_config=optim_config,
            transform_config=transform_config,
            rank_microbatch_size=rank_microbatch_size,
            compile_model=compile_model,
            dp_config=dp_config,
            compile_loss=compile_loss,
            autocast_precision=autocast_precision,
            max_grad_norm=max_grad_norm,
            scheduler=scheduler,
            device=device,
            state_dict_save_opts=state_dict_save_opts,
            state_dict_load_opts=state_dict_load_opts,
            warmup_duration=warmup_duration,
            find_unused_parameters=find_unused_parameters,
        )
        self.start_ema, self.end_ema = ema_decay
        self.token_exit_cfg = token_exit_cfg
        self.base_loss = loss_config.build()
        self.masking_strategy = masking_config.build()
        self.regularizer = (
            regularizer_config.build() if regularizer_config is not None else None
        )

        self.total_loss_name = self.base_loss.name
        if self.regularizer is not None:
            self.total_loss_name = f"{self.base_loss.name}+{self.regularizer.name}"

        self.mae_loss = mae_loss_config.build() if mae_loss_config is not None else None
        if self.mae_loss is not None:
            self.total_loss_name = f"{self.total_loss_name}+{self.mae_loss.name}"

    def loss_fn(self, pred: Any, targets: Any) -> torch.Tensor:
        """Compute the loss between the predicted and target tensors."""
        return self.base_loss.compute(pred, targets)

    def train_batch(
        self, batch: tuple[int, HeliosSample], dry_run: bool = False
    ) -> None:
        """Train a batch.

        NOTE: Gradient accumulation/microbatching is not invariant for all losses across the same global batch size.

        - All Disc loss with same global batch size but different micro-batch sizes result in different gradients,
        though this matches the implementation in gallileo.
        - If the min hw is too low when subsampling, we may get micro-batches with uneven
        numbers of tokens making the loss for token averaged losses
        like l1 and l2 weight microbatches with less tokens relatively more.

        NOTE: For non contrastive losses, the loss is invariant to the global batch size across GPUS as well
        """
        self.update_target_encoder()
        # Set the model to train mode
        self.model.train()
        total_batch_loss = torch.zeros([], device=self.device)
        total_batch_reg = torch.zeros([], device=self.device)
        patch_size, batch_data = batch
        # log the patch size
        logger.info(f"Patch size: {patch_size}")
        # Split into micro-batches.
        microbatches = split_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(microbatches)
        for microbatch_idx, microbatch in enumerate(microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                print(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch.batch_size} on rank {get_local_rank()}\n"
                )
                microbatch = self.transform.apply(microbatch).to_device(self.device)
                masked_batch = self.masking_strategy.apply_mask(
                    microbatch, patch_size=patch_size
                )
                # Run Encoder and decoder on the augmented input
                loss, latent, decoded, target_output = self.model_forward(
                    masked_batch, patch_size, self.token_exit_cfg, dry_run
                )
                if not dry_run:
                    pass
                    # with torch.no_grad():
                        # # Understand intramodal similarity within a sample
                        # for modality in target_output.modalities:
                        #     if modality == "latlon":
                        #         continue
                        #     data = getattr(target_output, modality)
                        #     if data is None:
                        #         continue
                        #     # logger.info(f"Target output modality {modality} has shape {data.shape}")
                        #     batch_size = data.shape[0]
                        #     mean_sims = []
                        #     min_sims = []
                        #     max_sims = []
                        #     num_high_sims = []
                        #     num_very_high_sims = []
                        #     batch_total_sim = 0
                        #     for i in range(batch_size):
                        #         data_i = data[i]
                        #         data_i = data_i.unsqueeze(0)
                        #         data_i = rearrange(data_i, "b ... d -> b (...) d")
                        #         # similarity distributions of all the different token embeddings
                        #         emb_norm = F.normalize(data_i, p=2, dim=-1)
                        #         emb_sim = torch.einsum("b n d, b m d -> b n m", emb_norm, emb_norm)
                        #         logger.info(f"Embedding similarity matrix shape: {emb_sim.shape}")

                        #         # The diagonal of the similarity matrix should be 1 so we don't want to include it in the analysis we want to just filter it out
                        #         emb_sim = emb_sim[..., torch.eye(emb_sim.shape[-1]) == 0]
                        #         logger.info(f"Embedding similarity matrix shape after removing the diagonal: {emb_sim.shape}")
                        #         # Get mean median and std of the similarity
                        #         mean_sim = emb_sim.mean()
                        #         median_sim = emb_sim.median()
                        #         std_sim = emb_sim.std()
                        #         mean_sims.append(mean_sim)
                        #         logger.info(f"Mean similarity: {mean_sim}, Median similarity: {median_sim}, Std similarity: {std_sim}")
                        #         # log the min and max of the similarity
                        #         min_sim = emb_sim.min()
                        #         max_sim = emb_sim.max()
                        #         min_sims.append(min_sim)
                        #         max_sims.append(max_sim)
                        #         # logger.info(f"Min similarity: {min_sim}, Max similarity: {max_sim}")
                        #         # count the number of similarities that are greater than 0.9
                        #         total_sim = emb_sim.numel()
                        #         batch_total_sim += total_sim
                        #         num_high_sim = (emb_sim > 0.9).sum()
                        #         # logger.info(f"Number of similarities greater than 0.9: {num_high_sim} out of {total_sim}")
                        #         num_very_high_sim = (emb_sim > 0.95).sum()
                        #         num_very_high_sims.append(num_very_high_sim)
                        #         # logger.info(f"Number of similarities greater than 0.95: {num_very_high_sim} out of {total_sim}")
                        #         # log 75 and 90 and 95th percentile of the similarity
                        #         seventy_fifth_percentile = torch.quantile(emb_sim.float(), 0.75)
                        #         ninety_percentile = torch.quantile(emb_sim.float(), 0.9)
                        #         ninety_fifth_percentile = torch.quantile(emb_sim.float(), 0.95)
                        #         # logger.info(f"75th percentile similarity: {seventy_fifth_percentile}, 90th percentile similarity: {ninety_percentile}, 95th percentile similarity: {ninety_fifth_percentile}")
                        #     # Get the mean of the lists
                        #     mean_sims = torch.tensor(mean_sims).mean()
                        #     min_sims = torch.tensor(min_sims).mean()
                        #     max_sims = torch.tensor(max_sims).mean()
                        #     num_very_high_sims = torch.tensor(num_very_high_sims).float().sum()
                        #     percent_very_high_sims = num_very_high_sims / batch_total_sim
                        #     logger.info(f"Mean similarity: {mean_sims}, Min similarity: {min_sims}, Max similarity: {max_sims}, Number of similarities greater than 0.95: {num_very_high_sims} out of {batch_total_sim} total similarities for modality {modality} at percent {percent_very_high_sims}")
                        #     self.trainer.record_metric(
                        #         f"similarity/random_proj_{modality}_mean_sim",
                        #         mean_sims,
                        #         ReduceType.mean,
                        #     )
                        #     self.trainer.record_metric(
                        #         f"similarity/random_proj_{modality}_min_sim",
                        #         min_sims,
                        #         ReduceType.mean,
                        #     )
                        #     self.trainer.record_metric(
                        #         f"similarity/random_proj_{modality}_percent_very_high_sims",
                        #         percent_very_high_sims,
                        #         ReduceType.mean,
                        #     )

                        # Understand intermodal similarity
                        # For every combination of modalities compute and analyze the similarities
                        # modality_combinations = list(itertools.combinations(target_output.modalities, 2))
                        # for modality_combination in modality_combinations:
                        #     modality_1, modality_2 = modality_combination
                        #     data_1 = getattr(target_output, modality_1)
                        #     data_2 = getattr(target_output, modality_2)
                        #     if data_1 is None or data_2 is None:
                        #         continue
                        #     # Compute the similarity between the two modalities
                        #     batch_size = data_1.shape[0]
                        #     combination_total_sim = 0
                        #     combination_mean_sims = []
                        #     combination_min_sims = []
                        #     combination_max_sims = []
                        #     combination_num_high_sims = []
                        #     combination_num_very_high_sims = []
                        #     for i in range(batch_size):
                        #         data_1_i = data_1[i]
                        #         data_2_i = data_2[i]
                        #         data_1_i = rearrange(data_1_i, "b ... d -> b (...) d")
                        #         data_2_i = rearrange(data_2_i, "b ... d -> b (...) d")
                        #         # Compute the similarity between the two modalities
                        #         emb_norm_1 = F.normalize(data_1_i, p=2, dim=-1)
                        #         emb_norm_2 = F.normalize(data_2_i, p=2, dim=-1)
                        #         emb_sim = torch.einsum("b n d, b m d -> b n m", emb_norm_1, emb_norm_2)
                        #         # Get the mean of the similarity
                        #         mean_sim = emb_sim.mean()
                        #         logger.info(f"Mean similarity between {modality_1} and {modality_2}: {mean_sim}")
                        #         combination_mean_sims.append(mean_sim)
                        #         # Get the min of the similarity
                        #         min_sim = emb_sim.min()
                        #         logger.info(f"Min similarity between {modality_1} and {modality_2}: {min_sim}")
                        #         combination_min_sims.append(min_sim)
                        #         # Get the max of the similarity
                        #         max_sim = emb_sim.max()
                        #         logger.info(f"Max similarity between {modality_1} and {modality_2}: {max_sim}")
                        #         combination_max_sims.append(max_sim)
                        #         # Get the 75th percentile of the similarity
                        #         seventy_fifth_percentile = torch.quantile(emb_sim.float(), 0.75)
                        #         logger.info(f"75th percentile similarity between {modality_1} and {modality_2}: {seventy_fifth_percentile}")
                        #         # get the number of similarities that are greater than 0.9
                        #         num_high_sim = (emb_sim > 0.9).sum()
                        #         logger.info(f"Number of similarities greater than 0.9 between {modality_1} and {modality_2}: {num_high_sim}")
                        #         combination_num_high_sims.append(num_high_sim)
                        #         # get the number of similarities that are greater than 0.95
                        #         num_very_high_sim = (emb_sim > 0.95).sum()
                        #         logger.info(f"Number of similarities greater than 0.95 between {modality_1} and {modality_2}: {num_very_high_sim}")
                        #         combination_num_very_high_sims.append(num_very_high_sim)
                        #         # get the total number of similarities
                        #         total_sim = emb_sim.numel()
                        #         combination_total_sim += total_sim
                        #     # Get the mean of the lists
                        #     combination_mean_sims = torch.tensor(combination_mean_sims).mean()
                        #     combination_min_sims = torch.tensor(combination_min_sims).mean()
                        #     combination_max_sims = torch.tensor(combination_max_sims).mean()
                        #     combination_num_high_sims = torch.tensor(combination_num_high_sims).float().sum()
                        #     combination_num_very_high_sims = torch.tensor(combination_num_very_high_sims).float().sum()
                        #     percent_very_high_sims = combination_num_very_high_sims / combination_total_sim
                        #     logger.info(f"Mean similarity between {modality_1} and {modality_2}: {combination_mean_sims}, Min similarity: {combination_min_sims}, Max similarity: {combination_max_sims}, Number of similarities greater than 0.95: {combination_num_very_high_sims} out of {combination_total_sim} total similarities for modality {modality_1} and {modality_2} at percent {percent_very_high_sims}")
                        #     self.trainer.record_metric(
                        #         f"intermodal_similarity/random_proj_intermodal_{modality_1}_{modality_2}_mean_sim",
                        #         combination_mean_sims,
                        #         ReduceType.mean,
                        #     )
                        #     self.trainer.record_metric(
                        #         f"intermodal_similarity/random_proj_intermodal_{modality_1}_{modality_2}_min_sim",
                        #         combination_min_sims,
                        #         ReduceType.mean,
                        #     )
                        #     self.trainer.record_metric(
                        #         f"intermodal_similarity/random_proj_intermodal_{modality_1}_{modality_2}_max_sim",
                        #         combination_max_sims,
                        #         ReduceType.mean,
                        #     )
                        #     self.trainer.record_metric(
                        #         f"intermodal_similarity/random_proj_intermodal_{modality_1}_{modality_2}_percent_very_high_sims",
                        #         percent_very_high_sims,
                        #         ReduceType.mean,
                        #     )

                reg_term = self.compute_regularization(latent)
                if reg_term is not None:
                    loss = loss + reg_term
                    total_batch_reg += (
                        get_local_tensor(reg_term.detach()) / num_microbatches
                    )
                # Scale loss by number of microbatches
                loss = loss / num_microbatches

                loss_val = get_local_tensor(loss.detach())
                total_batch_loss += loss_val

                # Skip bad batches
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    logger.warning(
                        f"NaN or Inf detected in loss at microbatch {microbatch_idx}, stopping training for this batch."
                    )
                    print(f"rank {get_local_rank()} has nan or inf")

                loss.backward()

        self.trainer.record_metric(
            f"train/{self.total_loss_name}",
            total_batch_loss,
            ReduceType.mean,
        )
        self.log_regularization(total_batch_reg)

        if dry_run:
            return

        del batch, batch_data  # In case this helps with memory utilization.
        del masked_batch
        del latent, decoded, target_output

    def model_forward(
        self, batch: MaskedHeliosSample, patch_size: int, token_exit_cfg: dict[str, int], dry_run: bool = False
    ) -> tuple[torch.Tensor, TokensAndMasks, TokensAndMasks, TokensAndMasks]:
        """Run a forward pass."""
        with self._model_forward_context():
            latent, decoded, _, reconstructed = self.model(batch, patch_size)
            with torch.no_grad():
                logger.info("Target Encoder forward pass...")
                target_output, _ = self.model.target_encoder.forward(
                    batch.unmask(),
                    patch_size=patch_size,
                    token_exit_cfg=token_exit_cfg,
                )
            loss, similarities_dict = self.loss_fn(decoded, target_output)
            logger.info(f"Similarities dict: {similarities_dict}")
            import time
            time.sleep(10)
            if not dry_run:
                # record all the similarities for every modality
                for key, per_modality_dict in similarities_dict.items():
                    if not isinstance(per_modality_dict, dict):
                        self.trainer.record_metric(
                            f"similarity/{key}",
                            per_modality_dict,
                            ReduceType.mean,
                        )
                    else:
                        for modality, value in per_modality_dict.items():
                            if modality == "latlon":
                                continue
                            name = f"similarity/{key}_{modality}"
                            logger.info(f"Recording similarity metric {name} with value {value}")
                            self.trainer.record_metric(
                                name,
                                value,
                                ReduceType.mean,
                            )
            if self.mae_loss is not None:
                loss += self.mae_loss.compute(reconstructed, batch)
            return loss, latent, decoded, target_output
