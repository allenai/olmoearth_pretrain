"""Trainer based on olmo_core."""

from logging import getLogger
from typing import Any

import torch
from einops import rearrange
from olmo_core.train.trainer import Trainer

from helios.data.collator import PerModalityCollatedOutput
from helios.train.loss import patch_disc_loss

logger = getLogger(__name__)


def move_to_device_helios(
    batch: PerModalityCollatedOutput, device: torch.device, non_blocking: bool = True
) -> PerModalityCollatedOutput:
    """Move the batch to the device."""
    return PerModalityCollatedOutput(
        sentinel2=batch.sentinel2.to(device, non_blocking=non_blocking),
        naip=batch.naip.to(device, non_blocking=non_blocking),
        worldcover=batch.worldcover.to(device, non_blocking=non_blocking),
        sample_metadata=batch.sample_metadata,
    )


class HeliosTrainer(Trainer):
    """Trainer for Helios."""

    def model_forward(self, micro_batch: dict[str, Any]) -> torch.Tensor:
        """Run a forward pass on a micro-batch, returning the logits."""
        raise NotImplementedError("model forward helios")

    def get_losses(
        self,
    ) -> dict[str, Any]:
        """Compute the losses for a micro-batch and logits."""
        raise NotImplementedError("get losses helios")

    def eval_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a batch."""
        raise NotImplementedError("eval batch helios")

    def _train_batch(self, batch: PerModalityCollatedOutput) -> None:
        """Train a batch."""
        # Record how many instances are going to be skipped (masked out).
        # if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
        #     self.record_metric("train/masked instances", (~instance_mask).sum(), ReduceType.sum)
        ema_decay = 0.99
        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        # we may want to modify this
        batch = move_to_device_helios(batch, self.device)

        with torch.no_grad():
            s2_data = batch.sentinel2
            input = rearrange(s2_data, "b h w t c -> b c t h w")
            target_output = self.model.target_encoder.forward(input)

        # Run Encoder and decoder on the augmented input
        decoded = self.model.forward(input)
        loss = patch_disc_loss(
            pred=decoded,
            target=target_output,
            pred2unit=True,
        )

        # Backpropagate and optimize
        loss.backward()

        # Update target encoder with EMA
        with torch.no_grad():
            for param, target_param in zip(
                self.model.encoder.parameters(), self.model.target_encoder.parameters()
            ):
                target_param.data = (
                    ema_decay * target_param.data + (1 - ema_decay) * param.data
                )
        # Run through callbacks.
        for callback in self.callbacks.values():
            callback.pre_optim_step()

        # Optimizer step.
        self.optim.step()
        # if isinstance(self.optim, SkipStepOptimizer):
        #     self.record_metric(OPTIM_STEP_SKIPPED_METRIC, self.optim.step_skipped)

        # Run through callbacks.
        for callback in self.callbacks.values():
            callback.post_train_batch()

    def _dry_run_batch(self) -> None:
        logger.info("dry run batch helios")

    def _fit_epoch(self) -> None:
        """Copied almost directly from olmo_core.train.trainer.Trainer._fit_epoch.

        but removing the seq len logging metric collection
        """
        self.data_loader.reshuffle(self.epoch)

        logger.info(f"Starting epoch {self.epoch}...")

        for callback in self.callbacks.values():
            callback.pre_epoch()

        first_batch = True
        for batch in self._iter_batches():
            # Bookkeeping.
            self.global_step += 1
            # self.global_train_tokens_seen += self._validate_batch(batch)

            # self.record_metric(SEQ_LEN_METRIC, float(batch["input_ids"].shape[1]))

            for callback in self.callbacks.values():
                callback.pre_step(batch)

            self._train_batch(batch)

            for callback in self.callbacks.values():
                callback.post_step()

            if first_batch or self.global_step % self.metrics_collect_interval == 0:
                self._log_metrics()
                if torch.cuda.is_available():
                    torch.cuda.set_sync_debug_mode("warn")

            first_batch = False

            if self.training_complete:
                # Finishing before the epoch is complete.
                # Log any remaining metrics.
                self._log_metrics()
                return

        # Log any remaining metrics.
        self._log_metrics()

        logger.info("Epoch complete")

        for callback in self.callbacks.values():
            callback.post_epoch()

        # Bookkeeping
        self.epoch += 1
        self.data_loader.reset()
