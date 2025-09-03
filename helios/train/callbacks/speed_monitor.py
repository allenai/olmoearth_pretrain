"""Speed monitor callback for the trainer for Helios."""

import logging
import time
from typing import Any

from olmo_core.train.callbacks.speed_monitor import SpeedMonitorCallback

from helios.data.dataset import HeliosSample
from helios.train.train_module.contrastive_latentmim import (
    ContrastiveLatentMIMTrainModule,
)
from helios.train.train_module.galileo import GalileoTrainModule
from helios.train.train_module.latent_mim import LatentMIMTrainModule
from helios.train.train_module.mae import MAETrainModule

logger = logging.getLogger(__name__)


class HeliosSpeedMonitorCallback(SpeedMonitorCallback):
    """Speed monitor callback for the trainer for Helios."""

    priority = 10
    _total_tokens_encoded = 0
    _total_tokens_decoded = 0
    _total_tokens_target_encoder = 0

    def pre_train(self) -> None:
        """Pre-train callback for the speed monitor."""
        super().pre_train()
        train_module = self.trainer.train_module

        self._token_budget = self.trainer.data_loader.token_budget
        if isinstance(
            train_module,
            MAETrainModule | LatentMIMTrainModule | ContrastiveLatentMIMTrainModule,
        ):
            # Unwrap if the model is in DDP
            self._encoder_ratio = train_module.masking_strategy.encode_ratio
            self._decoder_ratio = train_module.masking_strategy.decode_ratio
            logger.warning(
                "Speed monitor callback bases token input based on token budget, "
                "encoder ratio, and decoder ratio"
            )
        elif isinstance(train_module, GalileoTrainModule):
            # Unwrap if the model is in DDP
            self._encoder_ratio = train_module.masking_strategy_a.encode_ratio
            self._decoder_ratio = train_module.masking_strategy_a.decode_ratio
            if train_module.masking_strategy_b.encode_ratio != self._encoder_ratio:
                logger.warning(
                    "Speed monitor callback bases token input based on encoder ratio "
                    "from masking_strategy_a"
                )
            if train_module.masking_strategy_b.decode_ratio != self._decoder_ratio:
                logger.warning(
                    "Speed monitor callback bases token input based on decoder ratio "
                    "from masking_strategy_a"
                )
            logger.warning(
                "Speed monitor callback bases token input based on token budget, "
                "encoder ratio, and decoder ratio"
            )
        else:
            logger.warning(
                "Speed monitor callback only calculates token throughput with "
                "MAETrainModule, LatentMIMTrainModule or GalileoTrainModule"
            )

    def pre_load_batch(self) -> None:
        """Pre-load batch callback for the speed monitor."""
        if hasattr(self, "callback_start_time"):
            self.callback_start_time: float
            # This is based on the assumption that this callback is the first one
            # to run. We are measuring the time between this callback ending in post step and starting the next pre load step.
            self.trainer.record_metric(
                "throughput/callback time (s)",
                time.perf_counter() - self.callback_start_time,
            )
        super().pre_load_batch()

    def pre_step(self, batch: Any) -> None:
        """Pre-step callback for the speed monitor."""
        self._batch_load_time = time.perf_counter() - self._batch_load_start
        if self._first_step:
            # We don't record the first batch since the first one tends to take
            # unusually long.
            return
        _, batch = batch
        # We need token budget times encoder ratio and token budget times decoder ratio
        if isinstance(batch, HeliosSample):
            self._step_tokens_encoded = (
                batch.batch_size * self._encoder_ratio * self._token_budget
            )
            self._step_tokens_decoded = (
                batch.batch_size * self._decoder_ratio * self._token_budget
            )
            self._step_tokens_target_encoder = batch.batch_size * self._token_budget

        self._total_steps += 1
        self._total_tokens_encoded += self._step_tokens_encoded
        self._total_tokens_decoded += self._step_tokens_decoded
        self._total_tokens_target_encoder += self._step_tokens_target_encoder
        self.model_start_time = time.perf_counter()

    def post_step(self) -> None:
        """Post-step callback for the speed monitor."""
        counter = time.perf_counter()
        self.model_end_time = counter

        self.trainer.record_metric(
            "throughput/device/data loading (s)", self._batch_load_time
        )
        self._first_step: bool
        if self._first_step:
            # Now we can start recording.
            self._total_steps = 0
            self._total_tokens = 0
            self._start_time = counter
            self._step_last_logged = counter
            self._first_step = False
            return

        self.model_duration = self.model_end_time - self.model_start_time
        step_time = counter - self._step_last_logged
        total_time = counter - self._start_time
        self._step_last_logged = counter

        bps = 1 / step_time
        bps_avg = self._total_steps / total_time
        # Save BPS average so we can use the beaker callback to estimate time remaining
        self._bps_avg = bps_avg
        data_pct = 100 * self._batch_load_time / step_time
        tps_encoded = self._total_tokens_encoded / step_time
        tps_encoded_avg = self._total_tokens_encoded / total_time
        tps_decoded = self._total_tokens_decoded / step_time
        tps_decoded_avg = self._total_tokens_decoded / total_time
        tps_target_encoder = self._total_tokens_target_encoder / step_time
        tps_target_encoder_avg = self._total_tokens_target_encoder / total_time
        self.trainer.record_metric(
            "throughput/total tokens target encoder-since-restart",
            self._total_tokens_target_encoder,
        )

        self.trainer.record_metric(
            "throughput/total tokens encoded-since-restart", self._total_tokens_encoded
        )
        self.trainer.record_metric(
            "throughput/total tokens decoded-since-restart", self._total_tokens_decoded
        )
        self.trainer.record_metric("throughput/device/TPS Encoded", tps_encoded)
        self.trainer.record_metric(
            "throughput/device/TPS Target Encoder", tps_target_encoder
        )
        self.trainer.record_metric(
            "throughput/device/TPS Target Encoder (estimated avg)",
            tps_target_encoder_avg,
        )
        self.trainer.record_metric(
            "throughput/device/TPS Encoded (estimated avg)", tps_encoded_avg
        )
        self.trainer.record_metric("throughput/device/TPS Decoded", tps_decoded)
        self.trainer.record_metric(
            "throughput/device/TPS Decoded (estimated avg)", tps_decoded_avg
        )
        self.trainer.record_metric("throughput/device/data loading (%)", data_pct)
        self.trainer.record_metric("throughput/device/BPS", bps)
        self.trainer.record_metric("throughput/device/BPS (estimated avg)", bps_avg)
        self.trainer.record_metric(
            "throughput/device/model duration (s)", self.model_duration
        )
        self.trainer.record_metric(
            "throughput/device/model duration (%)", self.model_duration / step_time
        )
        self.callback_start_time = time.perf_counter()
