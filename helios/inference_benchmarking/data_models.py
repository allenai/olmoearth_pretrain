"""Core data models for defining throughput runs."""

import os
import re
from dataclasses import dataclass

from helios.inference_benchmarking import constants


@dataclass
class RunParams:
    """Defines the parameters for a throughput run."""

    model_size: str
    use_s1: bool
    use_s2: bool
    use_landsat: bool
    image_size: int
    patch_size: int
    num_timesteps: int
    batch_sizes: list[int]
    gpu_type: str = "cpu"
    bf16: bool = False
    benchmark_interval_s: int = 180
    min_batches_per_interval: int = 10

    @property
    def run_name(self) -> str:
        """Generates a string representing the run."""
        return "_".join(
            [
                item
                for item in [
                    self.model_size,
                    self.gpu_type,
                    "bf16" if self.bf16 else None,
                    "s1" if self.use_s1 else None,
                    "s2" if self.use_s2 else None,
                    "ls" if self.use_landsat else None,
                    f"is{self.image_size}",
                    f"ps{self.patch_size}",
                    f"ts{self.num_timesteps}",
                    f"bs{'_'.join([str(bs) for bs in self.batch_sizes])}",
                ]
                if item is not None
            ]
        )

    def to_env_vars(self) -> dict[str, str]:
        """Prepares env vars from the run params.

        Object can be recreated from these subsequently.
        """
        keys = constants.PARAM_KEYS
        return {
            keys["checkpoint_path"]: os.path.join(
                "/artifacts", constants.MODEL_SIZE_MAP[self.model_size]
            ),
            keys["model_size"]: self.model_size,
            keys["use_s1"]: str(int(self.use_s1)),
            keys["use_s2"]: str(int(self.use_s2)),
            keys["use_landsat"]: str(int(self.use_landsat)),
            keys["image_size"]: str(self.image_size),
            keys["patch_size"]: str(self.patch_size),
            keys["num_timesteps"]: str(self.num_timesteps),
            keys["batch_sizes"]: ",".join([str(bs) for bs in self.batch_sizes]),
            keys["gpu_type"]: self.gpu_type,
            keys["bf16"]: str(int(self.bf16)),
            keys["benchmark_interval_s"]: str(self.benchmark_interval_s),
            keys["min_batches_per_interval"]: str(self.min_batches_per_interval),
            keys["name"]: self.run_name,
        }

    @staticmethod
    def from_env_vars() -> "RunParams":
        """Recreate an instance of `RunParams` from env vars."""
        keys = constants.PARAM_KEYS
        model_size = os.getenv(keys["model_size"], "Unknown")
        use_s1 = True if os.getenv(keys["use_s1"], "0") == "1" else False
        use_s2 = True if os.getenv(keys["use_s2"], "0") == "1" else False
        use_landsat = True if os.getenv(keys["use_landsat"], "0") == "1" else False
        image_size = int(os.getenv(keys["image_size"], "1"))
        patch_size = int(os.getenv(keys["patch_size"], "1"))
        num_timesteps = int(os.getenv(keys["num_timesteps"], "1"))
        batch_sizes = [
            int(b) for b in os.getenv(keys["batch_sizes"], "1,").split(",") if b
        ]
        gpu_type = os.getenv(keys["gpu_type"], "cpu")
        bf16 = True if os.getenv(keys["bf16"], "0") == "1" else False
        benchmark_interval_s = int(os.getenv(keys["benchmark_interval_s"], "180"))
        min_batches_per_interval = int(os.getenv(keys["min_batches_per_interval"], 10))

        return RunParams(
            model_size=model_size,
            use_s1=use_s1,
            use_s2=use_s2,
            use_landsat=use_landsat,
            image_size=image_size,
            patch_size=patch_size,
            num_timesteps=num_timesteps,
            batch_sizes=batch_sizes,
            gpu_type=gpu_type,
            bf16=bf16,
            benchmark_interval_s=benchmark_interval_s,
            min_batches_per_interval=min_batches_per_interval,
        )

    @staticmethod
    def from_run_name(name: str) -> "RunParams":
        """Recreate an instance of 'RunParams' from a prior run's stringified name."""
        split_name = name.split("_")
        model_size = split_name[0]
        gpu_type = split_name[1]
        use_s1 = "_s1_" in name
        use_s2 = "_s2_" in name
        use_landsat = "_ls_" in name
        bf16 = "_bf16_" in name

        for item in split_name:
            if item.startswith("is"):
                image_size = int(item.replace("is", ""))
            if item.startswith("ps"):
                patch_size = int(item.replace("ps", ""))
            if item.startswith("ts"):
                num_timesteps = int(item.replace("ts", ""))

        batch_size_raw = re.findall(r"bs((?:\d+_)*\d+)", name)[0]
        batch_sizes = [int(bs) for bs in batch_size_raw.split("_")]

        return RunParams(
            model_size=model_size,
            use_s1=use_s1,
            use_s2=use_s2,
            use_landsat=use_landsat,
            image_size=image_size,
            patch_size=patch_size,
            num_timesteps=num_timesteps,
            batch_sizes=batch_sizes,
            gpu_type=gpu_type,
            bf16=bf16,
        )
