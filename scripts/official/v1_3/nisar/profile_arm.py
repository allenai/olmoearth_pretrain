"""Run a v1.3 arm with an olmo_core ProfilerCallback on rank 0 (steps 150-158 traced).

Used by ``profile_<arm>.py`` wrappers; the op tables are logged and the chrome trace is
saved under the run's save folder (``profiler/``).
"""

import importlib.util
import sys
from pathlib import Path

from olmo_core.train.callbacks import ProfilerCallback

V13 = Path(__file__).resolve().parents[1]


def run_profiled(arm_path: str) -> None:
    """Load ``v1_3/<arm_path>.py``, add the profiler to its trainer and run it."""
    spec = importlib.util.spec_from_file_location("profiled_arm", V13 / f"{arm_path}.py")
    assert spec is not None and spec.loader is not None
    arm = importlib.util.module_from_spec(spec)
    sys.modules["profiled_arm"] = arm
    spec.loader.exec_module(arm)
    original = arm.build_trainer_config

    def build_trainer_config(common):
        config = original(common)
        config.callbacks["profiler"] = ProfilerCallback(
            skip_first=150, wait=1, warmup=3, active=5, with_stack=True
        )
        return config

    arm.build_trainer_config = build_trainer_config
    arm.run()
