"""Common types for OlmoEarth Pretrain."""

from typing import TypeAlias

import numpy as np
import torch

ArrayTensor: TypeAlias = np.ndarray | torch.Tensor
