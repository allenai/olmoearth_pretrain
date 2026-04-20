
from pathlib import Path
import random
import torch
from wandb.util import np

MISSING_VALUE = -99999
MAX_SEQUENCE_LENGTH = 12
IMAGE_TILE_SIZE = 256
BAND_ORDER = {'sentinel2_l2a': ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]}
BANDSETS = {'sentinel2_l2a': [
    [0, 1, 2, 3],
    [4, 5, 6, 7, 8, 9],
    [10, 11],
]
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data/dataset/6/"
    
COMPUTED_NORM_CONFIG_PATH = (
    REPO_ROOT / "olmoearth_pretrain/data/norm_configs/computed.json"
)
GLOBAL_BATCH_SIZE = 3
SEED = 3622
PATCH_SIZE = 8
SAMPLED_HW_P = 8
TOKEN_BUDGET = 2250
MODALITY = "sentinel2_l2a"
class MaskValue:
    ONLINE_ENCODER = 0
    TARGET_ENCODER_ONLY = 1
    DECODER = 2
    MISSING = 3

class RefactorSteps:
    DATALOADER = 1
    PATCHIFY = 2
    ENCODIGS = 3
    REMOVE_MASK = 4
    ATTENTION = 5
    ADD_MASK = 6
    POOL_PROJECT = 7
    ENCODER_OUTPUT = 8
    ENCODER_DECODER = 9
    ADD_MASK = 10
    DECODER_ENCODIGS = 11
    SPLIT_X_Y = 12
    DECODER_ATTENTION = 13
    COMBINE_X_Y = 14
    DECODER_OUTPUT = 15
    TARGET_ENCODER = 16
    DISCRIMINATION_LOSS = 17

def set_reproducible_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
