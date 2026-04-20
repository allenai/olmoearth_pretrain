#%% Imports
import functools

from olmoearth_pretrain.data.dataset import OlmoEarthDataset
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoader
from olmoearth_pretrain.data.collate import collate_double_masked_batched
from olmoearth_pretrain.train.masking import ModalityCrossRandomMaskingStrategy
from olmoearth_pretrain.data.constants import Modality

from upath import UPath
import numpy as np

#%% set global variables

DATA_DIR = 'data/dataset/6/'

#%% Build the dataset
dataset = OlmoEarthDataset(
    h5py_dir=UPath(DATA_DIR),
    dtype=np.float32,
    training_modalities=['sentinel2_l2a'],
    normalize=True,
)

num_samples = int(dataset.h5py_dir.name)
dataset.sample_indices = np.arange(num_samples)
#%% Build the dataloader

# define the masking strategy 
masking_strategy = ModalityCrossRandomMaskingStrategy(
    allow_encoding_decoding_same_bandset=True,
    only_decode_modalities=[
                Modality.WORLDCOVER.name,
                Modality.SRTM.name,
                Modality.OPENSTREETMAP_RASTER.name,
                Modality.WRI_CANOPY_HEIGHT_MAP.name,
                Modality.CDL.name,
                Modality.WORLDCEREAL.name
    ]
)
masking_strategy_b = None

# define the collator with the masking strategy
collator = functools.partial(
    collate_double_masked_batched,
    transform=None,
    masking_strategy=masking_strategy,
    masking_strategy_b=None,
)

# build the dataloader
data_loader = OlmoEarthDataLoader(
            dataset=dataset,
            work_dir='./local_output/checkpoints/anonymous/test_run',
            global_batch_size=3,
            dp_world_size=1,
            dp_rank=0,
            fs_local_rank=0,
            seed=3622,
            shuffle=False,
            num_workers=0,
            target_device_type='mps',
            collator=collator,
            drop_last=True,
            min_patch_size=8,
            max_patch_size=8,
            sampled_hw_p_list=[8],
            token_budget=2250,
            num_dataset_repeats_per_epoch=1,
            transform=None,
            masking_strategy=masking_strategy,
            masking_strategy_b=None,
            num_masked_views=2,
            tokenization_config=None,
        )

# data_loader.reshuffle(epoch=1)
data_loader._global_indices = data_loader._build_global_indices()
data_loader._epoch = 1
data_iterator = iter(data_loader)
batch = next(data_iterator)