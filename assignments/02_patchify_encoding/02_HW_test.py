from WIS_DSP_lib.dataloader import Dataset, DataLoader
from HW02 import PatchEmbeddings, CompositeEncoding
from WIS_DSP_lib.constants import *
from WIS_DSP_lib.olmo_helper import get_dataloader, modality, modality_spec, MAX_SEQUENCE_LENGTH, BASE_GSD
from olmoearth_pretrain.nn.flexi_vit import MultiModalPatchEmbeddings, CompositeEncodings
from olmoearth_pretrain.nn.tokenization import TokenizationConfig
from WIS_DSP_lib.test_helpers import assert_tensor_values_equal

seed = 3622

#%% Run the olmoearth_pretrain code 
set_reproducible_seeds(seed)

# load batch
data_loader, dataset = get_dataloader()
data_iterator = iter(data_loader)
batch = next(data_iterator)

# build patch embeddings
token_config = TokenizationConfig()
patch_embeddings = MultiModalPatchEmbeddings(
            supported_modality_names=[modality],
            max_patch_size=8,
            embedding_size=128,
            tokenization_config=token_config,
            band_dropout_rate=0,
            random_band_dropout=False,
            band_dropout_modalities=None,
        )

# patchify
patches = patch_embeddings(batch[1], patch_size=8)

# build encodings
encodings = CompositeEncodings(
        embedding_size=128,
        supported_modalities=[modality_spec],
        max_sequence_length=MAX_SEQUENCE_LENGTH,
        learnable_channel_embeddings=True,
        random_channel_embeddings=False,
        tokenization_config=token_config,
    )

# encode
output_olmo = encodings.forward(
            {modality: patches[modality]},
            batch[1].timestamps,
            8,
            BASE_GSD,
        )

#%% Run our refactored code
set_reproducible_seeds(seed)
dataset = Dataset(h5py_dir=DATA_DIR)
data_loader = DataLoader(
        dataset=dataset,
        batch_size=GLOBAL_BATCH_SIZE,
    )
batch = next(iter(data_loader))

patch_embeddings = PatchEmbeddings(
    patch_size=8,
    embedding_size=128,
    modality=MODALITY,
)

encodings = CompositeEncoding(MODALITY, embedding_size=128)
patches = patch_embeddings(batch[0], modality=MODALITY)
output = patches[MODALITY] + encodings(patches)

assert_tensor_values_equal(
        output_olmo[MODALITY],
        output,
        "composite_encoding_output",
    )
