# type: ignore
# ruff: noqa
"""CROMA model https://github.com/antofuller/CROMA/blob/main/use_croma.py ."""

import itertools
import logging
import math
import typing
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from olmo_core.config import Config
from torch import einsum, nn
from upath import UPath

from helios.data.constants import Modality
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)

HELIOS_SENTINEL2_BANDS = [
    Modality.SENTINEL2_L2A.band_order.index(b)
    for b in [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
]


class Croma(nn.Module):
    """Class containing the Croma model that can ingest MaskedHeliosSample objects."""

    patch_size: int = 8
    image_resolution: int = 120
    supported_modalities: list[str] = [
        Modality.SENTINEL2_L2A.name,
        Modality.SENTINEL1.name,
    ]

    def __init__(
        self,
        size: str = "base",
        load_directory: str = "/weka/dfive-default/helios/models/croma",
    ):
        """Initialize the Croma wrapper.

        Args:
            size: The model size
            load_directory: The directory to load from
        """
        super().__init__()
        load_dir = UPath(load_directory)
        if size == "base":
            load_path = load_dir / "CROMA_base.pt"
        else:
            load_path = load_dir / "CROMA_large.pt"

        self.model = PretrainedCROMA(
            pretrained_path=str(load_path),
            size=size,
            modality="both",
            image_resolution=self.image_resolution,
        )

    def _process_modality_data(
        self, data: torch.Tensor, modality: str
    ) -> list[torch.Tensor]:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]

        Returns:
            list of tensors of shape [B, C, H, W]
        """
        # Rearrange from "b h w t c -> b (c t) h w" for DinoV2/Panopticon format
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []

        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")

            # Rearrange sen2 data
            if modality == "sentinel2_l2a":
                data_i = data_i[:, HELIOS_SENTINEL2_BANDS, :, :]

            new_height = (
                self.model.patch_size if original_height == 1 else self.image_resolution
            )

            data_i = F.interpolate(
                data_i,
                size=(new_height, new_height),
                mode="bilinear",
                align_corners=False,
            )
            data_list.append(data_i)
        return data_list

    def prepare_input(
        self,
        masked_helios_sample: MaskedHeliosSample,
    ) -> list[dict[str, torch.Tensor]]:
        """Prepare input for the CHROMA model from MaskedHeliosSample."""
        input_data_timesteps: dict[int, dict[str, torch.Tensor]] = {}
        num_modalities = len(masked_helios_sample.modalities)
        for modality in masked_helios_sample.modalities:
            if modality not in self.supported_modalities:
                logger.warning(
                    f"Skipping modality {modality} as it is not in the supported modalities list {self.supported_modalities}"
                )
                continue

            data = getattr(masked_helios_sample, modality)

            if data is None:
                continue

            # Process the modality data
            processed_data = self._process_modality_data(data, modality)
            croma_modality = (
                "optical_images"
                if modality == Modality.SENTINEL2_L2A.name
                else "SAR_images"
            )
            for i, data_i in enumerate(processed_data):
                # start the list if it doesn't exist
                if i not in input_data_timesteps:
                    input_data_timesteps[i] = {}
                input_data_timesteps[i][croma_modality] = data_i

        if not input_data_timesteps:
            raise ValueError("No valid modalities found for processing")
        return [input_data_timesteps[i] for i in sorted(input_data_timesteps.keys())]

    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through croma model."""
        # Configure model modality

        if masked_helios_sample.sentinel2_l2a is not None:
            if masked_helios_sample.sentinel1 is not None:
                self.model.modality = "both"
            else:
                self.model.modality = "optical"
        else:
            self.model.modality = "SAR"

        # Prepare input
        per_timestep_inputs = self.prepare_input(masked_helios_sample)

        output_features = []
        output_keys = {
            "optical": "optical_encodings",
            "SAR": "SAR_encodings",
            "both": "joint_encodings",
        }
        output_key = output_keys[self.model.modality]
        for data in per_timestep_inputs:
            timestep_output = self.model(**data)[output_key]
            if spatial_pool:
                # TODO: maybe right?
                timestep_output = timestep_output.mean(dim=1)
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features


@dataclass
class CromaConfig(Config):
    """olmo_core style config for CromaWrapper."""

    size: str = "base"
    load_directory: str = "/weka/dfive-default/helios/models/croma"

    def build(self) -> Croma:
        """Build the Croma model."""
        return Croma(
            size=self.size,
            load_directory=self.load_directory,
        )


# fmt: off
@typing.no_type_check
class PretrainedCROMA(nn.Module):
    def __init__(
        self,
        pretrained_path="CROMA_base.pt",
        size="base",
        modality="both",
        image_resolution=120,
    ):
        """NOTE: image_resolution is not the spatial, spectral, or temporal resolution. It is the height and width of the image, in pixels.
        E.g., CROMA was pretrained on 120x120px images, hence image_resolution is 120 by default
        """
        super().__init__()
        # check types
        assert type(pretrained_path) == str, (
            f"pretrained_path must be a string, not {type(pretrained_path)}"
        )
        assert type(size) == str, f"size must be a string, not {type(size)}"
        assert type(modality) == str, f"modality must be a string, not {type(modality)}"
        assert type(image_resolution) == int, (
            f"image_resolution must be an int, not {type(image_resolution)}"
        )

        # check values
        assert size in ["base", "large"], (
            f"size must be either base or large, not {size}"
        )
        assert image_resolution % 8 == 0, (
            f"image_resolution must be a multiple of 8, not {image_resolution}"
        )
        assert modality in ["both", "SAR", "optical"], (
            f"modality must be either both, SAR, or optical, not {modality}"
        )

        # warn the user if the path contains a different size than the size parameter
        if size == "base" and "large" in pretrained_path:
            warnings.warn(
                "The size is set to base, but the word large appears in the pretrained path!"
            )
        elif size == "large" and "base" in pretrained_path:
            warnings.warn(
                "The size is set to large, but the word base appears in the pretrained path!"
            )

        if size == "base":
            self.encoder_dim = 768
            self.encoder_depth = 12
            self.num_heads = 16
            self.patch_size = 8
        else:
            # large by default
            self.encoder_dim = 1024
            self.encoder_depth = 24
            self.num_heads = 16
            self.patch_size = 8

        self.modality = modality
        self.num_patches = int((image_resolution / 8) ** 2)
        self.s1_channels = 2  # fixed at 2 SAR backscatter channels
        self.s2_channels = 12  # fixed at 12 multispectral optical channels
        self.attn_bias = get_2dalibi(
            num_heads=self.num_heads, num_patches=self.num_patches
        )

        if modality in ["SAR", "both"]:
            print("Initializing SAR encoder")
            self.s1_encoder = ViT(
                dim=self.encoder_dim,
                depth=int(self.encoder_depth / 2),
                in_channels=self.s1_channels,
            )
            self.GAP_FFN_s1 = nn.Sequential(
                nn.LayerNorm(self.encoder_dim),
                nn.Linear(
                    self.encoder_dim, int(4 * self.encoder_dim)
                ),  # (BSZ, num_patches, inner_dim)
                nn.GELU(),  # (BSZ, num_patches, inner_dim)
                nn.Linear(
                    int(4 * self.encoder_dim), self.encoder_dim
                ),  # (BSZ, num_patches, dim)
            )

            # load weights
            self.s1_encoder.load_state_dict(torch.load(pretrained_path)["s1_encoder"])
            self.GAP_FFN_s1.load_state_dict(torch.load(pretrained_path)["s1_GAP_FFN"])

        if modality in ["optical", "both"]:
            print("Initializing optical encoder")
            self.s2_encoder = ViT(
                dim=self.encoder_dim,
                depth=self.encoder_depth,
                in_channels=self.s2_channels,
            )
            self.GAP_FFN_s2 = nn.Sequential(
                nn.LayerNorm(self.encoder_dim),
                nn.Linear(
                    self.encoder_dim, int(4 * self.encoder_dim)
                ),  # (BSZ, num_patches, inner_dim)
                nn.GELU(),  # (BSZ, num_patches, inner_dim)
                nn.Linear(
                    int(4 * self.encoder_dim), self.encoder_dim
                ),  # (BSZ, num_patches, dim)
            )

            # load weights
            self.s2_encoder.load_state_dict(torch.load(pretrained_path)["s2_encoder"])
            self.GAP_FFN_s2.load_state_dict(torch.load(pretrained_path)["s2_GAP_FFN"])

        if modality == "both":
            print("Initializing joint SAR-optical encoder")
            self.cross_encoder = BaseTransformerCrossAttn(
                dim=self.encoder_dim,
                depth=int(self.encoder_depth / 2),
                num_heads=self.num_heads,
            )

            # load weights
            self.cross_encoder.load_state_dict(
                torch.load(pretrained_path)["joint_encoder"]
            )

    def forward(self, SAR_images=None, optical_images=None):
        return_dict = {}
        if self.modality in ["SAR", "both"]:
            assert SAR_images is not None, (
                f"Modality is set to {self.modality}, but SAR_images are None"
            )
            SAR_encodings = self.s1_encoder(
                imgs=SAR_images, attn_bias=self.attn_bias.to(SAR_images.device)
            )  # (bsz, num_patches, encoder_dim)
            SAR_GAP = self.GAP_FFN_s1(SAR_encodings.mean(dim=1))  # (bsz, encoder_dim)
            return_dict["SAR_encodings"] = SAR_encodings
            return_dict["SAR_GAP"] = SAR_GAP

        if self.modality in ["optical", "both"]:
            assert optical_images is not None, (
                f"Modality is set to {self.modality}, but optical_images are None"
            )
            optical_encodings = self.s2_encoder(
                imgs=optical_images, attn_bias=self.attn_bias.to(optical_images.device)
            )  # (bsz, num_patches, encoder_dim)
            optical_GAP = self.GAP_FFN_s2(
                optical_encodings.mean(dim=1)
            )  # (bsz, encoder_dim)
            return_dict["optical_encodings"] = optical_encodings
            return_dict["optical_GAP"] = optical_GAP

        if self.modality == "both":
            joint_encodings = self.cross_encoder(
                x=SAR_encodings,
                context=optical_encodings,
                relative_position_bias=self.attn_bias.to(optical_images.device),
            )  # (bsz, num_patches, encoder_dim)
            joint_GAP = joint_encodings.mean(dim=1)  # (bsz, encoder_dim)
            return_dict["joint_encodings"] = joint_encodings
            return_dict["joint_GAP"] = joint_GAP

        return return_dict


@typing.no_type_check
def get_2dalibi(num_heads, num_patches):
    # inspired by: https://github.com/ofirpress/attention_with_linear_biases
    points = list(
        itertools.product(
            range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))
        )
    )

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    slopes = torch.Tensor(get_slopes(num_heads)).unsqueeze(1)
    idxs = []
    for p1 in points:
        for p2 in points:
            dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            idxs.append(dist * slopes * -1)
    all_bias = torch.cat(idxs, dim=1)
    return all_bias.view(1, num_heads, num_patches, num_patches)


@typing.no_type_check
class FFN(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.0,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, dim),  # (BSZ, num_patches, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        return self.net(x)  # (BSZ, num_patches, dim)


@typing.no_type_check
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be evenly divisible by num_heads"
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = (
            einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        )  # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = (
            attention_scores + relative_position_bias
        )  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(
            dim=-1
        )  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


@typing.no_type_check
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, "dim must be evenly divisible by num_heads"
        dim_head = int(dim / num_heads)
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        context = self.input_norm(context)  # (BSZ, num_patches, dim)

        q = self.to_q(x)  # (BSZ, num_patches, dim)
        k = self.to_k(context)  # (BSZ, num_patches, dim)
        v = self.to_v(context)  # (BSZ, num_patches, dim)

        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v)
        )  # (BSZ, num_heads, num_patches, dim_head)

        attention_scores = (
            einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        )  # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = (
            attention_scores + relative_position_bias
        )  # (BSZ, num_heads, num_patches, num_patches)

        attn = attention_scores.softmax(
            dim=-1
        )  # (BSZ, num_heads, num_patches, num_patches)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        out = einsum(
            "b h i j, b h j d -> b h i d", attn, v
        )  # (BSZ, num_heads, num_patches, dim_head)
        out = rearrange(out, "b h n d -> b n (h d)")  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


@typing.no_type_check
class BaseTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        final_norm=True,
    ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, relative_position_bias=False):
        for self_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        if self.final_norm:
            return self.norm_out(x)
        else:
            return x


@typing.no_type_check
class BaseTransformerCrossAttn(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                        CrossAttention(
                            dim=dim, num_heads=num_heads, dropout=attn_dropout
                        ),
                        FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context, relative_position_bias):
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + x  # (BSZ, num_patches, dim)
            x = (
                cross_attn(x, context, relative_position_bias) + x
            )  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        x = self.norm_out(x)
        return x  # (BSZ, num_patches, dim)


@typing.no_type_check
class ViT(nn.Module):
    def __init__(self, dim, depth, in_channels):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.dim = dim
        self.num_heads = 16  # always 16, for base and large models
        self.patch_size = 8  # always 8, for base and large models

        pixels_per_patch = int(self.patch_size * self.patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(
            dim=self.dim,
            depth=self.depth,
            num_heads=self.num_heads,
        )

    def forward(self, imgs, attn_bias):
        x = rearrange(
            imgs,
            "b c (h i) (w j) -> b (h w) (c i j)",
            i=self.patch_size,
            j=self.patch_size,
        )
        # x is shape -> (bsz, num_patches, self.channels*self.patch_size*self.patch_size)

        x = self.linear_input(x)  # (bsz, num_patches, dim)
        x = self.transformer(x, relative_position_bias=attn_bias)
        return x

# fmt: on
