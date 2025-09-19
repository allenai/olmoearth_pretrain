# type: ignore
# ruff: noqa

import math
from collections import OrderedDict
from copy import deepcopy
from typing import cast

import numpy as np
import torch
from einops import repeat
from torch import nn
from torch.jit import Final
from torch.nn import functional as F

# naming convention matches helios.data.constants
PRESTO_S2_BANDS = [
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
PRESTO_S1_BANDS = ["vv", "vh"]

PRESTO_BANDS = (
    PRESTO_S1_BANDS
    + PRESTO_S2_BANDS
    + ["temperature_2m", "total_precipitation", "elevation", "slope", "NDVI"]
)

# used in normalization
PRESTO_ADD_BY = [
    25.0,
    25.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -272.15,
    0.0,
    0.0,
    0.0,
    0.0,
]
PRESTO_DIV_BY = [
    25.0,
    25.0,
    1e4,
    1e4,
    1e4,
    1e4,
    1e4,
    1e4,
    1e4,
    1e4,
    1e4,
    1e4,
    1e4,
    35.0,
    0.03,
    2000.0,
    50.0,
    1.0,
]


BANDS_GROUPS_IDX = OrderedDict(
    [
        ("S1", [0, 1]),
        ("S2_RGB", [2, 3, 4]),
        ("S2_Red_Edge", [5, 6, 7]),
        ("S2_NIR_10m", [8]),
        ("S2_NIR_20m", [9]),
        ("S2_SWIR", [10, 11]),
        ("ERA5", [12, 13]),
        ("SRTM", [14, 15]),
        ("NDVI", [16]),
    ]
)

NUM_DYNAMIC_WORLD_CLASSES = 9


class Attention(nn.Module):
    # https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    fast_attn: Final[bool]

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fast_attn = hasattr(
            torch.nn.functional, "scaled_dot_product_attention"
        )  # FIXME

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fast_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_norm=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)
    """
    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)


def get_month_encoding_table(d_hid):
    """Sinusoid month encoding table, for 12 months indexed from 0-11"""
    assert d_hid % 2 == 0
    angles = np.arange(0, 13) / (12 / (2 * np.pi))

    sin_table = np.sin(np.stack([angles for _ in range(d_hid // 2)], axis=-1))
    cos_table = np.cos(np.stack([angles for _ in range(d_hid // 2)], axis=-1))
    month_table = np.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    if torch.cuda.is_available():
        return torch.FloatTensor(month_table).cuda()
    else:
        return torch.FloatTensor(month_table)


def month_to_tensor(
    month: torch.Tensor | int, batch_size: int, seq_len: int, device: torch.device
):
    if isinstance(month, int):
        assert cast(int, month) < 12
    else:
        assert max(cast(torch.Tensor, month.flatten())) < 12

    if isinstance(month, int):
        # >>> torch.fmod(torch.tensor([9., 10, 11, 12, 13, 14]), 12)
        # tensor([ 9., 10., 11.,  0.,  1.,  2.])
        month = (
            torch.fmod(torch.arange(month, month + seq_len, dtype=torch.long), 12)
            .expand(batch_size, seq_len)
            .to(device)
        )
    elif len(month.shape) == 1:
        month = torch.stack(
            [
                torch.fmod(torch.arange(m, m + seq_len, dtype=torch.long), 12)
                for m in month
            ]
        ).to(device)
    return month


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_size: int = 128,
        channel_embed_ratio: float = 0.25,
        month_embed_ratio: float = 0.25,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
    ):
        super().__init__()

        self.band_groups = BANDS_GROUPS_IDX
        self.embedding_size = embedding_size

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx
            for idx, (group_name, _) in enumerate(self.band_groups.items())
        }
        self.band_group_to_idx["dynamic_world"] = (
            max(self.band_group_to_idx.values()) + 1
        )

        self.eo_patch_embed = nn.ModuleDict(
            {
                group_name: nn.Linear(len(group), embedding_size)
                for group_name, group in self.band_groups.items()
            }
        )
        self.dw_embed = nn.Embedding(
            num_embeddings=NUM_DYNAMIC_WORLD_CLASSES + 1, embedding_dim=embedding_size
        )
        self.latlon_embed = nn.Linear(3, embedding_size)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embedding_size,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embedding_size)

        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        pos_embedding_size = int(
            embedding_size * (1 - (channel_embed_ratio + month_embed_ratio))
        )
        channel_embedding_size = int(embedding_size * channel_embed_ratio)
        month_embedding_size = int(embedding_size * month_embed_ratio)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, pos_embedding_size), requires_grad=False
        )
        month_tab = get_month_encoding_table(month_embedding_size)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        self.channel_embed = nn.Embedding(
            num_embeddings=len(self.band_groups) + 1,
            embedding_dim=channel_embedding_size,
        )

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_sinusoid_encoding_table(
            self.pos_embed.shape[1], self.pos_embed.shape[-1]
        )
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def cartesian(latlons: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # an embedding is calculated for all timesteps. This is then expanded
            # for each timestep in the sequence
            latlon_radians = latlons * math.pi / 180
            lats, lons = latlon_radians[:, 0], latlon_radians[:, 1]
            x = torch.cos(lats) * torch.cos(lons)
            y = torch.cos(lats) * torch.sin(lons)
            z = torch.sin(lats)
        return torch.stack([x, y, z], dim=-1)

    @staticmethod
    def mask_tokens(x, mask):
        summed = mask.sum(
            dim=(1, 2)
        )  # summed tells me the number of masked elements per batch idx
        assert summed.max() == summed.min(), f"{summed.max()}, {summed.min()}"

        batch_size = x.shape[0]
        removed_elements_per_batch = int(summed.max() / mask.shape[2])
        kept_elements_per_batch = x.shape[1] - removed_elements_per_batch
        embedding_dim = x.shape[-1]

        # we want the mask to just be the indices of the masked tokens
        indices = repeat(
            torch.arange(0, x.shape[1]).long().to(x.device), "d -> b d", b=x.shape[0]
        )

        x = x[~mask.bool()].view(batch_size, kept_elements_per_batch, embedding_dim)

        mask = mask[:, :, 0]
        kept_indices = indices[~mask.bool()].view(batch_size, kept_elements_per_batch)
        removed_indices = indices[mask.bool()].view(
            batch_size, removed_elements_per_batch
        )

        return x, kept_indices, removed_indices

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        # different from the original
        # presto - latlons can be optionally ignored
        latlons: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        month: torch.Tensor | int = 0,
        eval_task: bool = True,
    ):
        device = x.device

        if mask is None:
            mask = torch.zeros_like(x, device=x.device).float()

        months = month_to_tensor(month, x.shape[0], x.shape[1], device)
        month_embedding = self.month_embed(months)
        positional_embedding = repeat(
            self.pos_embed[:, : x.shape[1], :],
            "b t d -> (repeat b) t d",
            repeat=x.shape[0],
        )

        # we assume the number of masked patches is the same
        # for all items in the batch. Otherwise things become a headache
        all_tokens, all_masks = [], []

        for channel_group, channel_idxs in self.band_groups.items():
            tokens = self.eo_patch_embed[channel_group](x[:, :, channel_idxs])
            channel_embedding = self.channel_embed(
                torch.tensor(self.band_group_to_idx[channel_group]).long().to(device)
            )
            channel_embedding = repeat(
                channel_embedding, "d -> b t d", b=x.shape[0], t=x.shape[1]
            )
            if channel_group == "SRTM":
                # for SRTM, we reduce it to a single token instead of
                # a token per timestep
                channel_wise_positional_embedding = torch.cat(
                    (
                        torch.zeros_like(month_embedding[:, 0:1]),
                        channel_embedding[:, 0:1],
                        torch.zeros_like(positional_embedding[:, 0:1]),
                    ),
                    dim=-1,
                )
                indices = slice(0, 1)
            else:
                channel_wise_positional_embedding = torch.cat(
                    (month_embedding, channel_embedding, positional_embedding), dim=-1
                )
                indices = slice(None)

            tokens = tokens[:, indices]
            tokens += channel_wise_positional_embedding
            all_tokens.append(tokens)
            group_mask = repeat(
                torch.max(mask[:, indices, channel_idxs], dim=-1)[0],
                "b t -> b t d",
                d=tokens.shape[-1],
            )
            all_masks.append(group_mask)

        # then, dynamic world
        tokens = self.dw_embed(dynamic_world)
        channel_embedding = self.channel_embed(
            torch.tensor(self.band_group_to_idx["dynamic_world"]).long().to(device)
        )
        channel_embedding = repeat(
            channel_embedding, "d -> b t d", b=x.shape[0], t=x.shape[1]
        )
        positional_embedding = torch.cat(
            (month_embedding, channel_embedding, positional_embedding), dim=-1
        )
        tokens += positional_embedding
        all_tokens.append(tokens)

        # now we calculate the mask for these [b, t] tokens
        group_mask = repeat(
            dynamic_world == NUM_DYNAMIC_WORLD_CLASSES,
            "b t -> b t d",
            d=tokens.shape[-1],
        )
        all_masks.append(group_mask)

        x = torch.cat(all_tokens, dim=1)  # [batch, timesteps, embedding_dim]
        mask = torch.cat(all_masks, dim=1)  # [batch, timesteps, embedding_dim]
        x, kept_indices, removed_indices = self.mask_tokens(x, mask)

        # append latlon tokens
        if latlons is not None:
            latlon_tokens = self.latlon_embed(self.cartesian(latlons)).unsqueeze(1)
            x = torch.cat((latlon_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # mask will be a boolean of shape [batch, total_num_tokens]
        if eval_task:
            return self.norm(x.mean(dim=1))
        return self.norm(x), kept_indices, removed_indices


class Decoder(nn.Module):
    def __init__(
        self,
        channel_embeddings: nn.Embedding,
        encoder_embed_dim=128,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=8,
        mlp_ratio=2,
        max_sequence_length=24,
    ):
        super().__init__()

        self.band_groups = BANDS_GROUPS_IDX

        # this is used for the channel embedding
        self.band_group_to_idx = {
            group_name: idx
            for idx, (group_name, _) in enumerate(self.band_groups.items())
        }
        self.band_group_to_idx["dynamic_world"] = (
            max(self.band_group_to_idx.values()) + 1
        )

        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.eo_decoder_pred = nn.ModuleDict(
            {
                group_name: nn.Linear(decoder_embed_dim, len(group))
                for group_name, group in self.band_groups.items()
            }
        )
        self.dw_decoder_pred = nn.Linear(decoder_embed_dim, NUM_DYNAMIC_WORLD_CLASSES)

        self.channel_embeddings = channel_embeddings
        channel_embedding_dims = channel_embeddings.weight.shape[-1]
        remaining_embeddings = decoder_embed_dim - channel_embedding_dims
        # the positional + monthly + channel embedding
        self.max_sequence_length = max_sequence_length
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_sequence_length, int(remaining_embeddings) // 2),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(int(remaining_embeddings) // 2)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_sinusoid_encoding_table(
            self.pos_embed.shape[1], self.pos_embed.shape[-1]
        )
        self.pos_embed.data.copy_(pos_embed)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def add_masked_tokens(self, x, kept_indices, removed_indices):
        mask_tokens = repeat(
            self.mask_token, "d -> b t d", b=x.shape[0], t=removed_indices.shape[1]
        )

        x = torch.cat([x, mask_tokens], dim=1)

        # sort according to their indices. Shape is [batch, index]
        combined_indices = torch.cat([kept_indices, removed_indices], dim=1) + 1
        # 0 for latlon index
        combined_indices = torch.sort(
            torch.cat(
                [torch.zeros_like(combined_indices[:, 0:1]), combined_indices], dim=1
            )
        )[1]
        # and then tile for each dimension
        combined_indices = repeat(combined_indices, "b t -> b t d", d=x.shape[-1])
        x = torch.gather(x, 1, combined_indices)
        return x

    def add_embeddings(self, x, month: torch.Tensor | int):
        num_channel_groups = len(self.band_group_to_idx)
        # -2 since we remove srtm and latlon, and -1 since the srtm
        # channel group doesn't have timesteps
        num_timesteps = int((x.shape[1] - 2) / (num_channel_groups - 1))
        srtm_index = self.band_group_to_idx["SRTM"] * num_timesteps
        months = month_to_tensor(month, x.shape[0], num_timesteps, x.device)

        # when we expand the encodings, each channel_group gets num_timesteps
        # encodings. However, there is only one SRTM token so we remove the
        # excess SRTM encodings
        remove_mask = torch.full(
            size=(num_timesteps * num_channel_groups,), fill_value=False
        )
        remove_mask[torch.arange(num_timesteps - 1) + srtm_index] = True

        month_embedding = repeat(
            self.month_embed(months),
            "b t d -> b (repeat t) d",
            repeat=num_channel_groups,
        )
        month_embedding = month_embedding[:, ~remove_mask]
        month_embedding[:, srtm_index] = 0

        positional_embedding = repeat(
            self.pos_embed[:, :num_timesteps, :],
            "b t d -> (b2 b) (t2 t) d",
            b2=x.shape[0],
            t2=num_channel_groups,
        )
        positional_embedding = positional_embedding[:, ~remove_mask]
        positional_embedding[:, srtm_index] = 0

        channel_embeddings = torch.repeat_interleave(
            self.channel_embeddings.weight, repeats=num_timesteps, dim=0
        )
        channel_embeddings = repeat(channel_embeddings, "c d -> b c d", b=x.shape[0])
        channel_embeddings = channel_embeddings[:, ~remove_mask]

        positional_embedding = torch.cat(
            (month_embedding, channel_embeddings, positional_embedding), dim=-1
        )

        # add the zero embedding for the latlon token
        positional_embedding = torch.cat(
            [torch.zeros_like(positional_embedding[:, 0:1, :]), positional_embedding],
            dim=1,
        )

        x += positional_embedding
        return x

    def reconstruct_inputs(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # remove the latlon token
        x = x[:, 1:, :]

        # split into channel groups
        num_channel_groups = len(self.band_group_to_idx) - 1
        num_timesteps = int((x.shape[1] - 1) / num_channel_groups)
        srtm_index = self.band_group_to_idx["SRTM"] * num_timesteps
        srtm_token = x[:, srtm_index : srtm_index + 1, :]

        mask = torch.full((x.shape[1],), True, device=x.device)
        mask[torch.tensor(srtm_index)] = False
        x = x[:, mask]

        x = x.view(x.shape[0], num_channel_groups, num_timesteps, x.shape[-1])

        eo_output, dw_output = [], None
        for group_name, idx in self.band_group_to_idx.items():
            if group_name == "SRTM":
                eo_output.append(
                    repeat(
                        self.eo_decoder_pred[group_name](srtm_token),
                        "b t d -> b (t2 t) d",
                        t2=num_timesteps,
                    )
                )
            else:
                if idx > self.band_group_to_idx["SRTM"]:
                    idx -= 1
                group_tokens = x[:, idx]
                if group_name == "dynamic_world":
                    dw_output = self.dw_decoder_pred(group_tokens)
                else:
                    eo_output.append(self.eo_decoder_pred[group_name](group_tokens))

        # we can just do this concatenation because the BANDS_GROUP_IDX
        # is ordered
        return torch.cat(eo_output, dim=-1), cast(torch.Tensor, dw_output)

    def forward(self, x, kept_indices, removed_indices, month):
        x = self.decoder_embed(x)
        x = self.add_masked_tokens(x, kept_indices, removed_indices)
        x = self.add_embeddings(x, month)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.reconstruct_inputs(x)


class PrestoFineTuningModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder: Encoder = deepcopy(encoder)
        # make sure the model is trainable, since we can call
        # this having called requires_grad_(False)
        self.encoder.requires_grad_(True)
        # but don't unfreeze the position encoder, which
        # shouldn't be trainable
        self.encoder.pos_embed.requires_grad_(False)
        self.encoder.month_embed.requires_grad_(False)
        self.head = head

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: torch.Tensor | None = None,
        month: torch.Tensor | int = 0,
    ) -> torch.Tensor:
        return self.head(
            self.encoder(
                x=x,
                dynamic_world=dynamic_world,
                latlons=latlons,
                mask=mask,
                month=month,
                eval_task=True,
            )
        )


class FinetuningHead(nn.Module):
    def __init__(self, hidden_size: int, num_outputs: int, regression: bool) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.regression = regression
        self.linear = nn.Linear(hidden_size, num_outputs)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        if (not self.regression) & (self.num_outputs == 1):
            x = torch.sigmoid(x)
        return x


class Presto(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder

    def forward(
        self,
        x: torch.Tensor,
        dynamic_world: torch.Tensor,
        latlons: torch.Tensor,
        mask: torch.Tensor | None = None,
        month: torch.Tensor | int = 0,
    ) -> torch.Tensor:
        x, kept_indices, removed_indices = self.encoder(
            x=x,
            dynamic_world=dynamic_world,
            latlons=latlons,
            mask=mask,
            month=month,
            eval_task=False,
        )

        return self.decoder(x, kept_indices, removed_indices, month)

    @classmethod
    def construct(
        cls,
        encoder_embedding_size: int = 128,
        channel_embed_ratio: float = 0.25,
        month_embed_ratio: float = 0.25,
        encoder_depth=2,
        mlp_ratio=4,
        encoder_num_heads=8,
        decoder_embedding_size=128,
        decoder_depth=2,
        decoder_num_heads=8,
        max_sequence_length=24,
    ):
        encoder = Encoder(
            embedding_size=encoder_embedding_size,
            channel_embed_ratio=channel_embed_ratio,
            month_embed_ratio=month_embed_ratio,
            depth=encoder_depth,
            mlp_ratio=mlp_ratio,
            num_heads=encoder_num_heads,
            max_sequence_length=max_sequence_length,
        )
        decoder = Decoder(
            channel_embeddings=encoder.channel_embed,
            encoder_embed_dim=encoder_embedding_size,
            decoder_embed_dim=decoder_embedding_size,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            max_sequence_length=max_sequence_length,
        )
        return cls(encoder, decoder)

    def construct_finetuning_model(
        self,
        num_outputs: int,
        regression: bool = False,
    ):
        head = FinetuningHead(
            num_outputs=num_outputs,
            hidden_size=self.encoder.embedding_size,
            regression=regression,
        )
        model = PrestoFineTuningModel(self.encoder, head).to(
            self.encoder.pos_embed.device
        )
        model.train()
        return model
