# type: ignore
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .aurora.fourier import FourierExpansion
from .dynamic_hypernetwork import Dynamic_MLP_OFA_spectral, Dynamic_MLP_OFA_variable
from .flexivit.utils import resize_abs_pos_embed


class CopernicusFMViT(nn.Module):
    """CopernicusFM: VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        drop_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=0,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        loc_option="lonlat",
        return_intermediate=False,
        intermediate_indices=None,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = norm_layer
            embed_dim = embed_dim
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        self.patch_embed_spectral = Dynamic_MLP_OFA_spectral(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )
        self.patch_embed_variable = Dynamic_MLP_OFA_variable(
            wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim
        )

        self.num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # ---------------------------------------------------------------------------

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.loc_option = loc_option
        if loc_option == "cartesian":
            self.coord_expansion = FourierExpansion(1e-7, 2)
        elif loc_option == "lonlat":
            self.coord_expansion = FourierExpansion(0.0001, 720)

        self.scale_expansion = FourierExpansion(0.001, 5.1e8)  # 1m2 to 5.1e8 km2
        self.time_expansion = FourierExpansion(
            1, 365.25, assert_range=False
        )  # 1 to 365.25 days, enable more than 1 year
        self.coord_fc = nn.Linear(embed_dim, embed_dim)
        self.scale_fc = nn.Linear(embed_dim, embed_dim)
        self.time_fc = nn.Linear(embed_dim, embed_dim)
        # if meta info is not available, set to a learned parameter
        self.coord_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scale_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        self.return_intermediate = return_intermediate
        self.intermediate_indices = intermediate_indices

    def get_coord_pos_embed(self, lons, lats, embed_dim):
        if self.loc_option == "cartesian":
            # convert to spherical coordinates
            spherical_x = (
                torch.cos(lons * math.pi / 180) * torch.cos(lats * math.pi / 180)
                + 1
                + 1e-7
            )
            spherical_y = (
                torch.sin(lons * math.pi / 180) * torch.cos(lats * math.pi / 180)
                + 1
                + 1e-7
            )
            spherical_z = torch.sin(lats * math.pi / 180) + 1 + 1e-7
            coord_embed_spherical_x = self.coord_expansion(spherical_x, embed_dim // 3)
            coord_embed_spherical_y = self.coord_expansion(spherical_y, embed_dim // 3)
            coord_embed_spherical_z = self.coord_expansion(spherical_z, embed_dim // 3)
            coord_embed = torch.cat(
                [
                    coord_embed_spherical_x,
                    coord_embed_spherical_y,
                    coord_embed_spherical_z,
                ],
                dim=-1,
            )  # [B,D]
        elif self.loc_option == "lonlat":
            coord_embed_lon = self.coord_expansion(lons + 180, embed_dim // 2)
            coord_embed_lat = self.coord_expansion(lats + 90, embed_dim // 2)
            coord_embed = torch.cat([coord_embed_lon, coord_embed_lat], dim=-1)

        if coord_embed.shape[-1] < embed_dim:
            # pad zeros
            coord_embed = torch.cat(
                (
                    coord_embed,
                    torch.zeros(
                        coord_embed.shape[0],
                        embed_dim - coord_embed.shape[-1],
                        device=coord_embed.device,
                    ),
                ),
                dim=-1,
            )

        return coord_embed.unsqueeze(1)  # [B,1,D]

    def get_area_pos_embed(self, areas, embed_dim):
        scale_embed = self.scale_expansion(areas, embed_dim)  # B, D
        return scale_embed.unsqueeze(1)  # [B,1,D]

    def get_time_pos_embed(self, times, embed_dim):
        time_embed = self.time_expansion(times, embed_dim)  # B, D
        return time_embed.unsqueeze(1)  # [B,1,D]

    def forward_features(
        self,
        x,
        meta_info,
        wave_list,
        bandwidth,
        language_embed,
        input_mode,
        kernel_size=None,
    ):
        # embed patches
        if input_mode == "spectral":
            wavelist = torch.tensor(wave_list, device=x.device).float()
            bandwidths = torch.tensor(bandwidth, device=x.device).float()
            self.waves = wavelist
            x, _ = self.patch_embed_spectral(x, self.waves, bandwidths, kernel_size)
        elif input_mode == "variable":
            x, _ = self.patch_embed_variable(x, language_embed, kernel_size)

        # resize pos embed
        num_patches = x.size(1)
        num_patches_sqrt = int(math.sqrt(num_patches))
        num_patches_sqrt_origin = int(math.sqrt(self.num_patches))
        pos_embed = resize_abs_pos_embed(
            self.pos_embed,
            num_patches_sqrt,
            (num_patches_sqrt_origin, num_patches_sqrt_origin),
            num_prefix_tokens=1,
        )

        # coord, scale and time pos embed
        lons, lats, times, areas = (
            meta_info[:, 0],
            meta_info[:, 1],
            meta_info[:, 2],
            meta_info[:, 3],
        )
        embed_dim = pos_embed.shape[-1]
        if torch.isnan(lons).any() or torch.isnan(lats).any():
            coord_embed = self.coord_token
        else:
            coord_embed = self.get_coord_pos_embed(lons, lats, embed_dim)
        coord_embed = self.coord_fc(coord_embed)
        if torch.isnan(areas).any():
            area_embed = self.scale_token
        else:
            area_embed = self.get_area_pos_embed(areas, embed_dim)
        area_embed = self.scale_fc(area_embed)
        if torch.isnan(times).any():
            time_embed = self.time_token
        else:
            time_embed = self.get_time_pos_embed(times, embed_dim)
        time_embed = self.time_fc(time_embed)
        pos_embed = pos_embed + coord_embed + area_embed + time_embed

        # add pos embed w/o cls token
        x = x + pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        intermediate_features = []
        hw = num_patches_sqrt
        hw_shape = (hw, hw)

        # apply Transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.return_intermediate and (i in self.intermediate_indices):
                out = x[:, 1:]
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                intermediate_features.append(out)

        # if self.global_pool:
        #     x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
        #     outcome = self.fc_norm(x)
        # else:
        #     x = self.norm(x)
        #     outcome = x[:, 0]

        # if self.return_intermediate:
        #     return outcome, intermediate_features

        # for segmentation tasks, ignore the norm
        # https://github.com/zhu-xlab/Copernicus-FM/blob/main/Copernicus-Bench/src/foundation_models/CopernicusFM/models_dwv_seg.py
        # for classification, we will apply the fc_norm in the wrapper
        return x[:, 1:, :]

    def forward_head(self, x, pre_logits=False):
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(
        self,
        x,
        meta_info,
        wave_list,
        bandwidth,
        language_embed,
        input_mode,
        kernel_size=None,
    ):
        if self.return_intermediate:
            x, intermediate_features = self.forward_features(
                x,
                meta_info,
                wave_list,
                bandwidth,
                language_embed,
                input_mode,
                kernel_size,
            )
            return x, intermediate_features
        else:
            fx = self.forward_features(
                x,
                meta_info,
                wave_list,
                bandwidth,
                language_embed,
                input_mode,
                kernel_size,
            )
            x = self.forward_head(fx)
            return x, fx


def vit_small_patch16(**kwargs):
    model = CopernicusFMViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = CopernicusFMViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch16(**kwargs):
    model = CopernicusFMViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = CopernicusFMViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
