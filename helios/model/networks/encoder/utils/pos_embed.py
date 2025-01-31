# https://github.com/bair-climate-initiative/scale-mae/blob/main/mae/util/pos_embed.py
import numpy as np
import torch


def get_2d_sincos_pos_embed_with_resolution(
    embed_dim, grid_size, res, cls_token=False, device="cpu"
):
    """
    grid_size: int of the grid height and width
    res: array of size n, representing the resolution of a pixel (say, in meters),
    return:
    pos_embed: [n,grid_size*grid_size, embed_dim] or [n,1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    res = res.to(device)
    grid_h = torch.arange(grid_size, device=device)
    grid_w = torch.arange(grid_size, device=device)
    grid = torch.meshgrid(
        grid_w, grid_h, indexing="xy"
    )  # here h goes first,direction reversed for numpy
    grid = torch.stack(grid, dim=0)  # 2 x h x w

    # grid = grid.reshape([2, 1, grid_size, grid_size])
    grid = torch.einsum("chw,n->cnhw", grid, res)  # 2 x n x h x w
    _, n, h, w = grid.shape
    pos_embed = get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid)  #  # (nxH*W, D/2)
    pos_embed = pos_embed.reshape(n, h * w, embed_dim)
    if cls_token:
        pos_embed = torch.cat(
            [
                torch.zeros([n, 1, embed_dim], device=pos_embed.device),
                pos_embed,
            ],
            dim=1,
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid_torch(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_torch(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, device=pos.device) / embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_month_encoding_table(embed_dim):
    """Sinusoid month encoding table, for 12 months indexed from 0-11"""
    assert embed_dim % 2 == 0
    angles = torch.arange(0, 13) / (12 / (2 * np.pi))

    sin_table = torch.sin(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    cos_table = torch.cos(torch.stack([angles for _ in range(embed_dim // 2)], axis=-1))
    month_table = torch.concatenate([sin_table[:-1], cos_table[:-1]], axis=-1)

    return month_table  # (M, D)