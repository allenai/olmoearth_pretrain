"""MOCK JEPA class

Derived for now from AnySat

"""

import torch
from torch import nn


class JEPAAny(nn.Module):
    """
    Masked Autoencoder part for OmniSat pretraining.
    """

    def __init__(
        self,
        encoder,
        predictor,
        modalities=["sentinel2", "naip"],
        embed_dim: int = 768,
        ratio: float = 0.0,
        num_patches: int = 0,
    ):
        super().__init__()
        self.modalities = modalities
        self.embed_dim = embed_dim
        self.ratio = ratio

        if self.ratio > 0:
            self.masked_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_patches = num_patches
        # --------------------------------------------------------------------------
        # JEPA encoder specifics
        self.encoder = encoder
        # --------------------------------------------------------------------------
        # JEPA predictor specifics
        self.predictor = predictor

    def mask_modalities(self, x):
        """
        :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
        :param mask: tensor of shape [B (batch-size), N (num-patches)]
        :param ratio: float between 0 and 1, fraction of patches to keep
        """
        N, L, D = x.shape
        # keep at least one patch per modality
        n_patches = L // len(self.modalities)
        keep = torch.randint(
            len(self.modalities),
            size=(
                N,
                n_patches,
            ),
            device=x.device,
        )
        keep = keep * n_patches + torch.arange(n_patches, device=x.device)

        len_keep = int(n_patches * (len(self.modalities) - 1) * (1 - self.ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        batch_indices = torch.arange(N).unsqueeze(1).expand(-1, n_patches)
        noise[batch_indices, keep] = 1.0
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        final_keep = torch.sum(
            torch.nn.functional.one_hot(
                torch.cat([keep, ids_keep], dim=1), num_classes=L
            ),
            dim=1,
        )
        final_keep = final_keep.unsqueeze(-1).repeat(1, 1, D)
        x_masked = x * final_keep + (1 - final_keep) * self.masked_token.repeat(N, L, 1)

        # x_masked = torch.cat([torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)),
        #                       torch.gather(x, dim=1, index=keep.unsqueeze(-1).repeat(1, 1, D))], dim=1)

        # # Get final mask
        # mask = torch.cat([mask + i * self.num_patches for i in range (len(self.modalities))], dim=1)
        # final_mask = torch.cat([torch.gather(mask, dim=1, index=ids_keep),
        #                         mask_keep], dim = 1)
        # return x_masked, final_mask
        return x_masked

    def forward_encoder(self, x, mask_enc):
        tokens, out = self.encoder.forward_proj(x)
        tokens, mask = apply_masks(tokens, mask_enc, self.modalities)
        if self.ratio > 0:
            tokens = self.mask_modalities(tokens)
        tokens = self.encoder.forward_transformer(tokens, mask)
        return out, tokens

    def forward_decoder(self, out, tokens, mask_enc, mask_pred):
        # embed tokens
        out["predicted_tokens"] = self.predictor(tokens[:, 1:, :], mask_enc, mask_pred)
        return out

    def forward(self, imgs, mask_enc, mask_pred):
        out, latent = self.forward_encoder(imgs, mask_enc)
        pred = self.forward_decoder(out, latent, mask_enc, mask_pred)
        return pred


def apply_masks(x, masks, modalities):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    n_patches = x.size(1) // (len(modalities) - int("modis" in modalities))
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        mask_keep = torch.cat(
            [
                mask_keep + i * n_patches
                for i in range(len(modalities) - int("modis" in modalities))
            ],
            dim=1,
        )
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0), m
