# type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# CopernicusFM: meta encoding (follow aurora)
from .aurora.fourier import FourierExpansion

# CopernicusFM: dynamic patch size (follow flexivit)
from .flexivit.patch_embed import pi_resize_patch_embed
from .util.pos_embed import get_1d_sincos_pos_embed_from_grid_torch

# from torchvision.datasets.utils import download_url


random_seed = 1234
torch.manual_seed(random_seed)


class TransformerWeightGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=4, num_layers=1):
        super(TransformerWeightGenerator, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            activation="gelu",
            norm_first=False,
            batch_first=False,
            dropout=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Linear layer to map transformer output to desired weight shape
        self.fc_weight = nn.Linear(input_dim, output_dim)
        self.fc_bias = nn.Linear(input_dim, embed_dim)
        self.wt_num = 128
        self.weight_tokens = nn.Parameter(torch.empty([self.wt_num, input_dim]))
        self.bias_token = nn.Parameter(torch.empty([1, input_dim]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.weight_tokens, std=0.02)
        torch.nn.init.normal_(self.bias_token, std=0.02)

    def forward(self, x):
        # x should have shape [seq_len, batch, input_dim]
        pos_wave = x
        x = torch.cat([self.weight_tokens, pos_wave], dim=0)
        x = torch.cat([x, self.bias_token], dim=0)
        transformer_output = self.transformer_encoder(x)
        weights = self.fc_weight(transformer_output[self.wt_num : -1] + pos_wave)
        bias = self.fc_bias(
            transformer_output[-1]
        )  # Using the last output to generate bias
        return weights, bias


class GaussianFourierFeatureTransform(torch.nn.Module):
    """An implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_size*2, width, height].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        torch.manual_seed(42)
        self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        assert x.dim() == 4, f"Expected 4D input (got {x.dim()}D input)"

        batches, channels, width, height = x.shape

        assert channels == self._num_input_channels, (
            f"Expected input to have {self._num_input_channels} channels (got {channels} channels)"
        )

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self._mapping_size)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class Basic1d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        conv = nn.Linear(in_channels, out_channels, bias)
        self.conv = nn.Sequential(
            conv,
        )
        if not bias:
            self.conv.add_module("ln", nn.LayerNorm(out_channels))
        self.conv.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class FCResLayer(nn.Module):
    def __init__(self, linear_size=128):
        super(FCResLayer, self).__init__()
        self.l_size = linear_size
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        # self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.w2 = nn.Linear(self.l_size, self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        # y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        out = x + y
        return out


class Dynamic_MLP_Decoder(nn.Module):
    def __init__(self, wv_planes, inter_dim=128, kernel_size=16, decoder_embed=512):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.inter_dim = inter_dim
        self.decoder_embed = decoder_embed
        self._num_kernel = self.kernel_size * self.kernel_size * self.decoder_embed

        # self.weight_generator = nn.Sequential(Basic1d(wv_planes, self.inter_dim, bias=True),
        #                                      nn.Linear(self.inter_dim, self._num_kernel))
        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, decoder_embed
        )
        self.scaler = 0.01

        self._init_weights()

    def _get_weights(self, waves, batch=True):
        dweights = []
        dynamic_weights = None
        if batch:
            dynamic_weights = self.weight_generator(waves)
        else:
            for i in range(waves.size(0)):
                dweights.append(self.weight_generator(waves[i]))
            dynamic_weights = torch.stack(dweights, dim=0)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """Initialize the base weights and dynamic mlp weights"""
        self.weight_generator.apply(self.weight_init)

    def forward(self, img_feat, waves, kernel_size=None):
        inplanes = waves.size(0)
        # wv_feats: 9,128 -> 9*16*16,512
        weight, bias = self._get_weights(waves)  # 9,16*16*512
        # dynamic_weight = weight.view(
        #     inplanes * self.kernel_size * self.kernel_size, self.decoder_embed
        # )  # 9*16*16,512

        # CopernicusFM: dynamic patch size
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.decoder_embed
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])
        # resize the weight to match different preferred kernel sizes
        if kernel_size != None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(
                dynamic_weight, (kernel_size, kernel_size)
            )  # 512, 9, p, p
        else:
            kernel_size = self.kernel_size
        dynamic_weight = (
            dynamic_weight.permute([1, 2, 3, 0])
            .contiguous()
            .view(-1, self.decoder_embed)
        )  # 9*p*p,512

        weights = dynamic_weight * self.scaler

        dynamic_out = F.linear(img_feat, weights, bias=None)
        x = dynamic_out
        return x


class Dynamic_Patch_Embed(nn.Module):
    """Input: channels of wavelength (normalized): List -> List
    kernel size of the depth-wise convolution: kernel_size, default 3x3
    wv_planes
    inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.patch_size = (kernel_size, kernel_size)
        self.weight2 = nn.Parameter(
            torch.empty([embed_dim, 2, kernel_size, kernel_size])
        )
        self.bias2 = nn.Parameter(torch.empty([embed_dim]))
        self.weight3 = nn.Parameter(
            torch.empty([embed_dim, 3, kernel_size, kernel_size])
        )
        self.bias3 = nn.Parameter(torch.empty([embed_dim]))
        self.weight4 = nn.Parameter(
            torch.empty([embed_dim, 4, kernel_size, kernel_size])
        )
        self.bias4 = nn.Parameter(torch.empty([embed_dim]))
        self.weight9 = nn.Parameter(
            torch.empty([embed_dim, 9, kernel_size, kernel_size])
        )
        self.bias9 = nn.Parameter(torch.empty([embed_dim]))
        self.weight70 = nn.Parameter(
            torch.empty([embed_dim, 70, kernel_size, kernel_size])
        )
        self.bias70 = nn.Parameter(torch.empty([embed_dim]))
        self.weights = {
            2: self.weight2,
            3: self.weight3,
            4: self.weight4,
            9: self.weight9,
            70: self.weight70,
        }
        self.biass = {
            2: self.bias2,
            3: self.bias3,
            4: self.bias4,
            9: self.bias9,
            70: self.bias70,
        }

    def forward(self, img_feat, waves):
        inplanes = waves.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        weights = self.weights[inplanes]
        bias = self.biass[inplanes]

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x


class Dynamic_MLP_OFA(nn.Module):
    """Input: channels of wavelength (normalized): List -> List
    kernel size of the depth-wise convolution: kernel_size, default 3x3
    wv_planes
    inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)
        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """Initialize the base weights and dynamic mlp weights"""
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs):
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000)
        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3
        # bias = None

        # dynamic_weight = weight.view(self.embed_dim, inplanes, self.kernel_size, self.kernel_size) #3xoutdx16x16
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])
        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=self.kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class Dynamic_MLP_OFA_spectral(nn.Module):
    """Input: channels of wavelength and bandwidth (normalized): List -> List
    kernel size of the depth-wise convolution: kernel_size, default 3x3
    wv_planes
    inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        ## CopernicusFM: fourier embedding for wavelength and bandwidth
        # min wavelength: ultraviolet light (100 nm)
        # max wavelength: radio waves (1 m)
        self.spectrum_central_expansion = FourierExpansion(100, 1e9)
        # min bandwidth: s2 ~ 10nm
        # max bandwidth: s1 ~ 1m
        self.spectrum_bandwidth_expansion = FourierExpansion(1, 1e9)

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """Initialize the base weights and dynamic mlp weights"""
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, wvs, bandwidths, kernel_size=None):
        """wvs: nm
        bandwidths: nm
        """
        inplanes = wvs.size(0)
        # wv_feats: 9,128 -> 9, 3x3x3
        # waves = get_1d_sincos_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000) # dofa: fixed sincos pos embedding
        # waves = get_1d_fourier_pos_embed_from_grid_torch(self.wv_planes, wvs * 1000) # new: fourier pos embedding
        emb_central = self.spectrum_central_expansion(wvs, self.wv_planes)
        emb_bandwidth = self.spectrum_bandwidth_expansion(bandwidths, self.wv_planes)
        waves = (
            emb_central + emb_bandwidth
        )  # simply add two embeddings, can be more complex later

        waves = self.fclayer(waves)
        weight, bias = self._get_weights(waves)  # 3x3x3

        # Fix bug
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )  # 9, 3, 3, 1024
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])  # 1024, 9, 3, 3
        # resize the weight to match different preferred kernel sizes
        if kernel_size != None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(
                dynamic_weight, (kernel_size, kernel_size)
            )
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves


class Dynamic_MLP_OFA_variable(nn.Module):
    """Input: language embedding of variable name: Pytorch tensor
    kernel size of the depth-wise convolution: kernel_size, default 3x3
    wv_planes
    inplanes
    """

    def __init__(self, wv_planes, inter_dim=128, kernel_size=3, embed_dim=1024):
        super().__init__()
        self.kernel_size = kernel_size
        self.wv_planes = wv_planes
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self._num_kernel = self.kernel_size * self.kernel_size * self.embed_dim
        self.inter_dim = inter_dim
        self.patch_size = (kernel_size, kernel_size)
        self.num_patches = -1

        self.language_proj = nn.Linear(
            2048, self.wv_planes
        )  # project to the same dimension as wv_planes

        self.weight_generator = TransformerWeightGenerator(
            wv_planes, self._num_kernel, embed_dim
        )
        self.scaler = 0.01

        self.fclayer = FCResLayer(wv_planes)

        self._init_weights()

    def _get_weights(self, waves):
        dynamic_weights = self.weight_generator(waves)

        return dynamic_weights

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def _init_weights(self):
        """Initialize the base weights and dynamic mlp weights"""
        self.weight_generator.apply(self.weight_init)
        self.fclayer.apply(self.weight_init)

    def forward(self, img_feat, language_embed, kernel_size=None):
        """wvs: nm
        bandwidths: nm
        """
        # wv_feats: 9,128 -> 9, 3x3x3
        emb_language = language_embed.unsqueeze(0)
        waves = self.language_proj(emb_language)
        # print(waves.size())

        waves = self.fclayer(waves)
        # print(waves.size())
        weight, bias = self._get_weights(waves)  # 3x3x3

        # inplanes = wvs.size(0)
        inplanes = waves.size(0)
        # print(inplanes)
        # Fix bug
        dynamic_weight = weight.view(
            inplanes, self.kernel_size, self.kernel_size, self.embed_dim
        )  # 9, 3, 3, 1024
        dynamic_weight = dynamic_weight.permute([3, 0, 1, 2])  # 1024, 9, 3, 3

        # resize the weight to match different preferred kernel sizes
        if kernel_size != None and self.kernel_size != kernel_size:
            dynamic_weight = pi_resize_patch_embed(
                dynamic_weight, (kernel_size, kernel_size)
            )
        else:
            kernel_size = self.kernel_size

        if bias is not None:
            bias = bias.view([self.embed_dim]) * self.scaler

        weights = dynamic_weight * self.scaler

        dynamic_out = F.conv2d(
            img_feat, weights, bias=bias, stride=kernel_size, padding=1, dilation=1
        )

        x = dynamic_out
        x = x.flatten(2).transpose(1, 2)

        return x, waves
