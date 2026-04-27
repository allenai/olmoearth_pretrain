from WIS_DSP_lib.constants import *
import torch
from torch import nn
from einops import rearrange, repeat

class PatchifyBandset(nn.Module):
    '''
    A module that takes in a batch of data for a specific bandset and patchifies it using a convolutional layer.
    The convolutional layer has a kernel size and stride equal to the patch size, which allows it to create non-overlapping patches of the input data. 
    The output of the convolutional layer is then rearranged to have the shape (B, H', W', T, D), where B is the batch size, H' and W' are the height 
    and width of the patchified data, T is the number of time steps, and D is the embedding size.
    '''
    def __init__(self, 
                patch_size: int,
                input_channels: int,
                embedding_size: int,
                bias: bool = True,
                ):
        '''
        Args:
            patch_size: the size of the patches to be created
            input_channels: the number of channels in the input data for this bandset
            embedding_size: the size of the output embedding for each patch
            bias: whether to include a bias term in the convolutional layer
        '''

        super().__init__()

        #PUT YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of the PatchifyBandset module.
        Args:
            x: a tensor of shape (B, H, W, T, C) containing the input data for a specific bandset
        Returns:            
        a tensor of shape (B, H', W', T, D) containing the patchified and embedded output, where H' and W' are the height and width of the patchified data
        '''
        
        #PUT YOUR CODE HERE

class PatchEmbeddings(nn.Module):
    '''
    A module that takes in a batch of data and patchifies it using a separate PatchifyBandset module for each bandset in the input data.
    The output of each PatchifyBandset module is then combined into a single tensor containing the patchified and embedded output for all 
    bandsets in the input data, along with the corresponding output mask.
    '''
    def __init__(self, 
                 patch_size: int,
                embedding_size: int,
                modality: str,
                ):
        '''
        Constructor for the PatchEmbeddings module.
        Args:
            patch_size: the size of the patches to be created
            embedding_size: the size of the output embedding for each patch
            modality: the name of the modality for which to create the patch embeddings
        '''

        super().__init__()
        
        self.patch_size = patch_size
        self.embedding_size = embedding_size
        self.modality = modality

        #PUT YOUR CODE HERE
        
    def forward(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        '''
        Forward pass of the PatchEmbeddings module.
        Args:
            x: a dictionary containing the input data and timestamps for a specific modality, with keys:
                - modality: a tensor of shape (B, H, W, T, C) containing the input data for the modality
                - modality_mask: a tensor of shape (B, H, W, T) containing the input mask for the modality
                - timestamps: a tensor of shape (B, T, 3) containing the timestamps for each patch, where the last dimension contains (day, month, year)
            modality: the name of the modality for which to create the patch embeddings
        Returns:
            a dictionary containing the patchified and embedded output for the modality, along with the corresponding output mask and timestamps, with keys:
                - modality: a tensor of shape (B, H', W', T, D) containing the patchified and embedded output for the modality, where H' and W' are the height and width of the patchified data
                - modality_mask: a tensor of shape (B, H', W', T) containing the output mask for the modality, where H' and W' are the height and width of the patchified data
                - timestamps: a tensor of shape (B, T, 3) containing the timestamps for each patch, where the last dimension contains (day, month, year)
        '''

        #PUT YOUR CODE HERE
    
class CompositeEncoding(nn.Module):
    '''
    A module that creates encodings for channel, time, month, and space.

    We have four types of encodings:
        1. Channel encoding: a learnable embedding for each channel in the input data
        2. Time encoding: a fixed sinusoidal encoding based on the timestamps of the input data
        3. Month encoding: a fixed embedding based on the month of the timestamps of the input data
        4. Space encoding: a fixed sinusoidal encoding based on the spatial location of the patches in the input data, scaled by the resolution of the input data.

        Since we have four types of encodings, we divide the embedding dimension by 4 and allocate an equal portion to each encoding type. 
        We then concatenate the four encodings together to get the final output embedding for each patch.

    '''
    def __init__(self, modality: str, embedding_size: int):
        '''
        Constructor for the CompositeEncoding module.
        Args:
            modality: the name of the modality for which to create the encodings
            embedding_size: the size of the output embedding for each patch
        '''
        super().__init__()
        
        #PUT YOUR CODE HERE

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Forward pass of the CompositeEncoding module. 

        Args:
            x: a dictionary containing the input data and timestamps for a specific modality, with keys:
                - modality: a tensor of shape (B, H', W', T, C) containing the patchified data for the modality
                - timestamps: a tensor of shape (B, T, 3) containing the timestamps for each patch, where the last dimension contains (day, month, year)
        Returns:
            a tensor of shape (B, H', W', T, D) containing the combined encoding for the modality, where D is the embedding size
        '''
        
        #PUT YOUR CODE HERE