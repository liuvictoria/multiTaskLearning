"""Docstring for unet.py

Split or multi-head U-Net implementation for unrolled block network.

"""
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

class MHUnet(nn.Module):
    """PyTorch implementation of a U-Net model.
    
    Depending on decoder_heads, can either:
    - Share encoder and split decoder for different tasks OR
    - Share encoder and shared decoder for different tasks

    Initialization Parameters
    -------------------------
    in_chans : int
        Number of channels in the input to the U-Net model.
    out_chans : int
        Number of channels in the output to the U-Net model.
    chans : int, default 32
        Number of output channels of the first convolution layer.
    num_pool_layers : int, default 4
        Number of down-sampling and up-sampling layers.
    drop_prob : float, default 0.0
        Dropout probability.
    decoder_heads : int, default None
        Number of tasks in dataset. Represents how many decoder heads.
        If doing fully shared U-Net, set decoder_heads = None


    Forward Parameters
    ------------------
    image: tensor
        4D tensor of shape `(N, in_chans, H, W)`
    int_task: int
        i.e. 0 for div_coronal_pd_fs, 1 for div_coronal_pd

    Returns
    -------
    4D tensor of shape `(N, out_chans, H, W)`

    References
    ----------
    https://github.com/facebookresearch/fastMRI/tree/master/fastmri/models

    """


    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        decoder_heads = None,
    ):

        super().__init__()

        # parameters / sizes
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.decoder_heads = decoder_heads

        # down sample layers
        self.down_sample_layers = nn.ModuleList() 
        self.down_sample_layers.add_module(
            f'sharedencoder_0',
            ConvBlock(in_chans, chans, drop_prob)
            )
        ch = chans
        for i in range(1, num_pool_layers):
            self.down_sample_layers.add_module(
                f'sharedencoder_{i}',
                ConvBlock(ch, ch * 2, drop_prob)
                )
            ch *= 2
        # before going to downsampling
        self.conv = nn.ModuleList()
        self.conv.add_module(
            f'sharedencoder_0',
            ConvBlock(ch, ch * 2, drop_prob)
        )

        # recall no. current channels after downsampling; constant
        downsampling_ch = ch

        # different decoder blocks for different tasks
        # up sample layers; task no. of modules
        self.up_transpose_convs = nn.ModuleList()
        # post-skip connection concat conv layers; task no. of modules
        self.up_convs = nn.ModuleList()

        hook_type = 'shareddecoder' if decoder_heads == 1 else 'splitdecoder'
        for idx_head in range(decoder_heads):
            # get no. channels from downsampling
            ch = downsampling_ch
            self.up_transpose_convs.append(nn.ModuleList())
            self.up_convs.append(nn.ModuleList())
        
            for i in range(num_pool_layers - 1):
                self.up_transpose_convs[idx_head].add_module(
                    f'task_{idx_head}_{hook_type}_{i}',
                    TransposeConvBlock(ch * 2, ch)
                    )
                self.up_convs[idx_head].add_module(
                    f'task_{idx_head}_{hook_type}_{i}',
                    ConvBlock(ch * 2, ch, drop_prob)
                    )
                ch //= 2

            self.up_transpose_convs[idx_head].add_module(
                f'task_{idx_head}_{hook_type}_{num_pool_layers}',
                TransposeConvBlock(ch * 2, ch)
                )
            self.up_convs[idx_head].add_module(
                f'task_{idx_head}_{hook_type}_{num_pool_layers}',
                nn.Sequential(
                    ConvBlock(ch * 2, ch, drop_prob),
                    nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                )
            )



    def forward(
        self, 
        image: torch.Tensor,
        int_task: int,
        ) -> torch.Tensor:  

        # figure out what initialized architecture was, so as to forward pass
        if self.decoder_heads == 1:
            int_task = 0
        elif self.decoder_heads > 1:
            assert int_task >= 0, 'if not sharing decoder, give indiv. int_task'
        
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        for layer in self.conv:
            output = layer(output)

        # apply up-sampling layers
        for idx_upsample, (transpose_conv, conv) in enumerate(zip(
            self.up_transpose_convs[int_task], 
            self.up_convs[int_task],
        )):
            downsample_layer = stack[-(idx_upsample + 1)]
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")
            
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output



class ConvBlock(nn.Module):
    """A Convolutional Block.
    
    Consists of two convolution layers, each followed by
    instance normalization, LeakyReLU activation and dropout.

    Initialization Parameters
    -------------------------
    in_chans : int
        Number of channels in the input
    out_chans : int
        Number of channels in the output
    drop_prob : float
        Dropout probability

    Forward Parameters
    ------------------
    image: tensor
        Input 4D tensor of shape `(N, in_chans, H, W)`

    Attributes
    ----------
    self.layers : nn.Sequential
        Conv2d, InstanceNorm2d, LeakyReLu

    Returns
    -------
    self.layers(image) : tensor
        4D tensor of shape `(N, out_chans, H, W)`

    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """A Transpose Convolutional Block. 
    Consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.

    Initialization Parameters
    -------------------------
    in_chans : int
        Number of channels in the input
    out_chans : int
        Number of channels in the output

    Forward Parameters
    ------------------
    image : tensor
        4D tensor of shape `(N, in_chans, H, W)`
    
    Attributes
    ----------
    self.layers : nn.Sequential
        convTranspose2d, InstanceNorm2d, LeakyReLu

    Returns
    -------
    self.layers(image) : tensor
        4D tensor of shape `(N, out_chans, H*2, W*2)`

    """

    def __init__(self, in_chans: int, out_chans: int):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image : Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        image = image.clone()
        return self.layers(image)