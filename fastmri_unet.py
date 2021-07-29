"""
code modified from fastMRI
https://github.com/facebookresearch/fastMRI/tree/master/fastmri/models
"""

import torch
from torch import nn
from torch.nn import functional as F

from typing import List
from utils import Hook


class MHUnet(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    Shared encoder and separatedly trained decoder
    for the different contrasts
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
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
            decoder_heads: number of split decoder heads
        """
        super().__init__()

        # parameters / sizes
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.decoder_heads = decoder_heads

        # down sample layers
        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        # before going to downsampling
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        # recall no. current channels after downsampling; constant
        downsampling_ch = ch

        # different decoder blocks for different contrasts
        # up sample layers; contrast no. of modules
        self.up_transpose_convs = nn.ModuleList()
        # post-skip connection concat conv layers; contrast no. of modules
        self.up_convs = nn.ModuleList()

        for idx_head in range(decoder_heads):
            # get no. channels from downsampling
            ch = downsampling_ch
            self.up_transpose_convs.append(nn.ModuleList())
            self.up_convs.append(nn.ModuleList())
        
            for _ in range(num_pool_layers - 1):
                self.up_transpose_convs[idx_head].append(TransposeConvBlock(ch * 2, ch))
                self.up_convs[idx_head].append(ConvBlock(ch * 2, ch, drop_prob))
                ch //= 2

            self.up_transpose_convs[idx_head].append(TransposeConvBlock(ch * 2, ch))
            self.up_convs[idx_head].append(
                nn.Sequential(
                    ConvBlock(ch * 2, ch, drop_prob),
                    nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                )
            )

        # initialize categories of hooks (shared, split)
        self.shared_hooks = []
        self.split_hooks = [[] for _ in range(decoder_heads)]


    def configure_hooks(self, contrast_batches):
        '''
        full backward hooks for gradient accumulation
        '''
        # double safety; supposedly already checked after removing hooks
        assert len(self.shared_hooks) == 0, 'unet shared hooks not cleared'
        assert len(self.split_hooks) == 0, 'unet split hooks not cleared'
        # register hooks for accumulated gradient
        # start with shared
        self.shared_hooks = [
            Hook(shared_module, accumulated_by = sum(contrast_batches))
            for shared_module in self.down_sample_layers
        ]
        self.shared_hooks.extend(
            Hook(self.conv, accumulated_by = sum(contrast_batches))
        )
        
        # determine if decoder head is shared or split  
        if self.decoder_heads == 1:
            # shared
            self.split_hooks = [
                Hook(split_module, accumulated_by = sum(contrast_batches))
                for decoder_head in self.up_transpose_convs
                for split_module in decoder_head
            ]
            self.split_hooks.extend(
                Hook(split_module, accumulated_by = sum(contrast_batches))
                for decoder_head in self.up_convs
                for split_module in decoder_head
            )
        else:
            # split
            self.split_hooks = [
                Hook(split_module, accumulated_by = contrast_batches[idx_head])
                for idx_head, decoder_head in enumerate(self.up_transpose_convs)
                for split_module in decoder_head
            ]
            self.split_hooks.extend(
                Hook(split_module, accumulated_by = contrast_batches[idx_head])
                for idx_head, decoder_head in enumerate(self.up_convs)
                for split_module in decoder_head
            )        

    def forward(
        self, 
        image: torch.Tensor,
        int_contrast: int,
        # for hooks: do positional, not keyword, arguments
        contrast_batches: List[int],
        create_hooks: bool,
        ) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
            int_contrast: i.e. 0 for div_coronal_pd_fs, 1 for div_coronal_pd
            contrast_batches: used for gradient acc; [no. contrast1, no. contrast2]
            create_hooks: if this is the last gradient acc before optimizer.step
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # figure out what initialized architecture was, so as to forward pass
        if self.decoder_heads == 1:
            int_contrast == 0
        elif self.decoder_heads > 1:
            assert int_contrast >= 0, 'if not sharing decoder, give indiv. int_contrast'
        
        if sum(contrast_batches) == 1:
            # remove all previous hooks at first batch of next grad acc.
            for shared_hook in self.shared_hooks:
                shared_hook.close()
            for decoder_head in self.split_hooks:
                for split_hook in decoder_head:
                    split_hook.close()
            self.shared_hooks = []
            self.split_hooks = []
        
        assert len(self.shared_hooks) == 0, 'did not clear unet shared hooks for next grad acc.'
        assert len(self.split_hooks) == 0, 'did not clear unet split hooks for next grad acc.'

        # if true, we are in the last batch before loss.backward() for grad. acc.
        if create_hooks:
            configure_hooks(contrast_batches)
          
            
        
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for idx_upsample, (transpose_conv, conv) in enumerate(zip(
            self.self.up_transpose_convs[int_contrast], 
            self.up_convs[int_contrast],
        )):
            downsample_layer = stack[-idx_upsample]
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
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
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
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
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
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)



# class Unet(nn.Module):
#     """
#     PyTorch implementation of a U-Net model.
#     O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
#     for biomedical image segmentation. In International Conference on Medical
#     image computing and computer-assisted intervention, pages 234–241.
#     Springer, 2015.
#     """

#     def __init__(
#         self,
#         in_chans: int,
#         out_chans: int,
#         chans: int = 32,
#         num_pool_layers: int = 4,
#         drop_prob: float = 0.0,
#     ):
#         """
#         Args:
#             in_chans: Number of channels in the input to the U-Net model.
#             out_chans: Number of channels in the output to the U-Net model.
#             chans: Number of output channels of the first convolution layer.
#             num_pool_layers: Number of down-sampling and up-sampling layers.
#             drop_prob: Dropout probability.
#         """
#         super().__init__()

#         self.in_chans = in_chans
#         self.out_chans = out_chans
#         self.chans = chans
#         self.num_pool_layers = num_pool_layers
#         self.drop_prob = drop_prob

#         self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
#         ch = chans
#         for _ in range(num_pool_layers - 1):
#             self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
#             ch *= 2
#         self.conv = ConvBlock(ch, ch * 2, drop_prob)

#         self.up_conv = nn.ModuleList()
#         self.up_transpose_conv = nn.ModuleList()
#         for _ in range(num_pool_layers - 1):
#             self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
#             self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
#             ch //= 2

#         self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
#         self.up_conv.append(
#             nn.Sequential(
#                 ConvBlock(ch * 2, ch, drop_prob),
#                 nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
#             )
#         )

#     def forward(
#         self, 
#         image: torch.Tensor,
        
#         ) -> torch.Tensor:
#         """
#         Args:
#             image: Input 4D tensor of shape `(N, in_chans, H, W)`.
#         Returns:
#             Output tensor of shape `(N, out_chans, H, W)`.
#         """
#         stack = []
#         output = image

#         # apply down-sampling layers
#         for layer in self.down_sample_layers:
#             output = layer(output)
#             stack.append(output)
#             output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

#         output = self.conv(output)

#         # apply up-sampling layers
#         for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
#             downsample_layer = stack.pop()
#             output = transpose_conv(output)

#             # reflect pad on the right/botton if needed to handle odd input dimensions
#             padding = [0, 0, 0, 0]
#             if output.shape[-1] != downsample_layer.shape[-1]:
#                 padding[1] = 1  # padding right
#             if output.shape[-2] != downsample_layer.shape[-2]:
#                 padding[3] = 1  # padding bottom
#             if torch.sum(torch.tensor(padding)) != 0:
#                 output = F.pad(output, padding, "reflect")

#             output = torch.cat([output, downsample_layer], dim=1)
#             output = conv(output)

#         return output
