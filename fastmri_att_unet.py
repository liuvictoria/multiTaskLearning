"""
based on code from fastMRI
https://github.com/facebookresearch/fastMRI/tree/master/fastmri/models
"""

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from typing import List


class AttUnet(nn.Module):
    """
    PyTorch implementation of a U-Net model with attention.
    attention in downsampling, bottleneck, and upsampling
    number of attentional gates = number of contrasts
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
            decoder_heads: number of contrasts in dataset
        """
        super().__init__()

        # parameters / sizes
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.decoder_heads = decoder_heads

        filters = [
            chans * 2 ** layer for layer in range(num_pool_layers + 1)
        ]

        ################################################
        ############# shared unet layers ###############
        ################################################

        #### down sample layers ####
        self.global_downsample = nn.ModuleList()
        self.global_downsample_conv = nn.ModuleList() 
        # first pooling layer
        self.global_downsample.append(
            ConvBlock(in_chans, chans, drop_prob)
            )
        self.global_downsample_conv.append(
            ConvBlock(chans, chans, drop_prob)
        )

        # rest of the pooling layers
        for idx_layer in range(0, num_pool_layers - 1):
            self.global_downsample.append(
                ConvBlock(filters[idx_layer], filters[idx_layer + 1], drop_prob)
            )
            self.global_downsample_conv.append(
                ConvBlock(filters[idx_layer + 1], filters[idx_layer + 1], drop_prob)
            )


        #### bottleneck ####
        self.global_bottleneck = nn.ModuleList([
            ConvBlock(filters[-2], filters[-1], drop_prob)
            ])
        self.global_bottleneck_conv = nn.ModuleList([
            ConvBlock(filters[-1], filters[-1], drop_prob)
        ])

        #### up sample layers ####
        self.global_uptranspose = nn.ModuleList()
        # post-skip connection concat conv layers
        self.global_upsample_conv = nn.ModuleList()

        for idx_layer in reversed(range(1, num_pool_layers)):
            self.global_uptranspose.append(
                TransposeConvBlock(filters[idx_layer + 1], filters[idx_layer])
                )
        
            # ConvBlock is one unit block at a time now
            self.global_upsample_conv.append(nn.Sequential(
                ConvBlock(filters[idx_layer + 1], filters[idx_layer], drop_prob),
                ConvBlock(filters[idx_layer], filters[idx_layer], drop_prob),
                ))

        self.global_uptranspose.append(
            TransposeConvBlock(filters[1], filters[0])
            )
        self.global_upsample_conv.append(
            nn.Sequential(
                ConvBlock(filters[1], filters[0], drop_prob),
                ConvBlock(filters[0], filters[0], drop_prob),
                nn.Conv2d(filters[0], self.out_chans, kernel_size = 1, stride = 1),
            )
        )

        ################################################
        ############ task attention layers #############
        ################################################
        self.downsample_att = nn.ModuleList()
        self.downsample_att_conv = nn.ModuleList()
        self.upsample_att = nn.ModuleList()
        self.upsample_att_conv = nn.ModuleList()
        self.bottleneck_att = nn.ModuleList()
        self.bottleneck_att_conv = nn.ModuleList()

        #### downsampling layers ####
        for idx_task in range(self.decoder_heads):
            # att filter layers are lists within a list [idx_task][idx_layer]
            # create [idx_task] list and add att filter for first layer
            self.downsample_att.append(
                nn.ModuleList([AttBlock(filters[0], filters[0])])
                )

            # add rest of att filters
            for idx_layer in range(num_pool_layers - 1):
                self.downsample_att[idx_task].append(
                    AttBlock(
                        2 * filters[idx_layer + 1], 
                        filters[idx_layer + 1]
                    )
                )
        
        # att conv block layers (shared between tasks, in accordance w MTAN)
        # subject to change; maybe it will be best to split this part
        for idx_layer in range(num_pool_layers):
            self.downsample_att_conv.append(
                ConvBlock(filters[idx_layer], filters[idx_layer + 1], drop_prob)
                )


        #### bottleneck ####
        for idx_task in range(self.decoder_heads):
            # att filter layers are lists within a list [idx_task]
            # create [idx_task] list and add att filter for bottleneck's single layer
            self.bottleneck_att.append(
                AttBlock(2 * filters[-1], filters[-1])
                )
        
        # att conv block layers (shared between tasks, in accordance w MTAN)
        # single bottleneck layer
        self.bottleneck_att_conv.append(
            ConvBlock(filters[-1], filters[-1], drop_prob)
            )


        #### upsampling layers ####
        # att conv block layers (shared between tasks, in accordance w MTAN)
        for idx_layer in reversed(range(num_pool_layers)):
            self.upsample_att_conv.append(nn.Sequential(
                TransposeConvBlock(filters[idx_layer + 1], filters[idx_layer]),
                ConvBlock(filters[idx_layer], filters[idx_layer], drop_prob),
            ))

        for idx_task in range(self.decoder_heads):
            # att filter layers are lists within a list [idx_task][idx_layer]
            # create [idx_task] list; filters follow similar pattern so not creating here
            self.upsample_att.append(nn.ModuleList())

            # add all att filters except last one
            for idx_layer in reversed(range(1, num_pool_layers)):
                self.upsample_att[idx_task].append(
                    AttBlock(
                        filters[idx_layer + 1],
                        filters[idx_layer]
                    )
                )

            self.upsample_att[idx_task].append(
                nn.Sequential(
                    AttBlock(
                        filters[1],
                        filters[0]
                    ),
                    nn.Conv2d(filters[0], self.out_chans, kernel_size = 1, stride = 1),
                )
            )

            
        


    def forward(
        self, 
        image: torch.Tensor,
        int_contrast: int,
        ) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
            int_contrast: i.e. 0 for div_coronal_pd_fs, 1 for div_coronal_pd
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        # create lists to hold intermediate values

        #### global ####
        g_avgpool, g_unpool = (
            np.zeros([self.num_pool_layers])
            for _ in range(2)
        )
    

        # down/up-sample have structure [idx_layer][idx_intermediate]
        # 2 intermediate tasks
        g_downsample, g_upsample = (
            np.zeros([self.num_pool_layers, 2])
            for _ in range(2)
        )

        g_bottleneck = np.zeros([2])

        #### task-specific ####
        # atten down/up-sample have structure [idx_layer][idx_intermediate]
        # atten bottleneck has structure [idx_intermediate]
        # three intermediate steps
        atten_downsample, atten_upsample = (
            np.zeros([self.num_pool_layers, 3])
            for _ in range(2)
        )
        atten_bottleneck = np.zeros([3])

        # convert np to lists
        (
            g_downsample, g_upsample,
            g_avgpool, g_unpool, g_bottleneck,
            atten_downsample, atten_upsample,
            atten_bottleneck 
        ) = (
            g_downsample.tolist(), g_upsample.tolist(),
            g_avgpool.tolist(), g_unpool.tolist(), g_bottleneck.tolist(),
            atten_downsample.tolist(), atten_upsample.tolist(),
            atten_bottleneck.tolist() 
        ) 

        # for skip connections
        stack = []


        # actual forward

        #### global ####

        # apply down-sampling layers
        for idx_layer in range(self.num_pool_layers):
            # if first layer, use input image
            if idx_layer == 0:
                g_downsample[idx_layer][0] = self.global_downsample[idx_layer](image)
            else:
                g_downsample[idx_layer][0] = self.global_downsample[idx_layer](
                    g_avgpool[idx_layer - 1]
                    )
            # go thru convs        
            g_downsample[idx_layer][1] = self.global_downsample_conv[idx_layer](
                g_downsample[idx_layer][0]
                )
            # for global skip connection
            stack.append(g_downsample[idx_layer][1])
            
            g_avgpool[idx_layer] = F.avg_pool2d(
                g_downsample[idx_layer][1], kernel_size=2, stride=2, padding=0
                )

        # bottleneck
        g_bottleneck[0] = self.global_bottleneck[0](g_avgpool[-1])
        g_bottleneck[1] = self.global_bottleneck_conv[0](g_bottleneck[0])
        # no pooling, and global_bottleneck_conv changes channel count for global only
        
        # apply up-sampling layers
        for idx_layer in range(self.num_pool_layers):

            # if first layer, use bottleneck output
            if idx_layer == 0:
                g_upsample[idx_layer][0] = self.global_uptranspose[idx_layer](g_bottleneck[-1])
            else:
                g_upsample[idx_layer][0] = self.global_uptranspose[idx_layer](
                    g_upsample[idx_layer - 1][-1]
                    )
            
            # skip connection
            downsample_layer = stack[-(idx_layer + 1)]

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if g_upsample[idx_layer][0].shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if g_upsample[idx_layer][0].shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                g_upsample[idx_layer][0] = F.pad(g_upsample[idx_layer][0], padding, "reflect")

            # go thru convs        
            g_upsample[idx_layer][1] = self.global_upsample_conv[idx_layer](
                # concat skip connection w upsampling
                torch.cat([g_upsample[idx_layer][0], downsample_layer], dim=1)
                )


        #### task-specific ####
        # downsampling layers
        for idx_layer in range(self.num_pool_layers):
            # if first layer, no merge step
            if idx_layer == 0:
                atten_downsample[idx_layer][0] = self.downsample_att[int_contrast][idx_layer](
                    g_downsample[idx_layer][0]
                )
            else:
                atten_downsample[idx_layer][0] = self.downsample_att[int_contrast][idx_layer](
                    # merge att and global
                    torch.cat(
                        [g_downsample[idx_layer][0], atten_downsample[idx_layer - 1][2]],
                        dim=1
                        )
                )
            atten_downsample[idx_layer][1] = (atten_downsample[idx_layer][0]) * g_downsample[idx_layer][1]
            atten_downsample[idx_layer][2] = self.downsample_att_conv[idx_layer](
                atten_downsample[idx_layer][1]
            )
            atten_downsample[idx_layer][2] = F.avg_pool2d(
                atten_downsample[idx_layer][2], kernel_size=2, stride=2, padding=0
                )
        
        # bottleneck
        atten_bottleneck[0] = self.bottleneck_att[int_contrast](
            # merge att and global
            torch.cat(
                (g_bottleneck[0], atten_downsample[-1][-1]),
                dim=1
                )
        )
        atten_bottleneck[1] = (atten_bottleneck[0]) * g_bottleneck[1]
        atten_bottleneck[2] = self.bottleneck_att_conv[0](
            atten_bottleneck[1]
        )
        # no pooling bc it's bottleneck. 

        # upsampling layers
        for idx_layer in range(self.num_pool_layers):
            # if first layer, use bottleneck output
            if idx_layer == 0:
                atten_upsample[idx_layer][0] = self.upsample_att_conv[idx_layer](
                    atten_bottleneck[-1]
                )
            else:
                atten_upsample[idx_layer][0] = self.upsample_att_conv[idx_layer](
                    atten_upsample[idx_layer - 1][-1]
                )
            
            atten_upsample[idx_layer][1] = self.upsample_att[int_contrast][idx_layer](
                torch.cat(
                    (g_upsample[idx_layer][0], atten_upsample[idx_layer][0]),
                    dim=1,
                )
            )
            
            atten_upsample[idx_layer][2] = (atten_upsample[idx_layer][1]) * g_upsample[idx_layer][-1]

        return atten_upsample[-1][-1]























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
        Differs from fastmri unet ConvBlock in that there is only one "block"
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size = 3, padding = 1, bias = False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
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
        image = image.clone()
        return self.layers(image)


class AttBlock(nn.Module):
    """
    attentional block 
    1x1 conv, Norm, ReLU
    1x1 conv, Norom, Sigmoid
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            chans: Number of channels after first layter
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size = 1, padding = 0),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope = 0.2, inplace = True),
            nn.Conv2d(out_chans, out_chans, kernel_size = 1, padding = 0),
            nn.InstanceNorm2d(out_chans),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.
        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        image = image.clone()
        return self.layers(image)