"""Docstring for varnet.py

Normalized U-Net implemetnation for unrolled block network.

"""

import math
from typing import List, Tuple

import fastmri
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastmri.data import transforms

from unet import MHUnet
from att_unet import AttUnet

class NormUnet(nn.Module):
    """PyTorch implementation of a Normalized U-Net model.

    This is the same as a regular U-Net, but with normalization applied to the
    input before the U-Net. This keeps the values more numerically stable
    during training.

    Initialization Parameters
    -------------------------
    chans : int
        Number of output channels of the first convolution layer.
    num_pools : int
        Number of down-sampling and up-sampling layers.
    in_chans : int
        Number of channels in the input to the U-Net model.
    out_chans : int
        Number of channels in the output to the U-Net model.
    drop_prob : float
        Dropout probability.
    which_unet : str
        One of [trueshare, mhushare, attenshare, split]
    task_count : int
        Number of dataset tasks

    Forward Parameters
    ------------------
    image : tensor
        4D tensor
    int_task : int
        i.e. 0 for div_coronal_pd_fs, 1 for div_coronal_pd

    Returns
    -------
    4D tensor

    References
    ----------
    https://github.com/facebookresearch/fastMRI/tree/master/fastmri/models

    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        which_unet: str = 'user input required',
        task_count: int = None,
    ):
 
        super().__init__()
        assert which_unet in ['trueshare', 'mhushare', 'attenshare', 'split'], "variable which_unet not supported"
        if which_unet == 'trueshare' or which_unet == 'split':
            decoder_heads = 1
        elif which_unet == 'mhushare' or which_unet == 'attenshare':
            assert task_count > 1, 'no. tasks must be int > 1 for mhu or att unet'
            decoder_heads = task_count

        # attentional network is a separate module
        if which_unet == 'attenshare':
            self.unet = AttUnet(
                in_chans = in_chans,
                out_chans = out_chans,
                chans = chans,
                num_pool_layers = num_pools,
                drop_prob = drop_prob,
                decoder_heads = decoder_heads,
            )

        # trueshare, mhushare, and split all use the same network
        # Differentiation between the three happens in MHUnet or VarNet_MTL
        else:
            self.unet = MHUnet(
                    in_chans = in_chans,
                    out_chans = out_chans,
                    chans = chans,
                    num_pool_layers = num_pools,
                    drop_prob = drop_prob,
                    decoder_heads = decoder_heads,
                )


    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # group norm
        b, c, h, w = x.shape
        x = x.view(b, 2, c // 2 * h * w)

        mean = x.mean(dim=2).view(b, c, 1, 1)
        std = x.std(dim=2).view(b, c, 1, 1)

        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean

    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """Ensure that dimensions match after rounding errors incurred
        during upsampling/downsampling by padding.
        
        """
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 15) + 1
        h_mult = ((h - 1) | 15) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        # TODO: fix this type when PyTorch fixes theirs
        # the documentation lies - this actually takes a list
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py#L3457
        # https://github.com/pytorch/pytorch/pull/16949
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0] : h_mult - h_pad[1], w_pad[0] : w_mult - w_pad[1]]

    def forward(
        self, 
        x: torch.Tensor,
        int_task: int = 0,
        ) -> torch.Tensor:
        if not x.shape[-1] == 2:
            raise ValueError("Last dimension must be 2 for complex.")

        # get shapes for unet and normalize
        x = self.complex_to_chan_dim(x)
        x, mean, std = self.norm(x)
        x, pad_sizes = self.pad(x)

        x = self.unet(
            x, int_task = int_task,
            )

        # get shapes back and unnormalize
        x = self.unpad(x, *pad_sizes)
        x = self.unnorm(x, mean, std)
        x = self.chan_complex_to_last_dim(x)

        return x
