import numpy as np

import torch
import torch.nn as nn

import fastmri
from fastmri.data import transforms
from fastmri.models.unet import Unet
from fastmri.models.varnet import *


"""
=========== VARNET_BLOCK ============
"""

# We can make one iteration block like this
class VarNetBlock(nn.Module):
    """
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.eta = nn.Parameter(torch.ones(1))

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps)) # F*S operator

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        ) # S^H * F^H operator

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.eta
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
        )

        return current_kspace - soft_dc - model_term



"""
=========== STL_VARNET ============
"""

    
# now we can stack VarNetBlocks to make a unrolled VarNet (with 10 blocks)
class STLVarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()

        self.cascades = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(num_cascades)]
        )
        
    def forward(
        self,
        masked_kspace: torch.Tensor, 
        mask: torch.Tensor,
        sens_maps: torch.Tensor
    ) -> torch.Tensor:
        
        kspace_pred = masked_kspace.clone()

        for cascade in self.cascades:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        
        im_coil = fastmri.ifft2c(kspace_pred)
        im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        )
        
        return kspace_pred, im_comb



"""
=========== MTL_VarNet ============
"""


class MTLVarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        shared_blocks: int = 10,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()

        task_blocks = num_cascades - shared_blocks
        
        # define shared trunk
        self.trunk = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(shared_blocks)]
        )
        
        # define task specific layers
        self.pred_contrast1 = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(task_blocks)]
        )
        self.pred_contrast2 = nn.ModuleList(
            [VarNetBlock(NormUnet(chans, pools)) for _ in range(task_blocks)]
        )

        # uncert
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5,]))

        
    def forward(
        self,
        masked_kspace: torch.Tensor, 
        mask: torch.Tensor,
        esp_maps: torch.Tensor,
        contrast: str,
    ) -> torch.Tensor:
        
        kspace_pred = masked_kspace.clone()

        for cascade in self.trunk:
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, esp_maps)


        ### change here for different runs    
        if contrast == 'div_coronal_pd_fs':
            for cascade in self.pred_contrast1:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, esp_maps)
                
        elif contrast == 'div_coronal_pd':
            for cascade in self.pred_contrast2:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, esp_maps)
        
        im_coil = fastmri.ifft2c(kspace_pred)
        im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(esp_maps)).sum(
            dim=1, keepdim=True
        )
        
        return kspace_pred, im_comb, self.logsigma