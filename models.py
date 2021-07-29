from collections import Counter
import numpy as np

import torch
import torch.nn as nn

from typing import List

import fastmri
from fastmri_varnet import NormUnet
from utils import Hook


"""
=========== VARNET_BLOCK STL vs MTL============
difference arises from two etas / need to pass in contrast for MTL
"""

class VarNetBlockSTL(nn.Module):
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

class VarNetBlockMTL(nn.Module):
    """
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    Adds 
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.etas = nn.Parameter(torch.ones(2)) # specifically two contrasts
        self.parameter_hooks = []
        

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps)) # F*S operator

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
            dim=1, keepdim=True
        ) # S^H * F^H operator

    def configure_hooks(self, contrast_batches):
        '''
        full backward hooks for gradient accumulation
        '''
        # double check that hooks are cleared
        assert len(self.parameter_hooks) == 0, 'VarNetBlock parameter hooks not cleared'

        # register hooks for accumulated gradient
        self.parameter_hooks = [
            Hook(eta, accumulated_by = contrast_batches[idx_contrast])
            for idx_contrast, eta in enumerate(self.etas)
        ] 

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        int_contrast: int,
        contrast_batches: List[int] = None,
        create_hooks: bool = False,
    ) -> torch.Tensor:
        '''
        note that contrast is not str, but rather int index of opt.datasets
        this is implemented in the VarNet portion
        '''
        if sum(contrast_batches) == 1:
            # remove all previous hooks at first batch of next grad acc.
            for parameter_hook in self.parameter_hooks:
                parameter_hook.close()
            self.parameter_hooks = []
        
        assert len(self.parameter_hooks) == 0, 'did not clear VarNetBlock hooks for next grad acc.'

        # if true, we are in the last batch before loss.backward() for grad. acc.
        if create_hooks:
            configure_hooks(contrast_batches)

        mask = mask.bool()
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)

        # dc eta
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.etas[int_contrast]

        model_term = self.sens_expand(
            self.model(
                self.sens_reduce(current_kspace, sens_maps),
                int_contrast = int_contrast,
                contrast_batches = contrast_batches, 
                create_hooks = create_hooks,
                ), sens_maps
        )

        return current_kspace - soft_dc - model_term



"""
=========== STL_VARNET ============
"""

    
# now we can stack VarNetBlocks to make a unrolled VarNet (with 10 blocks)
class STL_VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()

        self.cascades = nn.ModuleList(
            [VarNetBlockSTL(NormUnet(chans, pools)) for _ in range(num_cascades)]
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

class MTL_VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        datasets: list,
        blockstructures: list,
        chans: int = 18,
        pools: int = 4,
        
    ):
        super().__init__()

        # figure out how many blocks of each type:
        block_counts = Counter(blockstructures)

        self.trueshare = nn.ModuleList([
            VarNetBlockMTL(NormUnet(
                chans, pools, which_unet = 'Unet',
                )) for _ in range(block_counts['trueshare'])
        ])

        self.mhushare = nn.ModuleList([
            VarNetBlockMTL(NormUnet(
                chans, pools, which_unet = 'MHUnet', contrast_count = 2,
                )) for _ in range(block_counts['mhushare'])
        ])

        self.split_contrast1 = nn.ModuleList([
            VarNetBlockMTL(NormUnet(
                chans, pools, which_unet = 'Unet',
                )) for _ in range(block_counts['split'])
        ])

        self.split_contrast2 = nn.ModuleList([
            VarNetBlockMTL(NormUnet(
                chans, pools, which_unet= 'Unet',
                )) for _ in range(block_counts['split'])
        ])

        self.blockstructures = blockstructures

        # uncert (specifically 2 contrasts)
        self.logsigmas = nn.Parameter(torch.FloatTensor([-0.5, -0.5,]))
        self.uncert_hooks = []

        # datasets (i.e. div_coronal_pd_fs, div_coronal_pd)
        self.datasets = datasets

    def configure_hooks(self, contrast_batches):
        '''
        full backward hooks for gradient accumulation
        '''
        # double check that hooks are cleared
        assert len(self.uncert_hooks) == 0, 'VarNet uncert hooks not cleared'

        # register hooks for accumulated gradient
        self.uncert_hooks = [
            Hook(logsigma, accumulated_by = contrast_batches[idx_contrast])
            for idx_contrast, logsigma in enumerate(self.logsigmas)
        ] 

    def forward(
        self,
        masked_kspace: torch.Tensor, 
        mask: torch.Tensor,
        esp_maps: torch.Tensor,
        contrast: str,
        contrast_batches: List[int] = None,
        create_hooks: bool = False,
    ) -> torch.Tensor:
        
        kspace_pred = masked_kspace.clone()

        # contrast int for the block to determine which eta / 
        try:
            int_contrast = self.datasets.index(contrast)
        except:
            raise ValueError(f'{contrast} is not in self.datasets')
        
        
        if sum(contrast_batches) == 1:
            # remove all previous hooks at first batch of next grad acc.
            for uncert_hook in self.uncert_hooks:
                uncert_hook.close()
            self.uncert_hooks = []
        
        assert len(self.parameter_hooks) == 0, 'did not clear VarNetBlock hooks for next grad acc.'

        # if true, we are in the last batch before loss.backward() for grad. acc.
        if create_hooks:
            configure_hooks(contrast_batches)
        
        # make iterables for each type of block
        trueshare_loader = iter(self.trueshare)
        mhushare_loader = iter(self.mhushare)
        split_contrast1_loader = iter(self.split_contrast1)
        split_contrast2_loader = iter(self.split_contrast2)
        
        # go thru the blocks (usually 12)
        for structure in self.blockstructures:
            if structure == 'trueshare':
                kspace_pred = next(trueshare_loader(
                    kspace_pred, masked_kspace, mask, esp_maps, 
                    int_contrast = int_contrast, 
                    contrast_batches = contrast_batches, create_hooks = create_hooks,
                ))
            elif structure == 'mhushare':
                kspace_pred = next(mhushare_loader(
                    kspace_pred, masked_kspace, mask, esp_maps, 
                    int_contrast = int_contrast,
                    contrast_batches = contrast_batches, create_hooks = create_hooks,
                ))

            elif structure == 'split':
                if contrast == self.datasets[0]:
                    kspace_pred = next(split_contrast1_loader(
                        kspace_pred, masked_kspace, mask, esp_maps, 
                        int_contrast = int_contrast, 
                        contrast_batches = contrast_batches, create_hooks = create_hooks,
                    ))
                elif contrast == self.datasets[1]:
                    kspace_pred = next(split_contrast2_loader(
                        kspace_pred, masked_kspace, mask, esp_maps, 
                        int_contrast = int_contrast, 
                        contrast_batches = contrast_batches, create_hooks = create_hooks,
                    ))
            else:
                raise ValueError(f'{structure} block structure not supported')
        
        im_coil = fastmri.ifft2c(kspace_pred)
        im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(esp_maps)).sum(
            dim=1, keepdim=True
        )
        
        return kspace_pred, im_comb, self.logsigmas