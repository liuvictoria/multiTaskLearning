from utils import Tensor_Hook, Module_Hook


"""
=========== VARNET_BLOCK MTL============
difference arises from two etas / need to pass in contrast for MTL
"""

class VarNetBlockMTL(nn.Module):
    """
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    Adds 
    """

    def __init__(self, model: nn.Module, datasets: List[str]):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        
        self.parameter_hooks = []
  

    def configure_hooks(self, contrast_batches):
        '''
        full backward hooks for gradient accumulation
        '''
        self.debug += 1

        # register hooks for accumulated gradient; don't need hook for etas
        self.parameter_hooks = [
            Tensor_Hook(eta, name = f'eta{self.debug}', accumulated_by = contrast_batches[idx_contrast])
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
        ################################### 
        # deal with hook stuff (eta)
        if sum(contrast_batches) == 1:
            # remove all previous hooks at first batch of next grad acc.
            for parameter_hook in self.parameter_hooks:
                parameter_hook.close()
            # self.parameter_hooks = []
            

        # if true, we are in the last batch before loss.backward() for grad. acc.
        if create_hooks:
            self.configure_hooks(contrast_batches)
        # else:
        #     assert len(self.parameter_hooks) == 0, 'did not clear VarNetBlock hooks for next grad acc.'
        ################################### 
    


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

        self.uncert_hooks = []

    def configure_hooks(self, contrast_batches):
        '''
        full backward hooks for gradient accumulation
        '''
        self.debug += 1
        # register hooks for accumulated gradient
        self.uncert_hooks = [
            Tensor_Hook(logsigma, name = f'uncert hook{self.debug}', accumulated_by = contrast_batches[idx_contrast])
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
        
        ################################### 
        # deal with hook stuff (logsigma)
        if sum(contrast_batches) == 1:
            # remove all previous hooks at first batch of next grad acc.
            for uncert_hook in self.uncert_hooks:
                uncert_hook.close()

        # if true, we are in the last batch before loss.backward() for grad. acc.
        if create_hooks:
            self.configure_hooks(contrast_batches)
        ####################################





from collections import Counter
import numpy as np

import torch
import torch.nn as nn

from typing import List

import fastmri
from fastmri_varnet import NormUnet
from utils import Tensor_Hook


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
            self.model(self.sens_reduce(current_kspace, sens_maps)), 
            sens_maps
        )

        return current_kspace - soft_dc - model_term

class VarNetBlockMTL(nn.Module):
    """
    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.
    Adds 
    """

    def __init__(self, model: nn.Module, datasets: List[str]):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()

        self.model = model
        self.etas = nn.ParameterList(
            nn.Parameter(torch.ones(1))
            for _ in datasets
            )
        self.parameter_hooks = []
        self.debug = 0
        

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
        self.debug += 1

        # register hooks for accumulated gradient; don't need hook for etas
        self.parameter_hooks = [
            Tensor_Hook(eta, name = f'eta{self.debug}', accumulated_by = contrast_batches[idx_contrast])
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
        ################################### 
        # deal with hook stuff (eta)
        if sum(contrast_batches) == 1:
            # remove all previous hooks at first batch of next grad acc.
            for parameter_hook in self.parameter_hooks:
                parameter_hook.close()
            # self.parameter_hooks = []
            

        # if true, we are in the last batch before loss.backward() for grad. acc.
        if create_hooks:
            self.configure_hooks(contrast_batches)
        # else:
        #     assert len(self.parameter_hooks) == 0, 'did not clear VarNetBlock hooks for next grad acc.'
        ################################### 
        
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
                ), 
                sens_maps
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
            VarNetBlockMTL(
                NormUnet(chans, pools, which_unet = 'Unet',), 
                datasets
                ) for _ in range(block_counts['trueshare'])
        ])

        self.mhushare = nn.ModuleList([
            VarNetBlockMTL(
                NormUnet(chans, pools, which_unet = 'MHUnet', contrast_count = len(datasets),), 
                datasets
                ) for _ in range(block_counts['mhushare'])
        ])

        self.split_contrasts = nn.ModuleList()
        for _ in enumerate(datasets):
            self.split_contrasts.append(nn.ModuleList([
                VarNetBlockMTL(
                    NormUnet(chans, pools, which_unet = 'Unet',), 
                    datasets
                    ) for _ in range(block_counts['split'])
            ])
            )

        self.blockstructures = blockstructures

        # uncert (specifically 2 contrasts)
        self.logsigmas = nn.ParameterList(
            nn.Parameter(torch.FloatTensor([-0.5]))
            for _ in datasets
            )
        self.uncert_hooks = []

        # datasets (i.e. div_coronal_pd_fs, div_coronal_pd)
        self.datasets = datasets

        self.debug = 0

    def configure_hooks(self, contrast_batches):
        '''
        full backward hooks for gradient accumulation
        '''
        self.debug += 1
        # register hooks for accumulated gradient
        self.uncert_hooks = [
            Tensor_Hook(logsigma, name = f'uncert hook{self.debug}', accumulated_by = contrast_batches[idx_contrast])
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
        
        ################################### 
        # deal with hook stuff (logsigma)
        if sum(contrast_batches) == 1:
            # remove all previous hooks at first batch of next grad acc.
            for uncert_hook in self.uncert_hooks:
                uncert_hook.close()

        # if true, we are in the last batch before loss.backward() for grad. acc.
        if create_hooks:
            self.configure_hooks(contrast_batches)
        ####################################

        kspace_pred = masked_kspace.clone()

        # contrast int for the block to determine which eta / 
        try:
            int_contrast = self.datasets.index(contrast)
        except:
            raise ValueError(f'{contrast} is not in self.datasets')

        # make counter for each type of block
        counter = [0 for _ in range(3)] # currently three types of blocks

        # go thru the blocks (usually 12)
        for idx_structure, structure in enumerate(self.blockstructures):
            print(f'on {idx_structure} block, {structure}')
            if structure == 'trueshare':
                kspace_pred = self.trueshare[counter[0]](
                    kspace_pred, masked_kspace, mask, esp_maps, 
                    int_contrast = int_contrast, 
                    contrast_batches = contrast_batches, 
                    create_hooks = create_hooks,
                )
                counter[0] += 1
            elif structure == 'mhushare':
                kspace_pred = self.mhushare[counter[1]](
                    kspace_pred, masked_kspace, mask, esp_maps, 
                    int_contrast = int_contrast,
                    contrast_batches = contrast_batches, 
                    create_hooks = create_hooks,
                )
                counter[1] += 1

            elif structure == 'split':
                kspace_pred = self.split_contrasts[int_contrast][counter[2]](
                        kspace_pred, masked_kspace, mask, esp_maps, 
                        int_contrast = int_contrast, 
                        contrast_batches = contrast_batches, 
                        create_hooks = create_hooks,
                        # if idx_structure + 1 == len(self.blockstructures) else False,
                    )
            else:
                raise ValueError(f'{structure} block structure not supported')
        
        im_coil = fastmri.ifft2c(kspace_pred)
        im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(esp_maps)).sum(
            dim=1, keepdim=True
        )
        
        return kspace_pred, im_comb, self.logsigmas