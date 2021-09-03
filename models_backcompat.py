
############################ V1 ####################################
############################ sans eta ###################################

import argparse
import os
import numpy as np

import torch
import torch.nn as nn

import fastmri
from fastmri.data import transforms
from fastmri.models.unet import Unet
from fastmri.models.varnet import *

from torch.utils.tensorboard import SummaryWriter

from dloader import genDataLoader
from wrappers import multi_task_trainer



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

    def sens_expand(self, x: torch.Tensor, esp_maps: torch.Tensor) -> torch.Tensor:
        return fastmri.fft2c(fastmri.complex_mul(x, esp_maps)) # F*S operator

    def sens_reduce(self, x: torch.Tensor, esp_maps: torch.Tensor) -> torch.Tensor:
        x = fastmri.ifft2c(x)
        return fastmri.complex_mul(x, fastmri.complex_conj(esp_maps)).sum(
            dim=1, keepdim=True
        ) # S^H * F^H operator

    def forward(
        self,
        current_kspace: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        esp_maps: torch.Tensor,
    ) -> torch.Tensor:
        mask = mask.bool()
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.eta
        model_term = self.sens_expand(
            self.model(self.sens_reduce(current_kspace, esp_maps)), esp_maps
        )

        return current_kspace - soft_dc - model_term
    
   


class MTL_VarNet_backcompat(nn.Module):
    """
    A full variational network model.
    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        datasets,
        num_cascades: int = 12,
        shared_blocks: int = 2,
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
        self.datasets = datasets

        
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
            
        if contrast == self.datasets[0]:
            for cascade in self.pred_contrast1:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, esp_maps)
                
        elif contrast == self.datasets[1]:
            for cascade in self.pred_contrast2:
                kspace_pred = cascade(kspace_pred, masked_kspace, mask, esp_maps)
        
        im_coil = fastmri.ifft2c(kspace_pred)
        im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(esp_maps)).sum(
            dim=1, keepdim=True
        )
        
        return kspace_pred, im_comb, self.logsigma
    

########################### V2 ####################################
########################### two etas, no blockstructures list ###################################

# import numpy as np

# import torch
# import torch.nn as nn

# from typing import List

# import fastmri
# from fastmri.data import transforms
# from fastmri.models.unet import Unet
# from fastmri.models.varnet import *


# """
# =========== VARNET_BLOCK STL vs MTL============
# difference arises from two etas / need to pass in contrast for MTL
# """

# class VarNetBlockSTL(nn.Module):
#     """
#     This model applies a combination of soft data consistency with the input
#     model as a regularizer. A series of these blocks can be stacked to form
#     the full variational network.
#     """

#     def __init__(self, model: nn.Module):
#         """
#         Args:
#             model: Module for "regularization" component of variational
#                 network.
#         """
#         super().__init__()

#         self.model = model
#         self.eta = nn.Parameter(torch.ones(1))
        

#     def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         return fastmri.fft2c(fastmri.complex_mul(x, sens_maps)) # F*S operator

#     def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         x = fastmri.ifft2c(x)
#         return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
#             dim=1, keepdim=True
#         ) # S^H * F^H operator

#     def forward(
#         self,
#         current_kspace: torch.Tensor,
#         ref_kspace: torch.Tensor,
#         mask: torch.Tensor,
#         sens_maps: torch.Tensor,
#     ) -> torch.Tensor:
#         mask = mask.bool()
#         zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
#         soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.eta
#         model_term = self.sens_expand(
#             self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
#         )

#         return current_kspace - soft_dc - model_term

# class VarNetBlockMTL(nn.Module):
#     """
#     This model applies a combination of soft data consistency with the input
#     model as a regularizer. A series of these blocks can be stacked to form
#     the full variational network.
#     Adds 
#     """

#     def __init__(self, model: nn.Module):
#         """
#         Args:
#             model: Module for "regularization" component of variational
#                 network.
#         """
#         super().__init__()

#         self.model = model
#         self.eta_contrast1 = nn.Parameter(torch.ones(1))
#         self.eta_contrast2 = nn.Parameter(torch.ones(1))
        

#     def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         return fastmri.fft2c(fastmri.complex_mul(x, sens_maps)) # F*S operator

#     def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         x = fastmri.ifft2c(x)
#         return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
#             dim=1, keepdim=True
#         ) # S^H * F^H operator

#     def forward(
#         self,
#         current_kspace: torch.Tensor,
#         ref_kspace: torch.Tensor,
#         mask: torch.Tensor,
#         sens_maps: torch.Tensor,
#         int_contrast: int,
#     ) -> torch.Tensor:
#         '''
#         note that contrast is not str, but rather int index of opt.datasets
#         this is implemented in the VarNet portion
#         '''
#         mask = mask.bool()
#         zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
#         if int_contrast == 0:
#             soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.eta_contrast1
#         elif int_contrast == 1:
#             soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.eta_contrast2
#         else:
#             raise ValueError('contrast was neither 0 or 1; cannot find eta parameter')
#         model_term = self.sens_expand(
#             self.model(self.sens_reduce(current_kspace, sens_maps)), sens_maps
#         )

#         return current_kspace - soft_dc - model_term



# """
# =========== STL_VARNET ============
# """

    
# # now we can stack VarNetBlocks to make a unrolled VarNet (with 10 blocks)
# class STL_VarNet(nn.Module):
#     """
#     A full variational network model.
#     This model applies a combination of soft data consistency with a U-Net
#     regularizer. To use non-U-Net regularizers, use VarNetBock.
#     """

#     def __init__(
#         self,
#         num_cascades: int,
#         chans: int = 18,
#         pools: int = 4,
#     ):
#         super().__init__()

#         self.cascades = nn.ModuleList(
#             [VarNetBlockSTL(NormUnet(chans, pools)) for _ in range(num_cascades)]
#         )
        
#     def forward(
#         self,
#         masked_kspace: torch.Tensor, 
#         mask: torch.Tensor,
#         sens_maps: torch.Tensor
#     ) -> torch.Tensor:
        
#         kspace_pred = masked_kspace.clone()

#         for cascade in self.cascades:
#             kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        
#         im_coil = fastmri.ifft2c(kspace_pred)
#         im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(sens_maps)).sum(
#             dim=1, keepdim=True
#         )
        
#         return kspace_pred, im_comb



# """
# =========== MTL_VarNet ============
# """

# class MTL_VarNet_backcompat(nn.Module):
#     """
#     A full variational network model.
#     This model applies a combination of soft data consistency with a U-Net
#     regularizer. To use non-U-Net regularizers, use VarNetBock.
#     """

#     def __init__(
#         self,
#         datasets: list,
#         num_cascades: int,
#         begin_blocks: int,
#         shared_blocks: int,
#         chans: int = 18,
#         pools: int = 4,
        
#     ):
#         super().__init__()
#         if begin_blocks + shared_blocks > num_cascades:
#             raise ValueError(f'beginning and shared blocks are greater than the {num_cascades} allowed blocks')
        
#         task_blocks = num_cascades - shared_blocks - begin_blocks
        
#         # define task specific begin layers
#         self.begin_contrast1 = nn.ModuleList(
#             [VarNetBlockMTL(NormUnet(chans, pools)) for _ in range(begin_blocks)]
#         )
#         self.begin_contrast2 = nn.ModuleList(
#             [VarNetBlockMTL(NormUnet(chans, pools)) for _ in range(begin_blocks)]
#         )

#         # define shared trunk
#         self.trunk = nn.ModuleList(
#             [VarNetBlockMTL(NormUnet(chans, pools)) for _ in range(shared_blocks)]
#         )
        
#         # define task specific end layers
#         self.pred_contrast1 = nn.ModuleList(
#             [VarNetBlockMTL(NormUnet(chans, pools)) for _ in range(task_blocks)]
#         )
#         self.pred_contrast2 = nn.ModuleList(
#             [VarNetBlockMTL(NormUnet(chans, pools)) for _ in range(task_blocks)]
#         )

#         # uncert
#         self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5,]))

#         # datasets
#         self.datasets = datasets

#     def forward(
#         self,
#         masked_kspace: torch.Tensor, 
#         mask: torch.Tensor,
#         esp_maps: torch.Tensor,
#         contrast: str,
#     ) -> torch.Tensor:
        
#         kspace_pred = masked_kspace.clone()

#         # contrast int for the block to determine which eta to use
#         if contrast == self.datasets[0]:
#             int_contrast = 0
#         elif contrast == self.datasets[1]:
#             int_contrast = 1
#         else:
#             raise ValueError(f'{contrast} is not in opt.datasets')

#         # beginning, separate branches
#         if contrast == self.datasets[0]:
#             for cascade in self.begin_contrast1:
#                 kspace_pred = cascade(
#                     kspace_pred, masked_kspace, mask, esp_maps, int_contrast = int_contrast,
#                     )
                
#         elif contrast == self.datasets[1]:
#             for cascade in self.begin_contrast2:
#                 kspace_pred = cascade(
#                     kspace_pred, masked_kspace, mask, esp_maps, int_contrast = int_contrast,
#                     )

#         # merge into trunk
#         for cascade in self.trunk:
#             kspace_pred = cascade(
#                 kspace_pred, masked_kspace, mask, esp_maps, int_contrast = int_contrast,
#                 )
        
#         # split again
#         if contrast == self.datasets[0]:
#             for cascade in self.pred_contrast1:
#                 kspace_pred = cascade(
#                     kspace_pred, masked_kspace, mask, esp_maps, int_contrast = int_contrast
#                     )
                
#         elif contrast == self.datasets[1]:
#             for cascade in self.pred_contrast2:
#                 kspace_pred = cascade(
#                     kspace_pred, masked_kspace, mask, esp_maps, int_contrast = int_contrast,
#                     )
        
#         im_coil = fastmri.ifft2c(kspace_pred)
#         im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(esp_maps)).sum(
#             dim=1, keepdim=True
#         )
        
#         return kspace_pred, im_comb, self.logsigma


########################### V3 ####################################
########################### one eta, blockstructures list ###################################

# from collections import Counter
# import numpy as np

# import torch
# import torch.nn as nn

# from typing import List

# import fastmri
# from fastmri_varnet import NormUnet
# from fastmri.models.varnet import NormUnet as STLNormUnet
# from utils import Tensor_Hook, Module_Hook


# """
# =========== VARNET_BLOCK STL vs MTL============
# difference arises from two etas / need to pass in contrast for MTL
# """

# class VarNetBlockSTL(nn.Module):
#     """
#     This model applies a combination of soft data consistency with the input
#     model as a regularizer. A series of these blocks can be stacked to form
#     the full variational network.
#     """

#     def __init__(self, model: nn.Module):
#         """
#         Args:
#             model: Module for "regularization" component of variational
#                 network.
#         """
#         super().__init__()

#         self.model = model
#         self.eta = nn.Parameter(torch.ones(1))
        

#     def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         return fastmri.fft2c(fastmri.complex_mul(x, sens_maps)) # F*S operator

#     def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         x = fastmri.ifft2c(x)
#         return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
#             dim=1, keepdim=True
#         ) # S^H * F^H operator

#     def forward(
#         self,
#         current_kspace: torch.Tensor,
#         ref_kspace: torch.Tensor,
#         mask: torch.Tensor,
#         sens_maps: torch.Tensor,
#     ) -> torch.Tensor:
#         mask = mask.bool()
#         zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)
#         soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.eta
#         model_term = self.sens_expand(
#             self.model(self.sens_reduce(current_kspace, sens_maps)), 
#             sens_maps
#         )

#         return current_kspace - soft_dc - model_term

# class VarNetBlockMTL(nn.Module):
#     """
#     This model applies a combination of soft data consistency with the input
#     model as a regularizer. A series of these blocks can be stacked to form
#     the full variational network.
#     Adds 
#     """

#     def __init__(self, model: nn.Module, datasets: List[str], share_etas: bool):
#         """
#         Args:
#             model: Module for "regularization" component of variational
#                 network.
#         """
#         super().__init__()
#         eta_count = 1 if share_etas else len(datasets)

#         self.etas = nn.ParameterList(
#             nn.Parameter(torch.ones(1))
#             for _ in range(eta_count)
#             )
#         self.model = model
        
#         self.share_etas = share_etas
        

#     def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         return fastmri.fft2c(fastmri.complex_mul(x, sens_maps)) # F*S operator

#     def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
#         x = fastmri.ifft2c(x)
#         return fastmri.complex_mul(x, fastmri.complex_conj(sens_maps)).sum(
#             dim=1, keepdim=True
#         ) # S^H * F^H operator

#     def forward(
#         self,
#         current_kspace: torch.Tensor,
#         ref_kspace: torch.Tensor,
#         mask: torch.Tensor,
#         sens_maps: torch.Tensor,
#         int_contrast: int,
#     ) -> torch.Tensor:
#         '''
#         note that contrast is not str, but rather int index of opt.datasets
#         this is implemented in the VarNet portion
#         '''
        
#         mask = mask.bool()
#         zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)

#         # dc eta
#         idx_eta = 0 if self.share_etas else int_contrast
#         soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.etas[idx_eta]

#         model_term = self.sens_expand(
#             self.model(
#                 self.sens_reduce(current_kspace, sens_maps),
#                 int_contrast = int_contrast,
#                 ), 
#                 sens_maps
#         )

#         return current_kspace - soft_dc - model_term



# """
# =========== STL_VARNET ============
# """

    
# # now we can stack VarNetBlocks to make a unrolled VarNet (with 10 blocks)
# class STL_VarNet(nn.Module):
#     """
#     A full variational network model.
#     This model applies a combination of soft data consistency with a U-Net
#     regularizer. To use non-U-Net regularizers, use VarNetBock.
#     """

#     def __init__(
#         self,
#         num_cascades: int,
#         chans: int = 18,
#         pools: int = 4,
#     ):
#         super().__init__()

#         self.cascades = nn.ModuleList(
#             [VarNetBlockSTL(STLNormUnet(chans, pools)) for _ in range(num_cascades)]
#         )
        
#     def forward(
#         self,
#         masked_kspace: torch.Tensor, 
#         mask: torch.Tensor,
#         sens_maps: torch.Tensor
#     ) -> torch.Tensor:
        
#         kspace_pred = masked_kspace.clone()

#         for cascade in self.cascades:
#             kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)
        
#         im_coil = fastmri.ifft2c(kspace_pred)
#         im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(sens_maps)).sum(
#             dim=1, keepdim=True
#         )
        
#         return kspace_pred, im_comb



# """
# =========== MTL_VarNet ============
# """

# class MTL_VarNet_backcompat(nn.Module):
#     """
#     A full variational network model.
#     This model applies a combination of soft data consistency with a U-Net
#     regularizer. To use non-U-Net regularizers, use VarNetBock.
#     """

#     def __init__(
#         self,
#         datasets: list,
#         blockstructures: list,
#         share_etas: bool,
#         chans: int = 18,
#         pools: int = 4,
        
#     ):
#         super().__init__()

#         # inputs
#         self.blockstructures = blockstructures
#         self.datasets = datasets # datasets (i.e. div_coronal_pd_fs, div_coronal_pd)
#         self.share_etas = share_etas

#         # figure out how many blocks of each type:
#         block_counts = Counter(self.blockstructures)

#         self.trueshare = nn.ModuleList([
#             VarNetBlockMTL(
#                 NormUnet(chans, pools, which_unet = 'Unet',), 
#                 datasets,
#                 share_etas = share_etas,
#                 ) for _ in range(block_counts['trueshare'])
#         ])
        

#         self.mhushare = nn.ModuleList([
#             VarNetBlockMTL(
#                 NormUnet(chans, pools, which_unet = 'MHUnet', contrast_count = len(datasets),), 
#                 datasets,
#                 share_etas = share_etas,
#                 ) for _ in range(block_counts['mhushare'])
#         ])

#         self.split_contrasts = nn.ModuleList()
#         for idx_contrast, dataset in enumerate(datasets):
#             self.split_contrasts.add_module(
#                 f'splitblock_{idx_contrast}',
#                 nn.ModuleList([
#                 VarNetBlockMTL(
#                     NormUnet(chans, pools, which_unet = 'Unet',), 
#                     datasets,
#                     share_etas = share_etas,
#                     ) for _ in range(block_counts['split'])
#             ])
#             )

#         # a list of all VarNetBlock objects, useful for hooks
#         self.allblocks = nn.ModuleList()
#         for container in [self.trueshare, self.mhushare]:
#             for block in container:
#                 self.allblocks.append(block)
#         for split_container in self.split_contrasts:
#             for block in split_container:
#                 self.allblocks.append(block)

#         # uncert (specifically 2 contrasts)
#         self.logsigmas = nn.ParameterList(
#             nn.Parameter(torch.FloatTensor([-0.5]))
#             for _ in datasets
#             )


#     def forward(
#         self,
#         masked_kspace: torch.Tensor, 
#         mask: torch.Tensor,
#         esp_maps: torch.Tensor,
#         contrast: str,
#     ) -> torch.Tensor:
  

#         kspace_pred = masked_kspace.clone()

#         # contrast int for the block to determine which eta / 
#         try:
#             int_contrast = self.datasets.index(contrast)
#         except:
#             raise ValueError(f'{contrast} is not in self.datasets')

#         # make counter for each type of block
#         counter = [0 for _ in range(3)] # currently three types of blocks

#         # go thru the blocks (usually 12)
#         for idx_structure, structure in enumerate(self.blockstructures):
#             # print(f'on {idx_structure} block, {structure}')
#             if structure == 'trueshare':
#                 kspace_pred = self.trueshare[counter[0]](
#                     kspace_pred, masked_kspace, mask, esp_maps, 
#                     int_contrast = int_contrast,
#                 )
#                 counter[0] += 1
#             elif structure == 'mhushare':
#                 kspace_pred = self.mhushare[counter[1]](
#                     kspace_pred, masked_kspace, mask, esp_maps, 
#                     int_contrast = int_contrast,
#                 )
#                 counter[1] += 1

#             elif structure == 'split':
#                 block = f'splitblock_{int_contrast}'
#                 kspace_pred = getattr(self.split_contrasts, block)[counter[2]](
#                         kspace_pred, masked_kspace, mask, esp_maps, 
#                         int_contrast = int_contrast,
#                     )
#             else:
#                 raise ValueError(f'{structure} block structure not supported')
        
#         im_coil = fastmri.ifft2c(kspace_pred)
#         im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(esp_maps)).sum(
#             dim=1, keepdim=True
#         )
        
#         return kspace_pred, im_comb, self.logsigmas