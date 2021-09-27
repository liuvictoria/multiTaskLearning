"""Docstring for models.py

Defines STL/MTL unrolled block structure and STL/MTL architectures
"""

from collections import Counter
from typing import List
import numpy as np

import torch
import torch.nn as nn

import fastmri
from fastmri.models.varnet import NormUnet as STLNormUnet
from varnet import NormUnet


"""
=========== VARNET_BLOCK STL vs MTL============
difference between STL vs MTL block arises 
from splitting of etas / need to pass in task for MTL
"""

class VarNetBlockSTL(nn.Module):
    """One unrolled block for STL.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.

    Initialization parameters
    -------------------------
    model : nn.Module
        Fully convolutional network for regularization

    Forward parameters
    ------------------
    current_kspace : tensor
        Partially learned k-space from the previous unrolled block.
    ref_kspace : tensor
        Undersampled k-space input to the network
        (this is the same for all unrolled blocks)
    mask : tensor
        0 / 1 mask for retrospectively undersampling k-space
    sens_maps : tensor
        ESPIRiT-estimated sensitivity maps

    Returns
    -------
    tensor
        Newly estimated k-space after this unrolled block

    References
    ----------
    https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models

    """

    def __init__(self, model: nn.Module):
        super().__init__()

        self.model = model
        self.eta = nn.Parameter(torch.ones(1))
        

    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """Expand single-channel image into multi-channel images

        Uses estimated sensitivity maps.
        """
        return fastmri.fft2c(fastmri.complex_mul(x, sens_maps)) # F*S operator

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """Reduces multi-channel image into single-channel image for neural net

        Uses estimated sensitivity maps.
        """
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
    """One unrolled block for STL.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be composed to form
    the full MTL network.

    Initialization parameters
    -------------------------
    model : nn.Module
        Fully convolutional network for regularization
    datasets : List[str]
        List of task names
    share_etas : bool
        Whether or not share data consistency term amongst tasks
    share_blocks : bool
        Whether or not to share the entire block
        attenshare / mhushare / split are all False.

    Forward parameters
    ------------------
    current_kspace : tensor
        Partially learned k-space from the previous unrolled block.
    ref_kspace : tensor
        Undersampled k-space input to the network
        (this is the same for all unrolled blocks)
    mask : tensor
        0 / 1 mask for retrospectively undersampling k-space
    sens_maps : tensor
        ESPIRiT-estimated sensitivity maps
    int_task : int
        integer representation of task

    Returns
    -------
    tensor
        Newly estimated k-space after this unrolled block

    References
    ----------
    https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models

    """

    def __init__(
        self, model: nn.Module, 
        datasets: List[str], 
        share_etas: bool, 
        share_blocks: bool = True
        ):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
        """
        super().__init__()
        eta_count = 1 if share_etas else len(datasets)

        self.etas = nn.ParameterList(
            nn.Parameter(torch.ones(1))
            for _ in range(eta_count)
            )

        model_count = 1 if share_blocks else len(datasets)
        self.model = nn.ModuleList([model for _ in range(model_count)])
        
        self.share_etas = share_etas
        self.share_blocks = share_blocks
        

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
        int_task: int,
    ) -> torch.Tensor:
  
        mask = mask.bool()
        zero = torch.zeros(1, 1, 1, 1, 1).to(current_kspace)

        # dc eta
        idx_eta = 0 if self.share_etas else int_task
        soft_dc = torch.where(mask, current_kspace - ref_kspace, zero) * self.etas[idx_eta]

        # regularization step (i.e. UNet)
        idx_model = 0 if self.share_blocks else int_task
        model_term = self.sens_expand(
            self.model[idx_model](
                self.sens_reduce(current_kspace, sens_maps),
                int_task = int_task,
                ), 
                sens_maps
        )

        return current_kspace - soft_dc - model_term



"""
=========== STL_VARNET ============
"""

    
# now we can stack VarNetBlocks to make a unrolled VarNet (with 10 blocks)
class STL_VarNet(nn.Module):
    """Full variational STL network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.

    Initialization Parameters
    -------------------------
    num_cascades : int
        Number of cascades (i.e., layers) for variational network.
    chans : int, default = 18
        Number of channels for cascade U-Net.
    pools : int, default = 4
        Number of downsampling and upsampling layers for cascade U-Net.

    Forward parameters
    ------------------
    masked_kspace : tensor
        Undersampled k-space input to the network
    mask : tensor
        0 / 1 mask for retrospectively undersampling k-space
    sens_maps : tensor
        ESPIRiT-estimated sensitivity maps

    Returns
    -------
    kspace_pred : tensor
        predicted k-space; to be compared to ground truth
    img_comb : tensor
        reconstructed image using kspace_pred

    References
    ----------
    https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models

    """

    def __init__(
        self,
        num_cascades: int,
        chans: int = 18,
        pools: int = 4,
    ):
        super().__init__()

        self.cascades = nn.ModuleList(
            [VarNetBlockSTL(STLNormUnet(chans, pools)) for _ in range(num_cascades)]
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
    """Full variational MTL network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    Multi-task learning architecture is constructed according to user input.

    Currently, use must manually comment in / out the evaluation portion for
    distributed GPU training; future releases will automate this.

    Initialization Parameters
    -------------------------
    datasets : list
        task names
    blockstructures : list
        elements must be in [trueshare, mhushare, attenshare, split]
    share_etas : bool
        Whether or not to share data consistency term, eta, amongst tasks
    device : list
        GPU names. Accommodates one or two GPUs.
    chans : int, default = 18
        Number of channels for cascade U-Net
    pools : int, default = 4
        Number of downsampling and upsampling layers for cascade U-Net
    training : bool, default = True
        Training or evaluation. This determines distributed training on GPUs

    Forward parameters
    ------------------
    masked_kspace : tensor
        Undersampled k-space input to the network
    mask : tensor
        0 / 1 mask for retrospectively undersampling k-space
    esp_maps : tensor
        ESPIRiT-estimated sensitivity maps

    Returns
    -------
    kspace_pred : tensor
        predicted k-space; to be compared to ground truth
    img_comb : tensor
        reconstructed image using kspace_pred
    logsigmas : nn.ParameterList
        homoscedastic uncertainty for individual tasks

    References
    ----------
    https://github.com/facebookresearch/fastMRI/blob/main/fastmri/models

    """

    def __init__(
        self,
        datasets: list,
        blockstructures: list,
        share_etas: bool,
        device: list,
        chans: int = 18,
        pools: int = 4,
        training = True,
    ):
        super().__init__()

        # inputs
        self.blockstructures = blockstructures
        self.datasets = datasets # datasets (i.e. div_coronal_pd_fs, div_coronal_pd)
        self.share_etas = share_etas
        self.device = device

        # master list of all unrolled blocks, in sequential order
        self.unrolled = []

        for blockstructure in blockstructures:
            if blockstructure == 'trueshare':
                self.unrolled.append(VarNetBlockMTL(
                    NormUnet(chans, pools, which_unet = 'trueshare',), 
                    datasets,
                    share_etas = share_etas,
                ))
            
            elif blockstructure == 'mhushare':
                self.unrolled.append(VarNetBlockMTL(
                    NormUnet(chans, pools, which_unet = 'mhushare', task_count = len(datasets),), 
                    datasets,
                    share_etas = share_etas,
                ))
            
            elif blockstructure == 'attenshare':
                self.unrolled.append(VarNetBlockMTL(
                    NormUnet(chans, pools, which_unet = 'attenshare', task_count = len(datasets),), 
                    datasets,
                    share_etas = share_etas,
                ))

            elif blockstructure == 'split':
                self.unrolled.append(VarNetBlockMTL(
                    NormUnet(chans, pools, which_unet = 'split',), 
                    datasets,
                    share_etas = share_etas,
                    share_blocks = False,
                ))

            else:
                raise ValueError(f'{blockstructure} block structure not supported')

        # 

        if training:
            # gpu distributed training if we have two gpus (won't have more than 2 due to space)
            if len(device) > 1:
                self.seq1 = nn.ModuleList([
                    *self.unrolled[: len(blockstructures) // 2]
                ]).to(device[1])

                self.seq2 = nn.ModuleList([
                    *self.unrolled[len(blockstructures) // 2 : len(blockstructures)] 
                ]).to(device[0])

            else:
                self.seq1 = nn.ModuleList([
                    *self.unrolled[::]
                ]).to(device[0])
        

        # evaluation only has one gpu
        else:
            # # need to change this manually depending on if there were 
            # # 1 or 2 GPUs used in training
            # self.seq1 = nn.ModuleList([
            #         *self.unrolled[: len(blockstructures) // 2]
            #     ]).to(device[0])

            # self.seq2 = nn.ModuleList([
            #         *self.unrolled[len(blockstructures) // 2 : len(blockstructures)] 
            #     ]).to(device[0])

            self.seq1 = nn.ModuleList([
                    *self.unrolled[::]
                ]).to(device[0])


        # uncert (specifically 2 tasks)
        self.logsigmas = nn.ParameterList(
            nn.Parameter(torch.FloatTensor([-0.5]))
            for _ in datasets
            ).to(device[0])


    def forward(
        self,
        masked_kspace: torch.Tensor, 
        mask: torch.Tensor,
        esp_maps: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
  

        kspace_pred = masked_kspace.clone()

        # task int for the block to determine which eta / 
        try:
            int_task = self.datasets.index(task)
        except:
            raise ValueError(f'{task} is not in self.datasets')


        # distributed training
        if len(self.device) > 1:
            # start on second gpu
            kspace_pred, masked_kspace = kspace_pred.to(self.device[1]), masked_kspace.to(self.device[1])
            mask, esp_maps = mask.to(self.device[1]), esp_maps.to(self.device[1])

            for block in self.seq1:
                kspace_pred = block(
                    kspace_pred, masked_kspace, mask, esp_maps, 
                    int_task = int_task,
                )
            
            # go to first gpu
            kspace_pred, masked_kspace = kspace_pred.to(self.device[0]), masked_kspace.to(self.device[0])
            mask, esp_maps = mask.to(self.device[0]), esp_maps.to(self.device[0])
            for block in self.seq2:
                kspace_pred = block(
                    kspace_pred, masked_kspace, mask, esp_maps, 
                    int_task = int_task,
                )

        else:
            # do all work on first gpu
            kspace_pred, masked_kspace = kspace_pred.to(self.device[0]), masked_kspace.to(self.device[0])
            mask, esp_maps = mask.to(self.device[0]), esp_maps.to(self.device[0])
            for block in self.seq1:
                kspace_pred = block(
                    kspace_pred, masked_kspace, mask, esp_maps, 
                    int_task = int_task,
                )

        # training always ends on first gpu; important for loss functions
        
        im_coil = fastmri.ifft2c(kspace_pred)
        im_comb = fastmri.complex_mul(im_coil, fastmri.complex_conj(esp_maps)).sum(
            dim=1, keepdim=True
        )
        
        return kspace_pred, im_comb, self.logsigmas