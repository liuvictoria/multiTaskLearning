import argparse
import numpy as np

import torch
import torch.nn as nn

import fastmri
from fastmri.data import transforms
from fastmri.models.unet import Unet
from fastmri.models.varnet import *

from torch.utils.tensorboard import SummaryWriter

from dloader import genDataLoader
from utils import criterion, metrics
from utils import plot_quadrant, write_tensorboard
from utils import single_task_trainer


# command line argument parser
parser = argparse.ArgumentParser(
    description = 'define parameters and roots for STL training'
)

# hyperparameters
parser.add_argument(
    '--epochs', default=100, type=int,
    help='number of epochs to run'
)
parser.add_argument(
    '--lr', default=0.0002, type=float,
    help='learning rate'
)


# model training
parser.add_argument(
    '--numblocks', default=12, type=int,
    help='number of unrolled blocks in total'
)
parser.add_argument(
    '--network', default='varnet',
    help='type of network ie unet or varnet'
)
parser.add_argument(
    '--device', default='cuda:2',
    help='cuda:2 device default'
)


# dataset properties
parser.add_argument(
    '--datadir', default='/mnt/dense/vliu/summer_dset/',
    help='data root directory; where are datasets contained'
)

parser.add_argument(
    '--datasets', nargs='+',
    help='names of one or two sets of data files i.e. div_coronal_pd_fs div_coronal_pd; input the downsampled dataset first',
    required = True
)
parser.add_argument(
    '--scarcities', default=[0, 1, 2, 3], type=int, nargs='+',
    help='number of samples in second contrast will be decreased by 1/2^N; i.e. 0 1 2'
    )
parser.add_argument(
    '--accelerations', default=[6], type=int, nargs='+',
    help='list of undersampling factor of k-space; match with centerfracs'
    )
parser.add_argument(
    '--centerfracs', default=[0.06], type=int, nargs='+',
    help='list of center fractions sampled of k-space; match with accelerations'
    )



# save / display data
parser.add_argument(
    '--experimentname', default='unnamed_experiment',
    help='experiment name i.e. STL or MTAN_pareto etc.'
)
parser.add_argument(
    '--verbose', default=True, type=bool,
    help='''if true, prints to console and creatues full TensorBoard
    (if tensorboard is also True)'''
)
parser.add_argument(
    '--tensorboard', default=True, type=bool,
    help='if true, creates TensorBoard'
)
parser.add_argument(
    '--savefreq', default=20, type=int,
    help='how many epochs per saved recon image'
)

opt = parser.parse_args()


"""
=========== Model ============
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
    
    

    
# now we can stack VarNetBlocks to make a unrolled VarNet (with 10 blocks)


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBock.
    """

    def __init__(
        self,
        num_cascades: int = opt.numblocks,
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
=========== Runs ============
"""    

# datasets
run_name = f"runs/{opt.experimentname}_{opt.network}_{'_'.join(opt.datasets)}/"
writer_tensorboard = SummaryWriter(log_dir = run_name)

def main(opt):
    basedirs = [
        os.path.join(opt.datadir, dataset)
        for dataset in opt.datasets
    ]
    
    for scarcity in opt.scarcities:
        print(f'experiment w scarcity {scarcity}')
        train_dloader = genDataLoader(
            [f'{basedir}/Train' for basedir in basedirs], # choose randomly
            [scarcity, 0], # downsample
            center_fractions = opt.centerfracs,
            accelerations = opt.accelerations,
        )

        val_dloader = genDataLoader(
            [f'{basedir}/Val' for basedir in basedirs], # choose randomly
            [0, 0], # no downsampling
            center_fractions = opt.centerfracs,
            accelerations = opt.accelerations,
            shuffle = False, # no shuffling to allow visualization
        )
        print('generated dataloaders')

        # other inputs to STL wrapper
        device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
        varnet = VarNet().to(device)

        optimizer = torch.optim.Adam(varnet.parameters(),lr = opt.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        print('start training')
        single_task_trainer(
            train_dloader[0], val_dloader[0], 
            train_dloader[1], val_dloader[1], # ratios dicts
            varnet, device, writer_tensorboard,
            optimizer, scheduler,
            opt,
        )
        
        
main(opt)
writer_tensorboard.flush()
writer_tensorboard.close()
