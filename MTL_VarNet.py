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


"""
=========== Model ============
"""
from models import MTL_VarNet


"""
=========== command line parser ============
"""
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
    help='number of unrolled blocks in total for one forward pass'
)
parser.add_argument(
    '--beginblocks', default=0, type=int,
    help='number of unrolled, split blocks before trunk starts'
)
parser.add_argument(
    '--sharedblocks', default=6, type=int,
    help='number of unrolled blocks in trunk'
)

parser.add_argument(
    '--network', default='varnet',
    help='type of network ie unet or varnet'
)

parser.add_argument(
    '--temp', default=2.0, type=float, 
    help='temperature for DWA (must be positive)'
)
parser.add_argument(
    '--weighting', default='naive',
    help='naive, uncert, dwa, pareto'
)

parser.add_argument(
    '--device', default='cuda:2',
    help='cuda:2 device default'
)


# dataset properties
parser.add_argument(
    '--datasets', nargs='+',
    help='names of one or two sets of data files i.e. div_coronal_pd_fs div_coronal_pd; input the downsampled dataset first',
    required = True,
)

parser.add_argument(
    '--datadir', default='/mnt/dense/vliu/summer_dset/',
    help='data root directory; where are datasets contained'
)

parser.add_argument(
    '--scarcities', default=[0, 1, 2, 3], type=int, nargs='+',
    help='number of samples in second contrast will be decreased by 1/2^N; i.e. 0 1 2'
    )

parser.add_argument(
    '--accelerations', default=[5, 6, 7], type=int, nargs='+',
    help='list of undersampling factor of k-space for training; validation is average acceleration '
    )

parser.add_argument(
    '--centerfracs', default=[0.05, 0.06, 0.07], type=int, nargs='+',
    help='list of center fractions sampled of k-space for training; val is average centerfracs'
    )

# data loader properties
parser.add_argument(
    '--stratified', default=0, type=int,
    help='''if true, stratifies the dataloader'''
)

parser.add_argument(
    '--stratifymethod', default='upsample',
    help='''
    one of [upsample, downsample] for
    scarce, abundant dataset, respectively
    does not matter if --stratified is false'''
)

parser.add_argument(
    '--numworkers', default=16, type=int,
    help='number of workers for PyTorch dataloader'
)




# save / display data
parser.add_argument(
    '--experimentname', default='unnamed_experiment',
    help='experiment name i.e. STL or MTAN_pareto etc.'
)
parser.add_argument(
    '--verbose', default=1, type=int,
    help='''if true, prints to console average loss / metrics'''
)
parser.add_argument(
    '--tensorboard', default=1, type=int,
    help='if true, creates TensorBoard'
)
parser.add_argument(
    '--savefreq', default=10, type=int,
    help='how many epochs per saved recon image'
)

opt = parser.parse_args()





"""
=========== Runs ============
"""    

# datasets
run_name = f"runs/{opt.experimentname}_{opt.network}{opt.beginblocks}:{opt.sharedblocks}_{'_'.join(opt.datasets)}/"
writer_tensorboard = SummaryWriter(log_dir = run_name)

def main(opt):
    basedirs = [
        os.path.join(opt.datadir, dataset)
        for dataset in opt.datasets
    ]
    
    for scarcity in opt.scarcities:
        print(f'experiment w scarcity {scarcity}')
        train_dloader = genDataLoader(
            [f'{basedir}/Train' for basedir in basedirs],
            [scarcity, 0], # downsample
            center_fractions = opt.centerfracs,
            accelerations = opt.accelerations,
            shuffle = True,
            num_workers= opt.numworkers,
            stratified = opt.stratified, balancing = opt.stratifymethod
        )

        val_dloader = genDataLoader(
            [f'{basedir}/Val' for basedir in basedirs],
            [0, 0], # no downsampling
            center_fractions = [np.mean(opt.centerfracs)],
            accelerations = [int(np.mean(opt.accelerations))],
            shuffle = False, # no shuffling to allow visualization
            num_workers= opt.numworkers,
        )
        print('generated dataloaders')

        # other inputs to MTL wrapper
        device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
        varnet = MTL_VarNet(
            opt.datasets,
            num_cascades = opt.numblocks,
            begin_blocks = opt.beginblocks, 
            shared_blocks = opt.sharedblocks
            ).to(device)

        optimizer = torch.optim.Adam(varnet.parameters(),lr = opt.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        print('start training')
        multi_task_trainer(
            train_dloader[0], val_dloader[0], 
            train_dloader[1], val_dloader[1], # ratios dicts
            varnet, device, writer_tensorboard,
            optimizer, scheduler,
            opt,
        )
        
        
main(opt)
writer_tensorboard.flush()
writer_tensorboard.close()
