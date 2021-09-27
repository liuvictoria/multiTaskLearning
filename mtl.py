"""Docstring for mtl.py

This is the user-facing module for MTL training.
Run this module from the command line, and pass in the
appropriate arguments to define the model and data.
"""

import argparse
import json
import os
import numpy as np

import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from dloader import genDataLoader
from wrappers import multi_task_trainer
from utils import label_blockstructures

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
parser.add_argument(
    '--gradaccumulation', default=1, type=int,
    help='how many iterations per gradient accumulation; cannot be less than 1'
)
parser.add_argument(
    '--gradaverage', default=0, type=int,
    help="""if true, will average accumulated grads before optimizer.step
    if false, gradaccumulation will occur without averaging (i.e. no hooks)
    value does not matter if gradaccumulation is equal to 1"""
)


# model training

parser.add_argument(
    '--blockstructures', nargs='+',
    default=[
        'trueshare', 'mhushare', 'attenshare', 'mhushare',
        'split', 'attenshare', 'split', 'split',
        'mhushare', 'mhushare', 'attenshare', 'attenshare',
    ],
    help="""explicit list of what each block will be;
    defines total number of blocks;
    possible options = [
        trueshare, mhushare, attenshare, split
    ]
    trueshare block shares encoder and decoder;
    mhushare block shares encoder but not decoder;
    attenshare block has global shared unet, atten at all levels
    split does not share anything
    """
)

parser.add_argument(
    '--network', default='varnet',
    help='type of network ie unet or varnet'
)

parser.add_argument(
    '--shareetas', default=1, type=int,
    help="""if false, there will be len(opt.datasets) number of etas for each unrolled block
    if true, there will be 1 eta for each unrolled block
    """
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
    '--device', nargs='+', default=['cuda:2'],
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
    help='number of samples in second task will be decreased by 1/2^N; i.e. 0 1 2'
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
    help="""if true, stratifies the dataloader"""
)

parser.add_argument(
    '--stratifymethod', default='upsample',
    help="""
    one of [upsample, downsample] for
    scarce, abundant dataset, respectively
    does not matter if --stratified is false"""
)

parser.add_argument(
    '--numworkers', default=16, type=int,
    help='number of workers for PyTorch dataloader'
)

# save / display data
parser.add_argument(
    '--experimentname', default='unnamed_experiment',
    help='experiment name i.e. STL or MTAN_pareto or MHU_naive_etc'
)
parser.add_argument(
    '--verbose', default=1, type=int,
    help="""if true, prints to console average loss / metrics"""
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

# validation of user inputs
for structure in opt.blockstructures:
    assert structure in ['trueshare', 'mhushare', 'split', 'attenshare'], \
           f'unet structure is not yet a supported block structure'
assert opt.gradaccumulation > 0; 'opt.gradaccumulation must be greater than 0'

assert opt.weighting in ['naive', 'uncert', 'dwa'], f'weighting method not yet supported'



"""
=========== Runs ============
"""    

def main(opt):
    """Calls wrappers.py for training

    Creates data loaders, initializes model, and defines learning parameters.
    Trains MTL from command line.

    Parameters
    ----------
    opt : argparse.ArgumentParser
        Refer to help documentation above.
    
    Returns
    -------
    None

    See Also
    --------
    multi_task_trainer from wrappers.py
    
    """
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
            stratified = opt.stratified, method = opt.stratifymethod
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
        varnet = MTL_VarNet(
            opt.datasets,
            opt.blockstructures,
            opt.shareetas,
            opt.device,
            )

        optimizer = torch.optim.Adam(varnet.parameters(), lr = opt.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        print('start training')
        multi_task_trainer(
            train_dloader[0], val_dloader[0], 
            train_dloader[1], val_dloader[1], # ratios dicts
            varnet, writer_tensorboard,
            optimizer, scheduler,
            opt,
        )


if __name__ == '__main__':
    # run / model name
    run_name = f"runs/{opt.experimentname}_" + \
        f"{'strat_' if opt.stratified else ''}" + \
        f"{opt.network}{label_blockstructures(opt.blockstructures)}_{'_'.join(opt.datasets)}/"
    model_name = f"models/{opt.experimentname}_" + \
        f"{'strat_' if opt.stratified else ''}" + \
        f"{opt.network}{label_blockstructures(opt.blockstructures)}_{'_'.join(opt.datasets)}/"
    if not os.path.isdir(model_name):
        os.makedirs(model_name)
    writer_tensorboard = SummaryWriter(log_dir = run_name)


    # write json files to models and runs directories; for future reference
    with open(
        os.path.join(run_name,'parameters.json'), 'w'
        ) as parameter_file:
        json.dump(vars(opt), parameter_file)     

    with open(
        os.path.join(model_name,'parameters.json'), 'w'
        ) as parameter_file:
        json.dump(vars(opt), parameter_file)   

    main(opt)
    writer_tensorboard.flush()
    writer_tensorboard.close()
