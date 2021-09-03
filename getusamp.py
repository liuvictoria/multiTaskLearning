'''
evaluate models on test dataset
every time new model is changed, add it to ### portions
'''
import os
import argparse

from pathlib import Path
import glob
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import fastmri

from dloader import genDataLoader
import pandas as pd
import bokeh.plotting
# colors for plots
from bokeh.palettes import Category20_10, Category20c_20, Paired10, Set1_8, Set2_8, Colorblind8

from fastmri.data import transforms
from utils import criterion, metrics
from utils import plot_quadrant
from utils import interpret_blockstructures


### add to this every time new model is trained ###
from models_backcompat import MTL_VarNet_backcompat
from models import STL_VarNet
from models import MTL_VarNet
        
# command line argument parser
parser = argparse.ArgumentParser(
    description = 'define parameters and roots for STL training'
)

parser.add_argument(
    '--device', default='cuda:2',
    help='cuda:2 device default'
)

# dataset properties
############## required ##############
parser.add_argument(
    '--datasets', nargs='+',
    help='''names of two sets of data files 
        i.e. div_coronal_pd_fs div_coronal_pd; 
        input the downsampled dataset first''',
    required = True
)

parser.add_argument(
    '--datadir', default='/mnt/dense/vliu/summer_dset/',
    help='data root directory; where are datasets contained'
)

parser.add_argument(
    '--accelerations', default=[6], type=int, nargs='+',
    help='list of undersampling factor of k-space for training; validation is average acceleration '
    )

parser.add_argument(
    '--centerfracs', default=[0.07], type=int, nargs='+',
    help='list of center fractions sampled of k-space for training; val is average centerfracs'
    )

parser.add_argument(
    '--numworkers', default=16, type=int,
    help='number of workers for PyTorch dataloader'
)


# plot properties
############## required ##############
parser.add_argument(
    '--plotdir',
    help='name of plot directory',
    required = True,
)


parser.add_argument(
    '--tensorboard', default=1, type=int,
    help='''if true, creates TensorBoard of MR; 0 1
        note: even if 1, but already has summary.csv, won't give tensorboard'''
)

parser.add_argument(
    '--savefreq', default=2, type=int,
    help='how many slices per saved image'
)


opt = parser.parse_args()


def tensorboard_plot(
    test_dloader, contrast, writer
):
    '''
    creates dataframe ready for bokeh plotting
    '''
    
    test_dataset = iter(test_dloader[0])
    
    #test_dloader[2] contains number of slices per mri
    for idx_mri, nsl in enumerate(test_dloader[2]): 
        for idx_slice in range(nsl):
            kspace, mask, esp_maps, im_fs, contrast = next(test_dataset)
            contrast = contrast[0]
            kspace, mask = kspace.to(opt.device), mask.to(opt.device)
            esp_maps, im_fs = esp_maps.to(opt.device), im_fs.to(opt.device)

            im_us = fastmri.ifft2c(kspace)
            im_us = fastmri.complex_mul(im_us, fastmri.complex_conj(esp_maps)).sum(
                dim=1, keepdim=True
            )
            
            # crop so im_us has same size as im_fs
            im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
    
            if opt.tensorboard and idx_slice % opt.savefreq == 0:
                writer.add_figure(
                    f'undersamp/{contrast}/MRI_{idx_mri}', 
                    plot_quadrant(im_fs, im_us),
                    global_step = idx_slice,
                )
 

def save_usamp_images(writer, opt):

    # do one contrast at a time (two total contrasts)
    for idx_dataset, dataset in enumerate(opt.datasets):
        basedir = os.path.join(opt.datadir, dataset)

        # test loader for this one contrast
        test_dloader = genDataLoader(
            [f'{basedir}/Test'],
            [0, 0],
            center_fractions = [np.mean(opt.centerfracs)],
            accelerations = [int(np.mean(opt.accelerations))],
            shuffle = False,
            num_workers = opt.numworkers,
            # use same mask so aliasing patterns are comparable
            use_same_mask = True, 
        )

        tensorboard_plot(
            test_dloader, dataset, writer
        )
            


    return True


# main
log_dir = f"plots/{opt.plotdir}"

if opt.tensorboard:
    writer_tensorboard = SummaryWriter(
        log_dir = log_dir,
        max_queue = 20,
        flush_secs = 1,
    )
else:
    writer_tensorboard = None
    
save_usamp_images(writer_tensorboard, opt)