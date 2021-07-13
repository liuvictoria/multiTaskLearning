import os
import argparse
import glob
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter

import fastmri

from dloader import genDataLoader
import pandas as pd
import bokeh.plotting

from utils import criterion, metrics
from utils import plot_quadrant
        
# command line argument parser
parser = argparse.ArgumentParser(
    description = 'define parameters and roots for STL training'
)

# model stuff
parser.add_argument(
    '--network', default='varnet',
    help='type of network ie unet or varnet'
)

parser.add_argument(
    '--trunkblocks', type=int,
    help='number of unrolled blocks in trunk; only for MTL; for STL, use 0',
    required = True
)

parser.add_argument(
    '--numblocks', default=12, type=int,
    help='number of unrolled blocks in total for one forward pass'
)


parser.add_argument(
    '--device', default='cuda:2',
    help='cuda:2 device default'
)

parser.add_argument(
    '--mixeddata', default=True, type=bool,
    help='''If true, the model trained on mixed data;
        almost always true except for STL trained on single contrast'''
)

# dataset properties
parser.add_argument(
    '--datadir', default='/mnt/dense/vliu/summer_dset/',
    help='data root directory; where are datasets contained'
)

parser.add_argument(
    '--datasets', nargs='+',
    help='''names of two sets of data files 
        i.e. div_coronal_pd_fs div_coronal_pd; 
        input the downsampled dataset first''',
    required = True
)

parser.add_argument(
    '--accelerations', default=[5, 6, 7], type=int, nargs='+',
    help='list of undersampling factor of k-space for training; validation is average acceleration '
    )

parser.add_argument(
    '--centerfracs', default=[0.05, 0.06, 0.07], type=int, nargs='+',
    help='list of center fractions sampled of k-space for training; val is average centerfracs'
    )

parser.add_argument(
    '--numworkers', default=16, type=int,
    help='number of workers for PyTorch dataloader'
)


# plot properties
parser.add_argument(
    '--colors', nargs='+', default = ['red', 'green', 'blue', 'purple', 'brown', 'orange'],
    help='a list of plot colors of different contrasts / experiments',
)

parser.add_argument(
    '--plotnames', nargs='+', default = ['loss', 'ssim', 'psnr', 'nrmse'],
    help='a list of plot names for image title; probably shouldnt change',
)
parser.add_argument(
    '--tensorboard', default=True, type=bool,
    help='if true, creates TensorBoard of MR'
)
parser.add_argument(
    '--createplots', default=True, type=bool,
    help='if true, creates plots of metrics for different ratios of MRI'
)


# save / display data
parser.add_argument(
    '--experimentnames', nargs='+',
    help='''experiment name i.e. STL or MTAN_pareto etc.
        if data is not mixed, only give one test! 
        If doing tensorboard MR images, only give one test (to declutter dir names)
        At most three tests for mixed data to prevent clutter''',
    required = True
    
)

opt = parser.parse_args()


### global variables ###

# add to this every time new model is trained
from models import STLVarNet
from models import MTLVarNet
        
        
# preliminary plot initialization / colors
if opt.createplots:
    if opt.mixeddata:
        x_axis_label = f'no. slices of {opt.datasets[0]}'
    else:
        x_axis_label = f'no. slices'

    plots = [
        bokeh.plotting.figure(
            width = 700,
            height = 340,
            x_axis_label = x_axis_label,
            y_axis_label = opt.plotnames[i],
            tooltips=[
                (opt.plotnames[i], f"@{{{opt.plotnames[i]}}}"),
                (f"no. slices {opt.datasets[0]}", f"@{{{opt.datasets[0]}}}"),
                (f"no. slices {opt.datasets[1]}", f"@{{{opt.datasets[1]}}}"),
            ],
        ) for i in range(4) 
    ]


    _dataset_exps = [
        f'{dataset} ~ {experimentname}' 
        for dataset in opt.datasets 
        for experimentname in opt.experimentnames]

    colormap = {
        dataset_exp : opt.colors[i]
        for i, dataset_exp in enumerate(_dataset_exps)
    }
     
    
    
    
def df_single_contrast_all_models(
    the_model, test_dloader, model_filedir, contrast, writer
):
    '''
    creates dataframe ready for bokeh plotting
    '''
    
    modelpaths = glob.glob(f"{model_filedir}/*.pt") ###
    
    # column names
    if opt.mixeddata:
        df_row = np.zeros([len(modelpaths), 6])
        columns = ['loss', 'ssim', 'psnr', 'nrmse', opt.datasets[0], opt.datasets[1]]
    else:
        df_row = np.zeros([len(modelpaths), 5])
        columns = ['loss', 'ssim', 'psnr', 'nrmse', contrast]
    
    
    with torch.no_grad():
        
        for idx_model, model_filepath in enumerate(modelpaths):
            # model name
            model = os.path.basename(model_filepath)[:-6]
            # load model
            the_model.load_state_dict(torch.load(model_filepath))

            # iterate thru test set
            test_batch = len(test_dloader[0])
            test_dataset = iter(test_dloader[0])
            
            #test_dloader[2] contains number of slices per mri
            for idx_mri, nsl in enumerate(test_dloader[2]): 
                for idx_slice in range(nsl):
                    kspace, mask, esp_maps, im_fs, contrast = next(test_dataset)
                    contrast = contrast[0]
                    kspace, mask = kspace.to(opt.device), mask.to(opt.device)
                    esp_maps, im_fs = esp_maps.to(opt.device), im_fs.to(opt.device)

                    # forward pass
                    if 'STL' in opt.experimentnames:
                        _, im_us = the_model(kspace, mask, esp_maps) 
                    elif 'MTL' in opt.experimentnames:
                        _, im_us = the_model(kspace, mask, esp_maps, contrast)
                    
                    # L1 loss
                    loss = criterion(im_fs, im_us)
                    df_row[idx_model, 0] += loss.item() / test_batch

                    # ssim, psnr, nrmse
                    for j in range(3):
                        df_row[idx_model, j + 1] += metrics(im_fs, im_us)[j] / test_batch

                    if opt.tensorboard and idx_slice % 2 == 0:
                        writer.add_figure(
                            f'{model}/{contrast}/MRI_{idx_mri}', 
                            plot_quadrant(im_fs, im_us),
                            global_step = idx_slice,
                        )     
            
            # define x axis
            ratio_1 = int(model.split('_')[0].split('=')[1])
            df_row[idx_model, 4] = ratio_1

            if opt.mixeddata:
                ratio_2 = int(model.split('_')[1].split('=')[1])
                df_row[idx_model, 5] = ratio_2

        
    return pd.DataFrame(
        df_row,
        columns=columns
    )


def save_bokeh_plots(writer):
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
        )

        # iterate through each model folder
        for idx_experimentname, experimentname in enumerate(opt.experimentnames):

            # normally, we are in mixeddata case
            if opt.mixeddata:
                # MTL (experimental)
                if opt.trunkblocks != 0:
                    model_filedir = f"models/{experimentname}_{opt.network}{opt.trunkblocks}_{'_'.join(opt.datasets)}"
                # STL mixed (control)
                else:
                    model_filedir = f"models/{experimentname}_{opt.network}_{'_'.join(opt.datasets)}"
            # only for STL where data are not mixed (control)
            else:
                model_filedir = f"models/{experimentname}_{opt.network}_{dataset}"
 
            # figure out which model skeleton to use
            if 'STL' in experimentname:
                with torch.no_grad():
                    the_model = STLVarNet(
                        num_cascades = opt.numblocks,
                        ).to(opt.device)
            elif 'MTL' in experimentname:
                with torch.no_grad():
                    the_model = MTLVarNet(
                        num_cascades = opt.numblocks,
                        shared_blocks = opt.trunkblocks,
                        ).to(opt.device)

            # get df for all ratios of a particular model, for a single contrast
            df = df_single_contrast_all_models(
                the_model, test_dloader, model_filedir, dataset, writer
            )
            
            if opt.createplots:
                if opt.mixeddata:
                    df = df.sort_values(by=[f'{opt.datasets[0]}'])
                    x_data = opt.datasets[0]
                else:
                    df = df.sort_values(by=[f'{dataset}'])
                    x_data = dataset 


                for idx_plot in range(4):
                # Add glyphs
                    plots[idx_plot].circle(
                        source = df,
                        x = x_data,
                        y = opt.plotnames[idx_plot],
                        legend_label = f'{dataset} ~ {experimentname}',
                        color = colormap[f'{dataset} ~ {experimentname}'],
                    )

                    plots[idx_plot].line(
                        source = df,
                        x = x_data,
                        y = opt.plotnames[idx_plot],
                        legend_label = f'{dataset} ~ {experimentname}',
                        color = colormap[f'{dataset} ~ {experimentname}']
                    )
    
        
    if opt.createplots:     
        # customize and save plots
        if opt.trunkblocks != 0:
            plotdir = f"plots/{'_'.join(opt.experimentnames)}_{opt.network}{opt.trunkblocks}_{'_'.join(opt.datasets)}"
        else:
            plotdir = f"plots/{'_'.join(opt.experimentnames)}_{opt.network}_{'_'.join(opt.datasets)}"
            
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)

        for j in range(4):
            if opt.mixeddata:
                title = f"{opt.plotnames[j]} for scarce {opt.datasets[0]} and abundant {opt.datasets[1]}"
            else:
                title = f"{opt.plotnames[j]} for scarce {opt.datasets[0]} and scarce {opt.datasets[1]}"

            plots[j].add_layout(plots[j].legend[0], 'left')
            plots[j].legend.click_policy = "hide"
            plots[j].title = title
            bokeh.io.save(
                plots[j],
                filename = os.path.join(
                    plotdir,
                    f"{opt.plotnames[j]}.html"),
                title = "Bokeh plot",
            )
    return True





# main
if opt.trunkblocks != 0:
    log_dir = f"plots/{'_'.join(opt.experimentnames)}_{opt.network}{opt.trunkblocks}_{'_'.join(opt.datasets)}"
else:
    log_dir = f"plots/{'_'.join(opt.experimentnames)}_{opt.network}_{'_'.join(opt.datasets)}"
            


if opt.tensorboard:
    writer_tensorboard = SummaryWriter(
        log_dir = log_dir,
        max_queue = 20,
        flush_secs = 1,
    )
else:
    writer_tensorboard = None
    
save_bokeh_plots(writer_tensorboard)