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
from bokeh.palettes import Category20_10, Category20c_20, Paired10, Set1_8, Set2_8, Colorblind8, Category10_10

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

# model stuff
# backcompat stuff
parser.add_argument(
    '--backcompat', type=int, nargs='+',
    help='''whether to use back-compatible model
    with begin_blocks and shared_blocks and num_cascades
    arguments''',
    required = True,
)

# backcompat / STL stuff (don't delete)
parser.add_argument(
    '--numblocks', default = [], type=int, nargs = '+',
    help='''number of unrolled blocks in total for one forward pass;
    match to each experimentnames''',
)

# backcompat stuff
parser.add_argument(
    '--beginblocks', default = [], type=int, nargs = '+',
    help='''number of unrolled blocks in total for one forward pass;
    match to each experimentnames''',
)

# backcompat stuff
parser.add_argument(
    '--sharedblocks', default = [], type=int, nargs = '+',
    help='''number of unrolled blocks in total for one forward pass;
    match to each experimentnames''',
)

############## required ##############
parser.add_argument(
    '--blockstructures',
    nargs='+',
    help='''explicit string of what each block is in MTL network;
    i.e. IYYIVVIYV
    I : trueshare
    Y : mhushare
    V : split
    trueshare block shares encoder and decoder;
    mhushare block shares encoder but not decoder;
    split does not share anything.
    For STL, type 'S' for all 12 blocks
    ''',
    required = True,
)

############## required ##############
parser.add_argument(
    '--shareetas', type=int, nargs = '+',
    help='''for each --experimentnames, did we have one or more etas?
    1 for one eta; 0 for more than one etas
    ''',
    required = True,
    )

############## required ##############
parser.add_argument(
    '--stratified', type=int, nargs='+',
    help='''used to find the model name; give as list
    0 for STL experiments''',
    required = True,
)

############## required ##############
parser.add_argument(
    '--scarcemax', type=int,
    help='''497 for div_coronal_pd_fs''',
    required = True,
)

parser.add_argument(
    '--modeldir', default='/../../mnt/dense/vliu/models/',
    help='models root directory; where are pretrained models are'
)
############## required ##############
parser.add_argument(
    '--network', nargs = '+',
    help='type of network ie unet or varnet match to each experimentnames',
    required = True,
)
############## required ##############
parser.add_argument(
    '--mixeddata', type=int, nargs = '+',
    help='''If true, the model trained on mixed data;
        almost always true except for STL trained on single contrast
        give a list to match experimentnames;
        0 for False; 1 for True''',
    required = True,
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
    '--colors', default = ['Category10_10'], nargs = '+',
    help='''Category20_10, Category20c_20, Paired10, Set1_8, Set2_8, Colorblind8, Category10_10
    https://docs.bokeh.org/en/latest/docs/reference/palettes.html#bokeh-palette
    OR
    give a list of custom colors'''
)

parser.add_argument(
    '--plotnames', nargs='+', default = ['loss', 'ssim', 'psnr', 'nrmse'],
    help='a list of plot names for image title; DO NOT CHANGE',
)

parser.add_argument(
    '--createplots', default=1, type=int,
    help='if true, creates plots of metrics for different ratios of MRI; 0 1'
)

parser.add_argument(
    '--showbaselines', default=0, type=int,
    help='''if true, shows baselines for STL non-joint learning;
        line at N=20 for abundant contrast and N=2,5,10,20 for scarce contrast'''
)

parser.add_argument(
    '--showbest', default=0, type=int,
    help='''if true, shows best metric / loss out of all runs; fails for STL_nojoint'''
)

parser.add_argument(
    '--bestdir', default='plots/best',
    help='''directory of where best run summary plots are; csv's must be manually created'''
)

parser.add_argument(
    '--bestruncolor', default = '#FF6C0C',
    help='color for best run; only matters if --showbest is true; default Caltech orange',
)

parser.add_argument(
    '--baselinenetwork', default = 'varnet',
    help='varnet or unet for baselines',
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

# save / display data
############## required ##############
parser.add_argument(
    '--experimentnames', nargs='+',
    help='''list of experiment names i.e. STL or MTAN_pareto etc.''',
    required = True
    
)

opt = parser.parse_args()

# change opt.colors from string to variable of color palette
if len(opt.colors) == 1:
    # best color is in orange so don't have any runs plotted in orange.
    # this is for final poster presentation, so remove this if it's confusing
    if opt.colors[0] == 'Category10_10':
        exec("%s = %s" % ('opt.colors', opt.colors[0]))
        opt.colors = list(opt.colors)
        del opt.colors[1]
    else:
        exec("%s = %s" % ('opt.colors', opt.colors[0]))

        
# preliminary plot initialization / colors
def _initialize_plots(opt):
    x_axis_label = f'% of total PDw-FS MR slices'

    plots = [
        bokeh.plotting.figure(
            width = 900,
            height = 350,
            x_axis_label = x_axis_label,
            y_axis_label = opt.plotnames[i],
            tooltips=[
                (opt.plotnames[i], f"@{{{opt.plotnames[i]}}}"),
                (f"% slices {opt.datasets[0]}", f"@{{{opt.datasets[0]}}}"),
                (f"% slices {opt.datasets[1]}", f"@{{{opt.datasets[1]}}}"),
                (f"% best run", f"@{{{opt.plotnames[i]}name}}"),
            ],
        ) for i in range(4) 
    ]

    if opt.showbaselines:
        plots = _plot_baselines(plots, opt)

    return plots

def _initialize_colormap(opt):
    _dataset_exps = [
        f'{dataset} ~ {experimentname}' 
        for dataset in opt.datasets 
        for experimentname in opt.experimentnames
        ]
    colormap = {
        dataset_exp : opt.colors[i]
        for i, dataset_exp in enumerate(_dataset_exps)
    }
    return colormap


def _plot_best(plots, opt):
    for dataset in opt.datasets:
        df = pd.read_csv(os.path.join(
            opt.bestdir, f'summary_{dataset}.csv'
            ))
        df = df.drop('Unnamed: 0', axis = 1)
        df = df.sort_values(by=[f'{opt.datasets[0]}'])
        x_data = opt.datasets[0]
        for idx_plot in range(4):
            # we might not want to plot best run for each metric;
            # bokeh breaks when plotting NaN, so filter those out first
            if not df[opt.plotnames[idx_plot]].isnull().values.any():
                # Add glyphs
                plots[idx_plot].square(
                    source = df,
                    x = x_data,
                    y = opt.plotnames[idx_plot],
                    legend_label = f'{dataset} ~ best_MTL',
                    color = opt.bestruncolor,
                    line_width = 1.5,
                    size = 10,
                    fill_color = None,

                )

                plots[idx_plot].line(
                    source = df,
                    x = x_data,
                    y = opt.plotnames[idx_plot],
                    legend_label = f'{dataset} ~ best_MTL',
                    line_color = opt.bestruncolor,
                    line_width = 2,
                )
    return plots
    

def _plot_baselines(plots, opt):
    # name paths
    #scarce
    scarce_path = Path(os.path.join(
        opt.modeldir, 
        'STL_baselines',
        f"STL_{opt.baselinenetwork}_{'_'.join(opt.datasets)}", 
        f'summary_{opt.datasets[0]}.csv',
    ))

    # abundant
    abundant_path = Path(os.path.join(
        opt.modeldir,
        'STL_baselines',
        f'STL_nojoint_{opt.baselinenetwork}_{opt.datasets[1]}', 
        f'summary_{opt.datasets[1]}.csv'
    ))

    if scarce_path.is_file() and abundant_path.is_file():
        # read in dfs
        scarce_df = pd.read_csv(scarce_path)
        scarce_df = scarce_df.drop('Unnamed: 0', axis = 1)
        scarce_df = scarce_df.sort_values(by=[f'{opt.datasets[0]}'])
        abundant_df = pd.read_csv(abundant_path)
        abundant_df = abundant_df.drop('Unnamed: 0', axis = 1)
        abundant_df = abundant_df.sort_values(by=[f'{opt.datasets[1]}'])

        for idx_plot in range(4):
            # scarce
            plots[idx_plot].line(
                source = scarce_df,
                x = opt.datasets[0],
                y = opt.plotnames[idx_plot],
                legend_label = f'STL baseline {opt.datasets[0]}',
                color = 'black',
                line_width = 1.5,
                line_dash = [2, 2]
            )

            # abundant
            abundant_value = abundant_df.loc[
                abundant_df[opt.datasets[1]] == max(abundant_df[opt.datasets[1]]), 
                opt.plotnames[idx_plot]
            ]

            plots[idx_plot].line(
                x = [0, 1],
                y = [abundant_value, abundant_value],
                legend_label = f'STL baseline {opt.datasets[1]}',
                color = 'black',
                line_width = 1.5,
                line_dash = 'dashdot'
            )
        print ('successfully plotted baselines')
    else:
        print (f'    one or both of {scarce_path}, {abundant_path} does not exist; no baselines')
    return plots



def df_single_contrast_all_models(
    the_model, test_dloader, model_filedir, contrast, idx_experimentname, writer
):
    '''
    creates dataframe ready for bokeh plotting
    '''

    # check if this has already been run; if yes, don't rerun
    summary_file = Path(
        os.path.join(
            model_filedir, f'summary_{contrast}.csv'
        )
    )
    if summary_file.is_file():
        print (    f'  evaluation {summary_file} has been run before and will be used for current plot')
        return pd.read_csv(
            summary_file
            ).drop(
            'Unnamed: 0', axis = 1
            ).sort_values(
                by=[opt.datasets[0]]
            )

    modelpaths = glob.glob(f"{model_filedir}/*.pt")
    
    # column names
    if opt.mixeddata[idx_experimentname]:
        df_row = np.zeros([len(modelpaths), 6])
        columns = [
            opt.plotnames[0], opt.plotnames[1], 
            opt.plotnames[2], opt.plotnames[3], 
            opt.datasets[0], opt.datasets[1]
            ]
        sort_by = opt.datasets[0]
    else:
        df_row = np.zeros([len(modelpaths), 5])
        columns = [
            opt.plotnames[0], opt.plotnames[1], 
            opt.plotnames[2], opt.plotnames[3], 
            contrast
            ]
        sort_by = contrast
    
    
    with torch.no_grad():
        for idx_model, model_filepath in enumerate(modelpaths):
            # model name for saving MR image purposes
            model = '~'.join(model_filepath.split('models/')[1][:-6].split('/'))

            # load model
            the_model.load_state_dict(torch.load(
                model_filepath, 
                map_location = opt.device,
                )
            )
            the_model.eval()

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
                    if 'STL' in model_filepath:
                        _, im_us = the_model(kspace, mask, esp_maps) 
                    elif 'MTL' in model_filepath:
                        _, im_us, _ = the_model(kspace, mask, esp_maps, contrast)
                    else:
                        raise ValueError(
                            f'Could not go thru forward pass; could not find STL or MTL in {model_filepath}'
                        )
                    
                    # crop so im_us has same size as im_fs
                    im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))

                    # L1 loss
                    loss = criterion(im_fs, im_us)
                    df_row[idx_model, 0] += loss.item() / test_batch

                    # ssim, psnr, nrmse
                    for j in range(3):
                        df_row[idx_model, j + 1] += metrics(im_fs, im_us)[j] / test_batch

                    if opt.tensorboard and idx_slice % opt.savefreq == 0:
                        writer.add_figure(
                            f'{model}/{contrast}/MRI_{idx_mri}', 
                            plot_quadrant(im_fs, im_us),
                            global_step = idx_slice,
                        )
            
            # define x axis
            model = model.split('~') # 0 is STL_varnet_etc, 1 is N=_N=_etc
            ratio_1 = int(model[-1].split('_')[0].split('=')[1])
            df_row[idx_model, 4] = ratio_1

            if opt.mixeddata[idx_experimentname]:
                ratio_2 = int(model[-1].split('_')[1].split('=')[1])
                df_row[idx_model, 5] = ratio_2

    # fraction instead of absolute no. slices
    df_row[:, 4] /= opt.scarcemax
    if opt.mixeddata[idx_experimentname]:
        df_row[:, 5] /= np.max(df_row[:, 5])

    ### create csv file  
    df = pd.DataFrame(
        df_row,
        columns=columns
    )
    df = df.sort_values(by=[sort_by])
    df.to_csv(summary_file)

    return df

def _get_model_filedir(dataset, opt, idx_experimentname):
    # normally, we are in mixeddata case
    if opt.mixeddata[idx_experimentname]:
        # MTL (experimental)
        if opt.blockstructures[idx_experimentname] != 'S':
            model_filedir = os.path.join(
                opt.modeldir,
                f"{opt.experimentnames[idx_experimentname]}_" + \
                # f"{'strat_' if opt.stratified[idx_experimentname] else ''}" + \
                f"{opt.network[idx_experimentname]}" + \
                f"{opt.blockstructures[idx_experimentname]}_" +\
                f"{'_'.join(opt.datasets)}"
            )

        # STL mixed (control)
        else:
            model_filedir = os.path.join(
                opt.modeldir,
                f"{opt.experimentnames[idx_experimentname]}_" + \
                f"{opt.network[idx_experimentname]}_" + \
                f"{'_'.join(opt.datasets)}"
            )

    # only for STL where data are not mixed (control)
    else:
        model_filedir = os.path.join(
            opt.modeldir,
            f"{opt.experimentnames[idx_experimentname]}_" + \
            f"{opt.network[idx_experimentname]}_" + \
            f"{dataset}"
        )
    
    if not os.path.isdir(model_filedir):
        raise ValueError(f'{model_filedir} is not valid dir')

    return model_filedir

def _get_model_info(dataset, opt, idx_experimentname):

    # figure out where to load saved weights from
    model_filedir = _get_model_filedir(dataset, opt, idx_experimentname)


    ### figure out which model skeleton to use ###
    if 'STL' in opt.experimentnames[idx_experimentname]:
        with torch.no_grad():
            the_model = STL_VarNet(
                num_cascades = opt.numblocks[idx_experimentname],
                ).to(opt.device)
                
    elif 'MTL' in opt.experimentnames[idx_experimentname]:
        ### remember to change the inputs manually
        # v1
        if opt.backcompat[idx_experimentname]:
            with torch.no_grad():
                the_model = MTL_VarNet_backcompat(
                    datasets = opt.datasets,
                    num_cascades = opt.numblocks[idx_experimentname],
                    # begin_blocks = opt.beginblocks[idx_experimentname],
                    shared_blocks = opt.sharedblocks[idx_experimentname],
                    ).to(opt.device)
        # v3
        elif opt.backcompat[idx_experimentname]:
            with torch.no_grad():
                the_model = MTL_VarNet_backcompat(
                    datasets = opt.datasets,
                    blockstructures = interpret_blockstructures(
                        opt.blockstructures[idx_experimentname]
                        ),
                    share_etas = opt.shareetas[idx_experimentname],
                    ).to(opt.device)
    
        else:
            with torch.no_grad():
                the_model = MTL_VarNet(
                    datasets = opt.datasets,
                    blockstructures = interpret_blockstructures(
                        opt.blockstructures[idx_experimentname]
                        ),
                    share_etas = opt.shareetas[idx_experimentname],
                    ).to(opt.device)
    
    else:
        raise ValueError(f'{opt.experimentnames[idx_experimentname]} not valid')
    

    return the_model, model_filedir






def save_bokeh_plots(writer, opt):
    if opt.createplots:
        plots = _initialize_plots(opt)
        colormap = _initialize_colormap(opt)


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

        # iterate through each model (i.e. N = _) in the folder
        for idx_experimentname, experimentname in enumerate(opt.experimentnames): 

            # load model
            the_model, model_filedir = _get_model_info(
                dataset, opt, idx_experimentname
                )
            print(f'working on {dataset}, {model_filedir}') 

            # get df for all ratios of a particular model, for a single contrast
            df = df_single_contrast_all_models(
                the_model, test_dloader, model_filedir, dataset, idx_experimentname, writer
            )
            
            if opt.createplots:
                if opt.mixeddata[idx_experimentname]:
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
                        size = 6
                    )

                    plots[idx_plot].line(
                        source = df,
                        x = x_data,
                        y = opt.plotnames[idx_plot],
                        legend_label = f'{dataset} ~ {experimentname}',
                        line_color = colormap[f'{dataset} ~ {experimentname}'],
                        line_width = 2,
                    )

    # customize and save plots    
    if opt.createplots:
        if opt.showbest:
            plots = _plot_best(plots, opt)   
              
        plotdir = f"plots/{opt.plotdir}"
            
        if not os.path.isdir(plotdir):
            os.makedirs(plotdir)

        for j in range(4):
            title = f"{opt.plotnames[j]}; scarce {opt.datasets[0]}, abundant {opt.datasets[1]}"

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
log_dir = f"plots/{opt.plotdir}"

if opt.tensorboard:
    writer_tensorboard = SummaryWriter(
        log_dir = log_dir,
        max_queue = 20,
        flush_secs = 1,
    )
else:
    writer_tensorboard = None
    
save_bokeh_plots(writer_tensorboard, opt)