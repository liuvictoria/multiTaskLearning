"""Docstring for utils.py

Contains frequently-used losses/metrics, tensorboard utils, dictionaries
"""

import numpy as np
import skimage.metrics

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

import torch
import torch.nn as nn

from fastmri.data import transforms
import sigpy as sp
from sigpy import from_pytorch

import unet


"""
=========== Losses and Metrics =========== 
                user facing
"""

def criterion(im_fs: torch.Tensor, im_us: torch.Tensor):
    """l1 loss
    
    Parameters
    ----------
    im_fs : tensor
        fully sampled image (2D)
    im_us : tensor
        undersampled image (2D)
    
    Returns
    -------
    scalar
        l1 loss, as implemented by nn.L1Loss

    """  
    # use l1 loss between two images
    criterion = nn.L1Loss()
    
    # can add more fancy loss functions here later
    return criterion(im_us, im_fs)

def metrics(im_fs: torch.Tensor, im_us: torch.Tensor):
    """SSIM, pSNR, nRMSE on magnitude images.

    Normalization between 0 and 1 for SSIM only.

    Parameters
    ----------
    im_fs : tensor
        fully sampled image (2D)
    im_us : tensor
        undersampled image (2D)
    
    Returns
    -------
    ssim : scalar
    psnr : scalar
    nrmse : scalar

    See Also : 
    skimage.metrics

    """

    # change to ndarray
    im_us = transforms.tensor_to_complex_np(im_us.cpu().detach())
    im_fs = transforms.tensor_to_complex_np(im_fs.cpu().detach())
    
    # convert complex nums to magnitude
    im_us = np.absolute(im_us)
    im_fs = np.absolute(im_fs)
    
    im_us = im_us.reshape(
        (im_us.shape[2], im_us.shape[3])
    )
    
    im_fs = im_fs.reshape(
        (im_fs.shape[2], im_fs.shape[3])
    )
    
    # psnr
    psnr = skimage.metrics.peak_signal_noise_ratio(
        im_fs, 
        im_us, 
        data_range = np.max(im_fs) - np.min(im_fs)
    )
    
    #nrmse
    nrmse = skimage.metrics.normalized_root_mse(im_fs, im_us)
    
    # ssim
    # normalize 0 to 1
    im_fs -= np.min(im_fs)
    im_fs /= np.max(im_fs)
    im_us -= np.min(im_us)
    im_us /= np.max(im_us)
    
    ssim = skimage.metrics.structural_similarity(im_fs, im_us, data_range = 1)
    
    return ssim, psnr, nrmse

"""
=========== Misc ============= 
    naming for block structure (used in evaluate and MTL_VarNet)
"""

def label_blockstructures(blockstructures):
    """Conversion between long-hand and short-hand
    for MTL block structures.
    """

    conversion = {
        'trueshare' : 'I',
        'mhushare' : 'Y',
        'split' : 'V',
        'attenshare' : 'W',
    }
    labels = []
    for blockstructure in blockstructures:
        labels.append(conversion[blockstructure])
    return ''.join(labels)

def interpret_blockstructures(blockstructures):
    """Conversion between short-hand and long-hand
    for MTL block structures.
    """

    conversion = {
        'I' : 'trueshare',
        'Y' : 'mhushare',
        'V' : 'split',
        'W' : 'attenshare',
     }
    labels = []
    for blockstructure in blockstructures:
        labels.append(conversion[blockstructure])
    return labels


"""
=========== Custom Tensorboard ============= 
    utilities involved in plotting losses
    and metrics during training and validation
"""

def _count_parameters(model):
    """Count the number of trainable parameters in the network
    """
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

def _test_result(im_fs: torch.Tensor, im_us: torch.Tensor) -> np.ndarray:
    """Prepares MR images for viewing.

    - Crops MR images to 360 x 320.
    - Creates quadrant structure:
        network recon | ground truth
        --------------|------------------
        error map     | error map (black)

    """

    with torch.no_grad():
        im_us = from_pytorch(im_us.cpu().detach(),iscomplex = True)
        im_fs = from_pytorch(im_fs.cpu().detach(), iscomplex = True)
        im_us = np.abs(im_us).squeeze()
        im_fs = np.abs(im_fs).squeeze()
        
        im_us = sp.resize(im_us, [360, 320])
        im_fs = sp.resize(im_fs, [360, 320])
        
        out_cat = np.concatenate((im_fs, im_us), 1)
        error_cat = np.concatenate((im_fs, im_fs), 1)
        error_cat = np.abs(error_cat - out_cat) * 5
        
        out_cat = np.concatenate((error_cat, out_cat,), axis=0)
        out_cat = out_cat * 1.5  
        
    return np.flip(out_cat)


def plot_quadrant(im_fs: torch.Tensor, im_us: torch.Tensor):
    """Plot the reconstructed image and ground truth in matplotlib

    Parameters
    ----------
    im_fs : tensor
        fully sampled image (2D)
    im_us : tensor
        undersampled image (2D)
    
    Returns
    -------
    fig : plt.figure()
        matplotlib figure ready for rendering

    """
    fig = plt.figure()
    plt.imshow(_test_result(im_fs, im_us), cmap = 'gray', vmax = 2.5) # or normalize between 0-1
    plt.close(fig)
    return fig





def write_tensorboard(
    writer, cost, iteration, epoch, model, ratio, opt, weights = None
    ):

    """Tensorboard writer for one or two tasks

    Parameters
    writer : tensorboard SummaryWriter
        contains directory for tensorboard logs
    cost : dict
        each key is a task, and each value is an array of length 8
        containing loss / metrics information for training and validation
    iteration : int
        iteration in training; i.e. counts each forward pass
    epoch : int
        epoch of training; i.e. each time there is a full pass of all data
    model : model-like object
        Weights are loaded
    ratio : str
        i.e. N=32_N=481 for task ratios
    opt : argparse ArgumentParser
        Contains user-defined parameters.
        See documentation in stl.py or mtl.py for more information.
    weights : dict, default = None 
        None implies STL; keys are tasks, values are the task-weight for loss

    Returns
    -------
    None

    Notes
    -----
    Future versions will allow for writing for more than two tasks.
    """
    
    if epoch == 0:
        writer.add_text(
            'parameters', 
            f'{_count_parameters(model)} parameters'
        )
            
    if epoch >= 2:
        if len(opt.datasets) == 1:
            _write_tensorboard_one_tasks(
                writer, cost, iteration, ratio, opt
            )
        else:
            _write_tensorboard_two_tasks(
                writer, cost, iteration, ratio, opt, weights = weights
            )


def _write_tensorboard_two_tasks(writer, cost, iteration, ratio, opt, weights):
    """Private function that write_tensorboard calls.

    Writers to tensorboard for two tasks.
    """

    task_1, task_2 = opt.datasets

    writer.add_scalars(
        f'{ratio}/l1', {
            f'train/{task_1}' : cost[task_1][0],
            f'val/{task_1}' : cost[task_1][4],
            f'train/{task_2}' : cost[task_2][0],
            f'val/{task_2}' : cost[task_2][4],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/ssim', {
            f'train/{task_1}' : cost[task_1][1],
            f'val/{task_1}' : cost[task_1][5],
            f'train/{task_2}' : cost[task_2][1],
            f'val/{task_2}' : cost[task_2][5],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/psnr', {
            f'train/{task_1}' : cost[task_1][2],
            f'val/{task_1}' : cost[task_1][6],
            f'train/{task_2}' : cost[task_2][2],
            f'val/{task_2}' : cost[task_2][6],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/nrmse', {
            f'train/{task_1}' : cost[task_1][3],
            f'val/{task_1}' : cost[task_1][7],
            f'train/{task_2}' : cost[task_2][3],
            f'val/{task_2}' : cost[task_2][7],
        }, 
        iteration
    )

    writer.add_scalars(
        'overall/l1', {
            f'train/{ratio}' : cost['overall'][0],
            f'val/{ratio}' : cost['overall'][4],
        }, 
        iteration
    )

    writer.add_scalars(
        'overall/ssim', {
            f'train/{ratio}' : cost['overall'][1],
            f'val/{ratio}' : cost['overall'][5],
        }, 
        iteration
    )

    writer.add_scalars(
        'overall/psnr', {
            f'train/{ratio}' : cost['overall'][2],
            f'val/{ratio}' : cost['overall'][6],
        }, 
        iteration
    )

    writer.add_scalars(
        'overall/nrmse', {
            f'train/{ratio}' : cost['overall'][3],
            f'val/{ratio}' : cost['overall'][7],
        }, 
        iteration
    )

    if weights is not None:
        writer.add_scalars(
            f'{ratio}/weighting', {
                f'{task_1}' : weights[task_1],
                f'{task_2}' : weights[task_2],
            }, 
            iteration
        )

    
def _write_tensorboard_one_tasks(writer, cost, iteration, ratio, opt):
    """Private function that write_tensorboard calls.

    Writers to tensorboard for one task.
    """

    task_1 = opt.datasets[0]
        
    writer.add_scalars(
        f'{ratio}/l1', {
            f'train/{task_1}' : cost[task_1][0],
            f'val/{task_1}' : cost[task_1][4],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/ssim', {
            f'train/{task_1}' : cost[task_1][1],
            f'val/{task_1}' : cost[task_1][5],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/psnr', {
            f'train/{task_1}' : cost[task_1][2],
            f'val/{task_1}' : cost[task_1][6],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/nrmse', {
            f'train/{task_1}' : cost[task_1][3],
            f'val/{task_1}' : cost[task_1][7],
        }, 
        iteration
    )






