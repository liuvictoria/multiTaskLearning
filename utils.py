"""
=== losses/metrics, tensorboard utils, model wrappers=== 
"""
import numpy as np

import torch
import torch.nn as nn

from fastmri.data import transforms

import sigpy as sp
from sigpy import from_pytorch

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
import skimage.metrics



"""
=========== Losses and Metrics =========== 
                user facing
"""

def criterion(im_fs: torch.Tensor, im_us: torch.Tensor):
    '''
    @parameter im_us: undersampled image (2D)
    @parameter im_fs: fully sampled image (2D)
    should be on GPU device for fast computation
    '''  
    # use l1 loss between two images
    criterion = nn.L1Loss()
    
    # can add more fancy loss functions here later
    return criterion(im_us, im_fs)

def metrics(im_fs: torch.Tensor, im_us: torch.Tensor):
    '''
    @parameter im_us: undersampled image (2D)
    @parameter im_fs: fully sampled image (2D)
    should be on GPU device for fast computation
    '''

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
=========== customized TensorBoard ============= 
      plotting and tensorboard user facing;
   functions beginning with _ are helper funcs
"""

def _count_parameters(model):
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

def _test_result(im_fs: torch.Tensor, im_us: torch.Tensor) -> np.ndarray:

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
    fig = plt.figure()
    plt.imshow(_test_result(im_fs, im_us), cmap = 'gray', vmax = 2.5) # or normalize between 0-1
    plt.close(fig)
    return fig





def write_tensorboard(writer, cost, epoch, model, ratio, opt, weights = None):
    '''
    weights = None implies STL; weights should be dict of weights
    '''
    if epoch == 0:
        writer.add_text(
            'parameters', 
            f'{_count_parameters(model)} parameters'
        )
            
    if epoch >= 2:
        if len(opt.datasets) == 1:
            write_tensorboard_one_contrasts(
                writer, cost, epoch, ratio, opt
            )
        else:
            write_tensorboard_two_contrasts(
                writer, cost, epoch, ratio, opt, weights = weights
            )


def write_tensorboard_two_contrasts(writer, cost, epoch, ratio, opt, weights):
    # write to tensorboard ###opt###
    contrast_1, contrast_2 = opt.datasets
    # for display purposes
    epoch += 1 

    writer.add_scalars(
        f'{ratio}/l1', {
            f'train/{contrast_1}' : cost[contrast_1][0],
            f'val/{contrast_1}' : cost[contrast_1][4],
            f'train/{contrast_2}' : cost[contrast_2][0],
            f'val/{contrast_2}' : cost[contrast_2][4],
        }, 
        epoch
    )

    writer.add_scalars(
        f'{ratio}/ssim', {
            f'train/{contrast_1}' : cost[contrast_1][1],
            f'val/{contrast_1}' : cost[contrast_1][5],
            f'train/{contrast_2}' : cost[contrast_2][1],
            f'val/{contrast_2}' : cost[contrast_2][5],
        }, 
        epoch
    )

    writer.add_scalars(
        f'{ratio}/psnr', {
            f'train/{contrast_1}' : cost[contrast_1][2],
            f'val/{contrast_1}' : cost[contrast_1][6],
            f'train/{contrast_2}' : cost[contrast_2][2],
            f'val/{contrast_2}' : cost[contrast_2][6],
        }, 
        epoch
    )

    writer.add_scalars(
        f'{ratio}/nrmse', {
            f'train/{contrast_1}' : cost[contrast_1][3],
            f'val/{contrast_1}' : cost[contrast_1][7],
            f'train/{contrast_2}' : cost[contrast_2][3],
            f'val/{contrast_2}' : cost[contrast_2][7],
        }, 
        epoch
    )

    writer.add_scalars(
        'overall/l1', {
            f'train/{ratio}' : cost['overall'][0],
            f'val/{ratio}' : cost['overall'][4],
        }, 
        epoch
    )

    writer.add_scalars(
        'overall/ssim', {
            f'train/{ratio}' : cost['overall'][1],
            f'val/{ratio}' : cost['overall'][5],
        }, 
        epoch
    )

    writer.add_scalars(
        'overall/psnr', {
            f'train/{ratio}' : cost['overall'][2],
            f'val/{ratio}' : cost['overall'][6],
        }, 
        epoch
    )

    writer.add_scalars(
        'overall/nrmse', {
            f'train/{ratio}' : cost['overall'][3],
            f'val/{ratio}' : cost['overall'][7],
        }, 
        epoch
    )

    if weights is not None:
        writer.add_scalars(
            f'{ratio}/weighting', {
                f'{contrast_1}' : weights[contrast_1],
                f'{contrast_2}' : weights[contrast_2],
            }, 
            epoch
        )

    
def write_tensorboard_one_contrasts(writer, cost, epoch, ratio, opt):
    #write to tensorboard ###opt###
    contrast_1 = opt.datasets[0]
    epoch += 1
        
    writer.add_scalars(
        f'{ratio}/l1', {
            f'train/{contrast_1}' : cost[contrast_1][0],
            f'val/{contrast_1}' : cost[contrast_1][4],
        }, 
        epoch
    )

    writer.add_scalars(
        f'{ratio}/ssim', {
            f'train/{contrast_1}' : cost[contrast_1][1],
            f'val/{contrast_1}' : cost[contrast_1][5],
        }, 
        epoch
    )

    writer.add_scalars(
        f'{ratio}/psnr', {
            f'train/{contrast_1}' : cost[contrast_1][2],
            f'val/{contrast_1}' : cost[contrast_1][6],
        }, 
        epoch
    )

    writer.add_scalars(
        f'{ratio}/nrmse', {
            f'train/{contrast_1}' : cost[contrast_1][3],
            f'val/{contrast_1}' : cost[contrast_1][7],
        }, 
        epoch
    )






