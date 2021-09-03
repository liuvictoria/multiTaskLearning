"""
=== losses/metrics, tensorboard utils, model wrappers=== 
"""
import numpy as np

import torch
import torch.nn as nn

import fastmri_unet

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
=========== Hook (RETIRED) ============= 
    hooks for gradient accumulation
    involved with dividing gradient of
    split / shared layers properly
"""
class Module_Hook():
    def __init__(
        self, 
        module: nn.Module, 
        name: str = None,
        accumulated_by: int = None,
        ):
        
        assert type(accumulated_by) == int, 'accumulated_by must be an int, not None type'
        if accumulated_by == 0:
            accumulated_by = 1
        accumulated_by = float(accumulated_by)
        self.accumulated_by = accumulated_by
        self.name = name

        self.hook = module.register_full_backward_hook(self.hook_fn)
        

    def hook_fn(self, module, input, output):
        # print (f' using module {self.name} hook divided by {self.accumulated_by}')
        if type(input[0]) == torch.Tensor:
            return tuple(tensor / self.accumulated_by for tensor in input)
  

    def close(self):
        self.hook.remove()


class Tensor_Hook():
    def __init__(
        self, 
        tensor: torch.Tensor,
        name: str = None, 
        accumulated_by: int = None,
        ):
        assert type(accumulated_by) == int, 'accumulated_by must be an int, not None type'
        if accumulated_by == 0:
            accumulated_by = 1

        accumulated_by = float(accumulated_by)
        self.accumulated_by = 1
        self.hook = tensor.register_hook(self.hook_fn)
        self.name = name
        tensor.retain_grad()

    def hook_fn(self, tensor):
        print (f' using tensor hook {self.name}, divided by {self.accumulated_by}')
        return tensor / self.accumulated_by

    def close(self):
        self.hook.remove()

def configure_hooks(model, contrast_batches):
    '''
    full backward hooks for gradient accumulation
    creates hooks at all levels:
        MTL_VarNet (uncert)
        VarNetBlockMTL (etas)
        Unet levels (shared encoder vs split/shared decoder)
    '''
    # register hooks for accumulated gradient
    hooks = []
   
    # uncertainty at MTL_VarNet level
    hooks.extend([
        Tensor_Hook(
            logsigma, 
            name = f'uncert hook {idx_contrast}',
            accumulated_by = contrast_batches[idx_contrast]
            )
        for idx_contrast, logsigma in enumerate(model.logsigmas)
    ])
    
    # etas
    for idx_block, block in enumerate(model.allblocks):  
        hooks.extend([
            Tensor_Hook(
                eta, 
                name = f'eta_{idx_block}', 
                accumulated_by = sum(contrast_batches) if model.share_etas else contrast_batches[idx_contrast], 
                )
            for idx_contrast, eta in enumerate(block.etas)
        ])
    
    # encoder / decoders

    for name, module in model.named_modules():
        
        # initialize create_hook bool
        create_hook = False

        if type(module) == fastmri_unet.ConvBlock or type(module) == fastmri_unet.TransposeConvBlock:
            create_hook = True

        if create_hook:
            if 'logsigmas' in name or 'etas' in name:
                # we've alredy done these hooks using tensorhooks. Don't go to ModuleHook
                continue

            elif 'splitblock' in name:
                # '.splitblock_{int_contrast}.'
                int_contrast = int(name.split('splitblock_')[1].split('.')[0])
                accumulated_by = contrast_batches[int_contrast]
                
            elif 'splitdecoder' in name:
                # '_{int_contrast}_splitdecoder.'
                int_contrast = int(name.split('_splitdecoder')[0].split('_')[-1])
                accumulated_by = contrast_batches[int_contrast]

            elif 'shared' in name:
                accumulated_by = sum(contrast_batches)
        
            else:
                raise ValueError (f'unrecognized name {name}')

            hooks.extend([
                Module_Hook(
                    module,
                    f'{name}',
                    accumulated_by = accumulated_by,
                )
            ])    
        


    return hooks    

"""
=========== Misc ============= 
    naming for block structure (used in evaluate and MTL_VarNet)
"""
def label_blockstructures(blockstructures):
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





def write_tensorboard(writer, cost, iteration, epoch, model, ratio, opt, weights = None):
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
                writer, cost, iteration, ratio, opt
            )
        else:
            write_tensorboard_two_contrasts(
                writer, cost, iteration, ratio, opt, weights = weights
            )


def write_tensorboard_two_contrasts(writer, cost, iteration, ratio, opt, weights):
    # write to tensorboard ###opt###
    contrast_1, contrast_2 = opt.datasets

    writer.add_scalars(
        f'{ratio}/l1', {
            f'train/{contrast_1}' : cost[contrast_1][0],
            f'val/{contrast_1}' : cost[contrast_1][4],
            f'train/{contrast_2}' : cost[contrast_2][0],
            f'val/{contrast_2}' : cost[contrast_2][4],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/ssim', {
            f'train/{contrast_1}' : cost[contrast_1][1],
            f'val/{contrast_1}' : cost[contrast_1][5],
            f'train/{contrast_2}' : cost[contrast_2][1],
            f'val/{contrast_2}' : cost[contrast_2][5],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/psnr', {
            f'train/{contrast_1}' : cost[contrast_1][2],
            f'val/{contrast_1}' : cost[contrast_1][6],
            f'train/{contrast_2}' : cost[contrast_2][2],
            f'val/{contrast_2}' : cost[contrast_2][6],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/nrmse', {
            f'train/{contrast_1}' : cost[contrast_1][3],
            f'val/{contrast_1}' : cost[contrast_1][7],
            f'train/{contrast_2}' : cost[contrast_2][3],
            f'val/{contrast_2}' : cost[contrast_2][7],
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
                f'{contrast_1}' : weights[contrast_1],
                f'{contrast_2}' : weights[contrast_2],
            }, 
            iteration
        )

    
def write_tensorboard_one_contrasts(writer, cost, iteration, ratio, opt):
    #write to tensorboard ###opt###
    contrast_1 = opt.datasets[0]
    epoch += 1
        
    writer.add_scalars(
        f'{ratio}/l1', {
            f'train/{contrast_1}' : cost[contrast_1][0],
            f'val/{contrast_1}' : cost[contrast_1][4],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/ssim', {
            f'train/{contrast_1}' : cost[contrast_1][1],
            f'val/{contrast_1}' : cost[contrast_1][5],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/psnr', {
            f'train/{contrast_1}' : cost[contrast_1][2],
            f'val/{contrast_1}' : cost[contrast_1][6],
        }, 
        iteration
    )

    writer.add_scalars(
        f'{ratio}/nrmse', {
            f'train/{contrast_1}' : cost[contrast_1][3],
            f'val/{contrast_1}' : cost[contrast_1][7],
        }, 
        iteration
    )






