"""
=== losses/metrics, tensorboard utils, model wrappers=== 
"""
import os
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

def count_parameters(model):
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


def plot_quadrant(im_fs, im_us):
    fig = plt.figure()
    plt.imshow(_test_result(im_fs, im_us), cmap = 'gray', vmax = 2.5) # or normalize between 0-1
    plt.close(fig)
    return fig


def _param_dict(lr, epochs, numblocks, undersampling, center_fractions):
    params = {}
    params['lr'] = lr
    params['epochs'] = epochs
    params['number of blocks'] = numblocks
    
    for i in range(len(undersampling)):
        params[f'accerlation_{i}'] = undersampling[i]
    
    for i in range(len(center_fractions)):
        params[f'center_fraction_{i}'] = center_fractions[i]
        
    return params


def write_tensorboard(writer, cost, epoch, model, ratio, opt):
    if epoch == 0:
        writer.add_text(
            'parameters', 
            f'{count_parameters(model)} parameters'
        )

        ###opts###
        writer.add_hparams(
            _param_dict(
                opt.lr, opt.epochs, opt.numblocks, 
                opt.accelerations, opt.centerfracs
            ), 
            {'zdummy':0}
    )
    if epoch >= 2:
        if len(opt.datasets) == 1:
            write_tensorboard_one_contrasts(
                writer, cost, epoch, ratio, opt
            )
        else:
            write_tensorboard_two_contrasts(
                writer, cost, epoch, ratio, opt
            )


def write_tensorboard_two_contrasts(writer, cost, epoch, ratio, opt):
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
            f'val/{ratio}/{contrast_1}' : cost[contrast_1][4],
            f'val/{ratio}/{contrast_2}' : cost[contrast_2][4],
        }, 
        epoch
    )

    writer.add_scalars(
        'overall/ssim', {
            f'val/{ratio}/{contrast_1}' : cost[contrast_1][5],
            f'val/{ratio}/{contrast_2}' : cost[contrast_2][5],
        }, 
        epoch
    )

    writer.add_scalars(
        'overall/psnr', {
            f'val/{ratio}/{contrast_1}' : cost[contrast_1][6],
            f'val/{ratio}/{contrast_2}' : cost[contrast_2][6],
        }, 
        epoch
    )

    writer.add_scalars(
        'overall/nrmse', {
            f'val/{ratio}/{contrast_1}' : cost[contrast_1][7],
            f'val/{ratio}/{contrast_2}' : cost[contrast_2][7],
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






"""
=========== Universal Single-task Trainer =========== 
                    user facing
"""


def single_task_trainer(
    train_loader, val_loader,
    train_ratios, val_ratios,
    single_task_model, 
    device, writer, 
    optimizer, scheduler,
    opt
):
    # convenience
    contrast_count = len(opt.datasets)
    train_batch = len(train_loader)
    val_batch = len(val_loader)
    
    ratio = f"N={'_N='.join(str(key) for key in train_ratios.values())}"
    best_val_loss = np.infty
    
    for epoch in range(opt.epochs):
        # contains info for single epoch
        cost = {
            contrast : np.zeros(8)
            for contrast in opt.datasets
        }
        cost['overall'] = np.zeros(8)

        # train the data
        single_task_model.train()
        
        train_dataset = iter(train_loader)
        
        for kspace, mask, sens, im_fs, contrast in train_dataset:
            contrast = contrast[0] # torch dataset loader returns as tuple
            kspace, mask = kspace.to(device), mask.to(device)
            sens, im_fs = sens.to(device), im_fs.to(device)

            optimizer.zero_grad()
            _, im_us = single_task_model(kspace, mask, sens) # forward pass
            loss = criterion(im_fs, im_us)
            loss.backward()
            optimizer.step()
            
            # losses and metrics are averaged over epoch at the end 
            # L1 loss for now
            cost[contrast][0] += loss.item() 
            
            # ssim, psnr, nrmse
            ssim, psnr, nrmse = metrics(im_fs, im_us)
            for j in range(3):
                cost[contrast][j + 1] += metrics(im_fs, im_us)[j]

        
        # get losses and metrics for each epoch
        single_task_model.eval()
        with torch.no_grad():
        
            # validation data
            val_dataset = iter(val_loader)
            for val_idx, val_data in enumerate(val_dataset):
                kspace, mask, sens, im_fs, contrast = val_data
                contrast = contrast[0]
                kspace, mask = kspace.to(device), mask.to(device)
                sens, im_fs = sens.to(device), im_fs.to(device)

                _, im_us = single_task_model(kspace, mask, sens) # forward pass
                loss = criterion(im_fs, im_us)
                
                # losses and metrics are averaged over epoch at the end 
                # L1 loss for now
                cost[contrast][4] += loss.item() 

                # ssim, psnr, nrmse
                ssim, psnr, nrmse = metrics(im_fs, im_us)
                for j in range(3):
                    cost[contrast][j + 5] += metrics(im_fs, im_us)[j]
                
               # visualize reconstruction every few epochs
                if opt.tensorboard and epoch % opt.savefreq == 0: ###opt###
                    # if single contrast, only visualize 17th slice
                    if (
                        val_idx == 17 or 
                        val_idx == val_batch - 17 and contrast_count > 1
                    ):
                        writer.add_figure(
                            f'{ratio}/{contrast}', 
                            plot_quadrant(im_fs, im_us),
                            epoch, close = True,
                        )                    

        # update overall
        cost['overall'] = np.sum([cost[contrast] for contrast in opt.datasets], axis = 0)
        cost["overall"][:4] /= train_batch
        cost["overall"][4:] /= val_batch
        
        # average out
        for contrast in opt.datasets:
            cost[contrast][:4] /= train_ratios[contrast]
            cost[contrast][4:] /= val_ratios[contrast]
            

        
        # early stopping        
        if cost['overall'][4] < best_val_loss:
            best_val_loss = cost['overall'][4]
            filedir = f"models/{opt.experimentname}_{opt.network}_{'_'.join(opt.datasets)}"
            if not os.path.isdir(filedir):
                os.makedirs(filedir)
            torch.save(
                single_task_model.state_dict(), 
                os.path.join(filedir, f'{ratio}_l1.pt'),
            )
            
        scheduler.step()
        
        if opt.verbose:
            print(f'''
            >Epoch: {epoch + 1:04d}
            TRAIN: loss {cost['overall'][0]:.4f} | ssim {cost['overall'][1]:.4f} | psnr {cost['overall'][2]:.4f} | nrmse {cost['overall'][3]:.4f} 
            VAL: loss {cost['overall'][4]:.4f} | ssim {cost['overall'][5]:.4f} | psnr {cost['overall'][6]:.4f} | nrmse {cost['overall'][7]:.4f}

            ''')
    
        # write to tensorboard
        ###opt###
        if opt.tensorboard:
            write_tensorboard(writer, cost, epoch, single_task_model, ratio, opt)  
    