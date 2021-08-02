import os
import copy
import numpy as np
import torch

from fastmri.data import transforms

from utils import criterion, metrics
from utils import label_blockstructures, plot_quadrant, write_tensorboard


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

        for kspace, mask, esp, im_fs, contrast in train_dataset:
            contrast = contrast[0] # torch dataset loader returns as tuple
            kspace, mask = kspace.to(device), mask.to(device)
            esp, im_fs = esp.to(device), im_fs.to(device)

            optimizer.zero_grad()

            _, im_us = single_task_model(kspace, mask, esp) # forward pass
            # crop so im_us has same size as im_fs
            im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
            loss = criterion(im_fs, im_us)
            loss.backward()
            optimizer.step()

            # losses and metrics are averaged over epoch at the end
            # L1 loss for now
            cost[contrast][0] += loss.item()

            # ssim, psnr, nrmse
            for j in range(3):
                cost[contrast][j + 1] += metrics(im_fs, im_us)[j]


        # get losses and metrics for each epoch
        single_task_model.eval()
        with torch.no_grad():

            # validation data
            val_dataset = iter(val_loader)
            for val_idx, val_data in enumerate(val_dataset):
                kspace, mask, esp, im_fs, contrast = val_data
                contrast = contrast[0]
                kspace, mask = kspace.to(device), mask.to(device)
                esp, im_fs = esp.to(device), im_fs.to(device)

                _, im_us = single_task_model(kspace, mask, esp) # forward pass
                # crop so im_us has same size as im_fs
                im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
                loss = criterion(im_fs, im_us)

                # losses and metrics are averaged over epoch at the end
                # L1 loss for now
                cost[contrast][4] += loss.item()

                # ssim, psnr, nrmse
                for j in range(3):
                    cost[contrast][j + 5] += metrics(im_fs, im_us)[j]

               # visualize reconstruction every few epochs
                if opt.tensorboard and epoch % opt.savefreq == 0: ###opt###
                    # if single contrast, only visualize 17th slice
                    if (
                        val_idx == 17 and contrast == opt.bothdatasets[0] or
                        val_idx == val_batch - 17 and contrast == opt.bothdatasets[1]
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





"""
=========== Multi-task utilities ===========
        used by Universal MT Trainer
"""

def _get_naive_weights(train_ratios, opt):
    if opt.stratified:
        return {
            contrast : 1
            for contrast in train_ratios.keys()
        }
    total_slices = sum(train_ratios.values())

    # balance weights if not stratified; smaller datasets get larger weights
    naive_weights = {
        contrast : len(train_ratios) * (total_slices - train_ratios[contrast]) / total_slices
        for contrast in train_ratios.keys()
    }
    return naive_weights


"""
=========== Universal Multi-task Trainer ===========
                    user facing
"""


def multi_task_trainer(
    train_loader, val_loader,
    train_ratios, val_ratios,
    multi_task_model,
    device, writer,
    optimizer, scheduler,
    opt
):
    
    # naming (even if stratified, scarce / abundant ratio is preserved)
    ratio = f"N={'_N='.join(str(key) for key in train_ratios.values())}"

    # convenience
    contrast_count = len(opt.datasets)
    train_batch = len(train_loader)
    val_batch = len(val_loader)

    # naive weighting (accounts for stratified or not)
    weights = _get_naive_weights(train_ratios, opt)

    # for saving best validation model
    best_val_loss = np.infty

    # grad accumulation
    iteration = 0
    contrast_batches = [0 for _ in range(len(opt.datasets))] 

    for epoch in range(opt.epochs):

        if opt.weighting == 'dwa':
            # contains info for last two epochs; must be ex'd in this order
            if epoch > 1:
                cost_prevprev = copy.deepcopy(cost_prev)
            if epoch > 0:
                cost_prev = copy.deepcopy(cost)

            # get dwa weights
            if epoch > 1:
                w_0 = cost_prev[opt.datasets[0]][0] / cost_prevprev[opt.datasets[0]][0]
                w_1 = cost_prev[opt.datasets[1]][0] / cost_prevprev[opt.datasets[1]][0]

                weights[opt.datasets[0]] = contrast_count * np.exp(w_0 / opt.temp) / (
                    np.exp(w_0 / opt.temp) + np.exp(w_1 / opt.temp)
                    )
                weights[opt.datasets[1]] = contrast_count * np.exp(w_1 / opt.temp) / (
                    np.exp(w_0 / opt.temp) + np.exp(w_1 / opt.temp)
                    )              

        # contains info for current epoch:
        cost = {
            contrast : np.zeros(8)
            for contrast in opt.datasets
        }
        cost['overall'] = np.zeros(8)

        # train the data
        multi_task_model.train()

        train_dataset = iter(train_loader)

        for kspace, mask, esp, im_fs, contrast in train_dataset:
            contrast = contrast[0] # torch dataset loader returns as tuple
            kspace, mask = kspace.to(device), mask.to(device)
            esp, im_fs = esp.to(device), im_fs.to(device)

            

            # grad accumulation 
            iteration += 1
            print (f'iteration {iteration}')
            contrast_batches[opt.datasets.index(contrast)] += 1

            optimizer.zero_grad()
            # create hooks before last batch in accumulation
            if sum(contrast_batches) == opt.gradaccumulation:
                # forward
                _, im_us, logsigma = multi_task_model(
                    kspace, mask, esp, contrast,
                    contrast_batches = contrast_batches, create_hooks = True,
                    )
            else:
                # forward
                _, im_us, logsigma = multi_task_model(
                    kspace, mask, esp, contrast,
                    contrast_batches = contrast_batches, create_hooks = False,
                    )

            # crop so im_us has same size as im_fs
            im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))

            # loss
            if opt.weighting == 'naive' or opt.weighting == 'dwa':
                loss = weights[contrast] * criterion(im_fs, im_us)

            elif opt.weighting == 'uncert':
                idx_contrast = opt.datasets.index(contrast)
                loss = 1 / (2 * torch.exp(logsigma[idx_contrast])) * \
                    criterion(im_fs, im_us) + \
                        logsigma[idx_contrast] / 2
                # for plotting purposes
                weights[contrast] = logsigma[idx_contrast]

            loss.backward()
            
            # step optimizer once we've reached the right no. batches
            if sum(contrast_batches) == opt.gradaccumulation:      
                optimizer.step()
                # reset contrast batches
                contrast_batches = [0 for _ in range(len(opt.datasets))] 

            # losses and metrics are averaged over epoch at the end
            # L1 loss for now
            cost[contrast][0] += loss.item()
            # ssim, psnr, nrmse
            for j in range(3):
                cost[contrast][j + 1] += metrics(im_fs, im_us)[j]


        # validation
        multi_task_model.eval()
        with torch.no_grad():

            # validation data
            val_dataset = iter(val_loader)
            for val_idx, val_data in enumerate(val_dataset):
                kspace, mask, esp, im_fs, contrast = val_data
                contrast = contrast[0]
                kspace, mask = kspace.to(device), mask.to(device)
                esp, im_fs = esp.to(device), im_fs.to(device)

                _, im_us, logsigma = multi_task_model(
                    kspace, mask, esp, contrast,
                    contrast_batches = [1],
                    create_hooks = False,
                    ) # forward pass
                # crop so im_us has same size as im_fs
                im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
                loss = criterion(im_fs, im_us)

                # losses and metrics are averaged over epoch at the end
                # L1 loss for now
                cost[contrast][4] += loss.item() 

                # ssim, psnr, nrmse
                for j in range(3):
                    cost[contrast][j + 5] += metrics(im_fs, im_us)[j]

               # visualize reconstruction every few epochs
                if opt.tensorboard and epoch % opt.savefreq == 0: ###opt###
                    # if single contrast, only visualize 17th slice
                    if (val_idx == 17 or val_idx == val_batch - 17):
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
            if opt.stratified:
                # scarce / abundant have same effective sizes
                cost[contrast][:4] /= train_batch / len(opt.datasets)
            else:
                cost[contrast][:4] /= train_ratios[contrast]
            cost[contrast][4:] /= val_ratios[contrast]



        # early stopping
        if cost['overall'][4] < best_val_loss:
            best_val_loss = cost['overall'][4]
            filedir = f"models/{opt.experimentname}_" + \
                        f"{'strat_' if opt.stratified else ''}" + \
                        f"{opt.network}{label_blockstructures(opt.blockstructures)}_{'_'.join(opt.datasets)}/"
            if not os.path.isdir(filedir):
                os.makedirs(filedir)
            torch.save(
                multi_task_model.state_dict(),
                os.path.join(filedir, f'{ratio}_l1.pt'),
            )

        scheduler.step()

        if opt.verbose:
            print(f'''
            >EPOCH (full run-throughs of abundant dataset): {epoch + 1:04d}; ITERATION: {iteration}
            TRAIN: loss {cost['overall'][0]:.4f} | ssim {cost['overall'][1]:.4f} | psnr {cost['overall'][2]:.4f} | nrmse {cost['overall'][3]:.4f}
            VAL: loss {cost['overall'][4]:.4f} | ssim {cost['overall'][5]:.4f} | psnr {cost['overall'][6]:.4f} | nrmse {cost['overall'][7]:.4f}

            ''')

        # write to tensorboard
        ###opt###
        if opt.tensorboard:
            write_tensorboard(writer, cost, iteration, multi_task_model, ratio, opt, weights)
