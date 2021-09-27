"""Docstring for wrappers.py

STL and MTL training wrappers. Takes care of:
- feeding data to network
- gradient accumulation (optional), back-prop, and weight update
- calculating metrics
- drawing validation images to Tensorboard

"""
import os
import copy
import numpy as np
import torch
import torch.nn as nn


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
    """Wrapper for single task learning training.

    Takes care of:
    - saving best validation model
    - forward pass and backward propagation
    - gradient accumulation / averaging, is specified in opt
    - calculating and recording metrics
    - plotting validation iamges
    - displaying averaged metrics on terminal

    Called in stl.py

    Parameters
    ----------
    train_loader : genDataLoader
        Contains slices from Train dataset. Slices shuffled.
    val_loader : genDataLoader
        Contains slices from Val dataset. Slices not shuffled.
    train_ratios : dict
        Keys are tasks. Values are the number of slices for each task.
        Even tho this is STL, we keep track of tasks for calculating
        individual task metrics. The loss func is not affected by task.
    val_ratios : dict
        Keys are tasks. Values are the number of slices for each task
    single_task_model : model-like object
        Weights are not loaded
    device : str
        Name of single GPU
    writer : tensorboard SummaryWriter
        Contains log directory information for tensorboard
    optimizer : torch.optim.<optimizer>
        usually Adam
    scheduler : torch.optim.<scheduler>
        usually StepLR
    opt : argparse.ArgumentParser
        Refer to help documentation in stl.py

    Returns
    -------
    None

    """
    # convenience
    task_count = len(opt.datasets)
    train_batch = len(train_loader)
    val_batch = len(val_loader)

    ratio = f"N={'_N='.join(str(key) for key in train_ratios.values())}"
    best_val_loss = np.infty

    # grad accumulation
    iteration = 0
    batch_count = 0

    for epoch in range(opt.epochs):
        # contains info for single epoch
        cost = {
            task : np.zeros(8)
            for task in opt.datasets
        }
        cost['overall'] = np.zeros(8)

        # train the data
        single_task_model.train()
        train_dataset = iter(train_loader)

        # grad accumulation
        optimizer.zero_grad() 

        for kspace, mask, esp, im_fs, task in train_dataset:
            task = task[0] # torch dataset loader returns as tuple
            kspace, mask = kspace.to(device), mask.to(device)
            esp, im_fs = esp.to(device), im_fs.to(device)

            # grad accumulation 
            iteration += 1
            batch_count += 1

            _, im_us = single_task_model(kspace, mask, esp) # forward pass
            # crop so im_us has same size as im_fs
            im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
            loss = criterion(im_fs, im_us)

            # grad accumulation
            if opt.gradaverage:
                loss /= opt.gradaccumulation

            loss.backward()

            # step optimizer once we've reached the right no. batches
            if batch_count == opt.gradaccumulation:
                optimizer.step()
                optimizer.zero_grad()
                
                # reset task batches
                batch_count = 0 

            # losses and metrics are averaged over epoch at the end
            # L1 loss for now
            cost[task][0] += loss.item()

            # ssim, psnr, nrmse
            for j in range(3):
                cost[task][j + 1] += metrics(im_fs, im_us)[j]


        # get losses and metrics for each epoch
        single_task_model.eval()
        with torch.no_grad():

            # validation data
            val_dataset = iter(val_loader)
            for val_idx, val_data in enumerate(val_dataset):
                kspace, mask, esp, im_fs, task = val_data
                task = task[0]
                kspace, mask = kspace.to(device), mask.to(device)
                esp, im_fs = esp.to(device), im_fs.to(device)

                _, im_us = single_task_model(kspace, mask, esp) # forward pass
                # crop so im_us has same size as im_fs
                im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
                loss = criterion(im_fs, im_us)

                # losses and metrics are averaged over epoch at the end
                # L1 loss for now
                cost[task][4] += loss.item()

                # ssim, psnr, nrmse
                for j in range(3):
                    cost[task][j + 5] += metrics(im_fs, im_us)[j]

               # visualize reconstruction every few epochs
                if opt.tensorboard and epoch % opt.savefreq == 0: ###opt###
                    # if single task, only visualize 17th slice
                    if (
                        val_idx == 17 and task == opt.bothdatasets[0] or
                        val_idx == val_batch - 17 and task == opt.bothdatasets[1]
                    ):
                        writer.add_figure(
                            f'{ratio}/{task}',
                            plot_quadrant(im_fs, im_us),
                            epoch, close = True,
                        )

        # update overall
        cost['overall'] = np.sum([cost[task] for task in opt.datasets], axis = 0)
        cost["overall"][:4] /= train_batch
        cost["overall"][4:] /= val_batch

        # average out
        for task in opt.datasets:
            cost[task][:4] /= train_ratios[task]
            cost[task][4:] /= val_ratios[task]



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
            print(f"""
            >Epoch: {epoch + 1:04d}
            TRAIN: loss {cost['overall'][0]:.4f} | ssim {cost['overall'][1]:.4f} | psnr {cost['overall'][2]:.4f} | nrmse {cost['overall'][3]:.4f}
            VAL: loss {cost['overall'][4]:.4f} | ssim {cost['overall'][5]:.4f} | psnr {cost['overall'][6]:.4f} | nrmse {cost['overall'][7]:.4f}

            """)

        # write to tensorboard
        ###opt###
        if opt.tensorboard:
            write_tensorboard(writer, cost, iteration, epoch, single_task_model, ratio, opt)





"""
=========== Multi-task utilities ===========
        used by Universal MT Trainer
"""

def _get_naive_weights(train_ratios, opt):
    """Get weights based on inverse proportion to dataset size
    """
    if opt.stratified:
        return {
            task : 1
            for task in train_ratios.keys()
        }
    total_slices = sum(train_ratios.values())

    # balance weights if not stratified; smaller datasets get larger weights
    naive_weights = {
        task : len(train_ratios) * (total_slices - train_ratios[task]) / total_slices
        for task in train_ratios.keys()
    }
    return naive_weights

           

"""
=========== Universal Multi-task Trainer ===========
                    user facing
"""


def multi_task_trainer(
    train_loader, val_loader,
    train_ratios, val_ratios,
    multi_task_model, writer,
    optimizer, scheduler,
    opt
):
    """Wrapper for multi task learning training.

    Takes care of:
    - loss weighting (naive, DWA, uncert)
    - saving best validation model
    - forward pass and backward propagation
    - gradient accumulation / averaging, is specified in opt
    - calculating and recording metrics
    - plotting validation iamges
    - displaying averaged metrics on terminal

    Called in mtl.py

    Parameters
    ----------
    train_loader : genDataLoader
        Contains slices from Train dataset. Slices shuffled.
    val_loader : genDataLoader
        Contains slices from Val dataset. Slices not shuffled.
    train_ratios : dict
        Keys are tasks. Values are the number of slices for each task.
    val_ratios : dict
        Keys are tasks. Values are the number of slices for each task
    multi_task_model : model-like object
        Weights are not loaded
    writer : tensorboard SummaryWriter
        Contains log directory information for tensorboard
    optimizer : torch.optim.<optimizer>
        usually Adam
    scheduler : torch.optim.<scheduler>
        usually StepLR
    opt : argparse.ArgumentParser
        Refer to help documentation in mtl.py

    Returns
    -------
    None

    """
    # naming (even if stratified, scarce / abundant ratio is preserved)
    ratio = f"N={'_N='.join(str(key) for key in train_ratios.values())}"

    # convenience
    task_count = len(opt.datasets)
    train_batch = len(train_loader)
    val_batch = len(val_loader)

    # naive weighting (accounts for stratified or not)
    weights = _get_naive_weights(train_ratios, opt)

    # for saving best validation model
    best_val_loss = np.infty

    # grad accumulation
    iteration = 0
    batch_count = 0 

    for epoch in range(opt.epochs):

        if opt.weighting == 'dwa':
            # contains info for last two epochs; must be ex'd in this order
            if epoch > 1:
                cost_prevprev = copy.deepcopy(cost_prev)
            if epoch > 0:
                cost_prev = copy.deepcopy(cost)

            # get dwa weights
            if epoch > 1:
                w = []
                for dataset in opt.datasets:
                    w.append(cost_prev[dataset][0] / cost_prevprev[dataset][0])

                softmax_sum = np.sum([
                    np.exp(w[idx_dataset] / opt.temp) 
                    for idx_dataset in range(len(opt.datasets))
                ])
                for idx_dataset, dataset in enumerate(opt.datasets):
                    weights[dataset] = task_count * np.exp(w[idx_dataset] / opt.temp) / softmax_sum
    

        # contains info for current epoch:
        cost = {
            task : np.zeros(8)
            for task in opt.datasets
        }
        cost['overall'] = np.zeros(8)

        # train the data
        multi_task_model.train()
        train_dataset = iter(train_loader)

        # grad accumulation
        optimizer.zero_grad() 

        for kspace, mask, esp, im_fs, task in train_dataset:
            task = task[0] # torch dataset loader returns as tuple

            # grad accumulation 
            iteration += 1
            batch_count += 1

            # forward; outputs on opt.device[0]
            _, im_us, logsigma = multi_task_model(
                kspace, mask, esp, task,
                )

            # crop so im_us has same size as im_fs
            im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
            # bring im_fs to the same gpu as im_us
            im_fs = im_fs.to(opt.device[0])

            # loss
            if opt.weighting == 'naive' or opt.weighting == 'dwa':
                loss = weights[task] * criterion(im_fs, im_us)

            elif opt.weighting == 'uncert':
                idx_task = opt.datasets.index(task)
                loss = 1 / (2 * torch.exp(logsigma[idx_task])) * \
                    criterion(im_fs, im_us) + \
                        logsigma[idx_task] / 2
                # for plotting purposes
                weights[task] = logsigma[idx_task]
            
            # grad accumulation
            if opt.gradaverage:
                loss /= opt.gradaccumulation

            loss.backward()

            # step optimizer once we've reached the right no. batches
            if batch_count == opt.gradaccumulation:
                optimizer.step()
                optimizer.zero_grad()
                
                # reset task batches
                batch_count = 0 

            # losses and metrics are averaged over epoch at the end
            # L1 loss for now
            cost[task][0] += loss.item()
            # ssim, psnr, nrmse
            for j in range(3):
                cost[task][j + 1] += metrics(im_fs, im_us)[j]


        # validation
        multi_task_model.eval()
        with torch.no_grad():

            # validation data
            val_dataset = iter(val_loader)
            for val_idx, val_data in enumerate(val_dataset):
                kspace, mask, esp, im_fs, task = val_data
                task = task[0]

                # forward pass; outputs are on opt.device[0]
                _, im_us, logsigma = multi_task_model(
                    kspace, mask, esp, task,
                    ) 

                # crop so im_us has same size as im_fs
                im_us = transforms.complex_center_crop(im_us, tuple(im_fs.shape[2:4]))
                # bring im_fs to the same device as im_us
                im_fs = im_fs.to(opt.device[0])
                loss = criterion(im_fs, im_us)

                # losses and metrics are averaged over epoch at the end
                # L1 loss for now
                cost[task][4] += loss.item() 

                # ssim, psnr, nrmse
                for j in range(3):
                    cost[task][j + 5] += metrics(im_fs, im_us)[j]

               # visualize reconstruction every few epochs
                if opt.tensorboard and epoch % opt.savefreq == 0: ###opt###
                    # if single task, only visualize 17th slice
                    if (val_idx == 17 or val_idx == val_batch - 17):
                        writer.add_figure(
                            f'{ratio}/{task}',
                            plot_quadrant(im_fs, im_us),
                            epoch, close = True,
                        )

        # update overall
        cost['overall'] = np.sum([cost[task] for task in opt.datasets], axis = 0)
        cost["overall"][:4] /= train_batch
        cost["overall"][4:] /= val_batch

        # average out
        for task in opt.datasets:
            if opt.stratified:
                # scarce / abundant have same effective sizes
                cost[task][:4] /= train_batch / len(opt.datasets)
            else:
                cost[task][:4] /= train_ratios[task]
            cost[task][4:] /= val_ratios[task]



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
            print(f"""
            >EPOCH (full run-throughs of abundant dataset): {epoch + 1:04d}; ITERATION: {iteration}
            TRAIN: loss {cost['overall'][0]:.4f} | ssim {cost['overall'][1]:.4f} | psnr {cost['overall'][2]:.4f} | nrmse {cost['overall'][3]:.4f}
            VAL: loss {cost['overall'][4]:.4f} | ssim {cost['overall'][5]:.4f} | psnr {cost['overall'][6]:.4f} | nrmse {cost['overall'][7]:.4f}

            """)

        # write to tensorboard
        ###opt###
        if opt.tensorboard:
            write_tensorboard(writer, cost, iteration, epoch, multi_task_model, ratio, opt, weights)
