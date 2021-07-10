import os
import numpy as np
import torch

from utils import criterion, metrics
from utils import plot_quadrant, write_tensorboard


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





"""
=========== Multi-task utilities ===========
        used by Universal MT Trainer
"""

def _get_naive_weights(train_ratios):
    total_slices = sum(train_ratios.values())
    naive_weights = {}
    for contrast in train_ratios.keys():
        # balance weights; smaller datasets get larger weights
        naive_weights[contrast] = len(train_ratios) * (total_slices - train_ratios[contrast]) / total_slices
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
    # convenience
    train_batch = len(train_loader)
    val_batch = len(val_loader)

    # naive weighting
    weights = _get_naive_weights(train_ratios)

    # naming
    ratio = f"N={'_N='.join(str(key) for key in train_ratios.values())}"

    # for saving best validation model
    best_val_loss = np.infty

    for epoch in range(opt.epochs):
        # contains info for single epoch
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

            optimizer.zero_grad()
            _, im_us = multi_task_model(kspace, mask, esp, contrast) # forward pass
            loss = weights[contrast] * criterion(im_fs, im_us)
            loss.backward()
            
            # combine two losses and then do back-prop
            # gradient accumulation; for shared layer, want to go in between; two accumulation
            # four datasets might be unstable?
            # sgd vs mini-batch
            
            optimizer.step()

            # losses and metrics are averaged over epoch at the end
            # L1 loss for now
            cost[contrast][0] += loss.item() / weights[contrast] # undo weighting to get on same scales
            # ssim, psnr, nrmse
            for j in range(3):
                cost[contrast][j + 1] += metrics(im_fs, im_us)[j]

            # # if not using naive weights, change weights here
            # if opt.weighting == 'uncert':
            #     continue
            # elif opt.weighting == 'dwa':
            #     continue
            # at end of each epoch should be fine; bc unbalanced; weighting could fluctuate a lot from batch size of 1


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

                _, im_us = multi_task_model(kspace, mask, esp, contrast) # forward pass
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
            cost[contrast][:4] /= train_ratios[contrast]
            cost[contrast][4:] /= val_ratios[contrast]



        # early stopping
        if cost['overall'][4] < best_val_loss:
            best_val_loss = cost['overall'][4]
            filedir = f"models/{opt.experimentname}_{opt.network}_{'_'.join(opt.datasets)}"
            if not os.path.isdir(filedir):
                os.makedirs(filedir)
            torch.save(
                multi_task_model.state_dict(),
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
            write_tensorboard(writer, cost, epoch, multi_task_model, ratio, opt, weights)
