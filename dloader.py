import numpy as np
import sigpy as sp
import os
import h5py
import pathlib  # pathlib is a good library for reading files in a nested folders

from torch.utils.data import DataLoader, Dataset
import torch

import fastmri  # We will also use fastmri library
# use for generating undersampling mask, transforming tensors
from fastmri.data import subsample, transforms

# This is how you can make a custom dataset class


class MRIDataset(Dataset):
    def __init__(self, root, center_fractions=[0.06, 0.06, 0.06], accelerations=[4, 5, 6]):
        self.examples = []
        Files = list(pathlib.Path(root).glob('*.h5'))
        for fname in Files:
            h5file = h5py.File(fname, 'r')
            kspace = h5file['kspace']
            nsl = kspace.shape[0]  # get number of slices
            self.examples += [(fname, sl) for sl in range(nsl)]

        self.mask_func = subsample.EquispacedMaskFunc(
            center_fractions=center_fractions, accelerations=accelerations)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, sl = self.examples[idx]
        with h5py.File(fname, 'r') as hr:
            kspace, sens = hr['kspace'][sl], hr['sens'][sl]
        kspace = kspace / 10  # divide by 10 because the values are too large
        im_coil = sp.ifft(kspace, axes=[1, 2])
        # im_true is the fully sampled reconned image
        im_true = np.sum(im_coil * np.conj(sens), axis=0)

        mask = self.mask_func(list(im_true.shape) + [1])[..., 0]
        mask = np.expand_dims(mask, axis=0)
        masked_kspace = kspace * mask  # undersampled kspace
        mask = np.expand_dims(mask, axis=-1)

        # Now transform everything to tensor. The complex kspace will be changed to [real, imag] in the final axis
        masked_kspace = transforms.to_tensor(masked_kspace)
        mask = transforms.to_tensor(mask)
        sens = transforms.to_tensor(sens)
        im_true = np.expand_dims(im_true, axis=0)
        im_true = transforms.to_tensor(im_true)

        return masked_kspace, mask.byte(), sens, im_true


def genDataLoader(dset, shuffle=True):
    return DataLoader(dset, batch_size=1, shuffle=shuffle, num_workers=16)
