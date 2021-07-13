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
    '''
    self.ratios: dict of number of slices per contrast
    self.slices: list of number of slices in each MRI, in order from root
    '''

    def __init__(
        self, roots, scarcities, seed,
        center_fractions, accelerations
    ):
        self.rng = np.random.default_rng(seed)
        self.examples = []
        self.ratios = {}
        self.slices = []
        for idx, root in enumerate(roots):
            contrast = root.split('/')[-2]
            Files = sorted(list(pathlib.Path(root).glob('*.h5')))
            
            # subsample files
            Files = self.subset_sample(Files, scarcities[idx])
            # track file count for ratios
            file_count = 0
            # individual slices
            for fname in Files:
                h5file = h5py.File(fname, 'r')
                kspace = h5file['kspace']
                nsl = kspace.shape[0]  # get number of slices
                self.examples += [(fname, sl, contrast) for sl in range(1, nsl)]
                file_count += nsl - 1
                self.slices.append(nsl - 1)

            self.ratios[contrast] = file_count
            
        center_fractions, accelerations = self.combine_cenfrac_acc(
            center_fractions, accelerations,
            )    
        self.mask_func = subsample.EquispacedMaskFunc(
            center_fractions=center_fractions, accelerations=accelerations
        )

    def subset_sample(self, Files, scarcity):
        '''
        decrease number of Files by 1/2^{scarcity} in a reproducible manner
        '''
        for _ in range(scarcity):
            if int(len(Files) / 2) > 0:
                Files = self.rng.choice(
                    Files, 
                    int(len(Files) / 2), 
                    replace=False
                )
        return list(Files)

    def combine_cenfrac_acc(self, cen_fracs, accs):
        '''
        [c_1, c_2] [a_1, a_2]
        becomes
        [c_1, c_2, c_1, c_2] [a_1, a_1, a_2, a_2]
        to match for EquispacedMaskFunc
        '''
        accs_final = []
        for acc in accs:
            accs_final += [acc] * len(cen_fracs)
        cen_fracs_final = cen_fracs * len(accs)
        return cen_fracs_final, accs_final

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, sl, contrast = self.examples[idx]
        with h5py.File(fname, 'r') as hr:
            kspace, esp_maps = hr['kspace'][sl], hr['esp_maps'][sl]
        esp_maps = np.complex64(esp_maps)
        kspace = kspace / 10  # divide by 10 because the values are too large
        im_coil = sp.ifft(kspace, axes=[1, 2])
        im_true = np.sum(im_coil * np.conj(esp_maps), axis=0)

        mask = self.mask_func(list(im_true.shape) + [1])[..., 0]
        mask = np.expand_dims(mask, axis=0)
        masked_kspace = kspace * mask  # undersampled kspace
        mask = np.expand_dims(mask, axis=-1)

        # Now transform everything to tensor.
        # The complex kspace will be changed to [real, imag] in the final axis
        masked_kspace = transforms.to_tensor(masked_kspace)
        mask = transforms.to_tensor(mask)
        esp_maps = transforms.to_tensor(esp_maps)
        im_true = np.expand_dims(im_true, axis=0)
        im_true = transforms.to_tensor(im_true)

        # crop to center to get rid of coil artifacts from sensitivity maps
        im_true = transforms.complex_center_crop(im_true, (360, 320))

        return masked_kspace, mask.byte(), esp_maps, im_true, contrast


def genDataLoader(
    roots, scarcities,
    center_fractions, accelerations,
     shuffle, num_workers, seed=333,
):
    dset = MRIDataset(
        roots = roots, scarcities = scarcities, seed = seed, 
        center_fractions = center_fractions, accelerations = accelerations,
        )
    return (DataLoader(dset, batch_size=1, shuffle=shuffle, num_workers=num_workers), dset.ratios, dset.slices)