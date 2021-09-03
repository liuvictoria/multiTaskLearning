import numpy as np
import sigpy as sp
import os
import h5py
import pathlib  # pathlib is a good library for reading files in a nested folders

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler
from typing import Iterator

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
        center_fractions, accelerations,
        use_same_mask = False,
    ):
        self.rng = np.random.default_rng(seed)
        # where dataset references are stored
        self.examples = []
        # ratios of each contrast
        self.ratios = {}
        # for evaluation; plot MRI slices contiguously
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
        self.use_same_mask = use_same_mask
    

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

    
    def contrast_labels(self):
        # will be populated as [0, 0, 0, 1, 1, 1, 1, 1, ...]
        labels = np.empty(sum(self.ratios.values())).astype(int)

        # self.ratios keys are in insertion order, python >= 3.7
        start_idx = 0
        for idx_contrast, contrast_count in enumerate(self.ratios.values()):
            labels[start_idx : start_idx + contrast_count] = np.full(
                contrast_count, idx_contrast
                )
            start_idx += contrast_count
        return labels


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

        if self.use_same_mask:
            # for evaluation only
            mask = self.mask_func(list(im_true.shape) + [1], seed = 0)[..., 0]
        else:
            # different masks for training; data augmentation
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


# BalancedSampler code modified from kaggle #
# https://www.kaggle.com/shonenkov/class-balance-with-pytorch-xla #

class BalancedSampler(Sampler):
    """
    Abstraction over data sampler.
    Allows you to create stratified sample on unbalanced classes.
    """

    def __init__(
        self, 
        labels: np.ndarray, 
        method: str = 'upsample'
        ):
        """
        Args:
            labels (np.ndarray): ndarray of class label
                for each elem in the dataset
            method (str): Strategy to balance classes.
                Must be one of [downsample, upsample]
        """
        super().__init__(labels)

        samples_per_class = {
            label: (labels == label).sum() for label in np.unique(labels)
        }

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist()
            for label in np.unique(labels)
        }

        assert method in ['downsample', 'upsample'], 'method for stratification invalid'

        if method == 'downsample':
            samples_per_class = min(samples_per_class.values())
        else:
            samples_per_class = max(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        
        # len of total slices used in the iteration; includes repeats
        self.length = self.samples_per_class * len(np.unique(labels)) 

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        # holds correct number of randomized indices for each label
        indices = np.empty((
            len(np.unique(self.labels)), 
            self.samples_per_class
            ))

        for key in sorted(self.lbl2idx):
            # for abundant dataset, repeat_times = 1
            repeat_times = int(np.ceil(
                self.samples_per_class / len(self.lbl2idx[key])
            ))

            indices_repeated = np.tile(
                self.lbl2idx[key],
                repeat_times,
            )
        
            indices[key][:] = np.random.choice(
                indices_repeated, self.samples_per_class, replace = False,
            )

        # interleave the randomized indices of each label
        interleaved = np.empty(
            self.samples_per_class * len(np.unique(self.labels)), 
            dtype = int
            )
        for idx_start, label_indices in enumerate(indices):
            # every nth will be a slice from the same contrast (n contrasts total)
            interleaved[idx_start::len(np.unique(self.labels))] = label_indices
        return iter(interleaved)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length



def genDataLoader(
    roots, scarcities,
    center_fractions, accelerations,
    shuffle, num_workers, seed=123,
    stratified = False, method = 'upsample',
    use_same_mask = False,
):
    '''
    if shuffle = True, but stratified = True, 
    then shuffle will be overriden w/ shuffle = False
    so in general, this allows us to say shuffle = True
    for all train dataloaders, even if by accident

    the second returned element is always the ratio
    between scarce / abundant without stratification.
    Stratification counts are taken care of in wrappers.py
    '''

    dset = MRIDataset(
        roots = roots, scarcities = scarcities, seed = seed, 
        center_fractions = center_fractions, accelerations = accelerations,
        use_same_mask = use_same_mask,
        )
    # only for beginning of training
    if stratified:
        sampler = BalancedSampler(
            labels = dset.contrast_labels(),
            method = method,
        )
        return (
            DataLoader(
                dset, batch_size = 1, 
                sampler = sampler,
                shuffle = False, num_workers = num_workers
            ),
            dset.ratios,
        )
    # if val or test, we will never have stratified.
    else:
        return (
            DataLoader(
                dset, batch_size=1, 
                shuffle = shuffle, num_workers = num_workers,
                ), 
            dset.ratios, 
            dset.slices
            )