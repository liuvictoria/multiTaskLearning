"""Docstring for the dloader.py module.

Custom data loader for MRI data.

"""

import os
import pathlib 
# pathlib is a good library for reading files in a nested folders
from typing import Iterator
import h5py
import numpy as np
import sigpy as sp

from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler
import torch

import fastmri 
# used for generating undersampling mask, transforming tensors
from fastmri.data import subsample, transforms

class MRIDataset(Dataset):
    """Creates custom dataloader for specified data.

    Parameters
    ----------
    roots : List
        Directories of MR data to comprise the dataloader.
        Specify the appropriate TaskName/Split for each root.
    scarcities : List[int]
        Each scarcity represents the downsampling factor for dataset size.
        The ith scarcity is the downsampling factor for the ith
        dataset in parameter roots. Ideally, len(scarcities) == len(roots),
        but this is not enforced.
    seed : int
        RNG seed. Keep it constant between runs.
    center_fractions : List
        Option to provide multiple center_fractions for data augmentation.
    accelerations : List
        Option to provide multiple accelerations for data augmentation.
    use_same_mask : bool, default False
        Typically used for evaluation, where it may be good to have the 
        same pseudorandom masks for all evaluation samples.

    Attributes
    ----------
    self.examples : List
        Each element contains sample filename, slice count, & task info.
    self.ratios : Dict
        Keys are tasks. Values are the number of slices for each task.
    self.slices : List
        Keeps track of how many slices are in each volume; across tasks.
        primarily used in evaluation for visualization purposes.
    self.mask_func : EquispacedMaskFunc
        Mask with the appropriate center fracs and accelerations.

    Yields
    -------
    masked_kspace : Tensor
        The undersampled k-space
    mask.byte() : Byte
        The mask itself.
    esp_maps : Tensor
        ESPIRiT estimated expression maps, taken from h5 files.
    im_true : Tensor
        Ground truth
    task : str
        The task identifier (i.e. 'div_coronal_pd')
    
    Other Parameters
    ----------------
    Files : List
        Sorted list of full paths to MR samples.

    """

    def __init__(
        self, roots, scarcities, seed,
        center_fractions, accelerations,
        use_same_mask = False,
    ):
        self.rng = np.random.default_rng(seed)
        # where dataset references are stored
        self.examples = []
        # ratios of each task
        self.ratios = {}
        # for evaluation; plot MRI slices contiguously
        self.slices = []
        for idx, root in enumerate(roots):
            task = root.split('/')[-2]
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
                self.examples += [(fname, sl, task) for sl in range(1, nsl)]
                file_count += nsl - 1
                self.slices.append(nsl - 1)

            self.ratios[task] = file_count
        center_fractions, accelerations = self.combine_cenfrac_acc(
            center_fractions, accelerations,
            )    
        self.mask_func = subsample.EquispacedMaskFunc(
            center_fractions=center_fractions, accelerations=accelerations
        )
        self.use_same_mask = use_same_mask

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, sl, task = self.examples[idx]
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

        return masked_kspace, mask.byte(), esp_maps, im_true, task    

    def subset_sample(self, Files, scarcity):
        """Decreases the number of Files by 1/2^{scarcity} pseudorandomly.

        Parameters
        ----------
        Files : List
            Sorted list of full paths to a specific task's .h5 files
        scarcity : int
            The downsampling factor for this particular task.
        Returns
        -------
        Files : List
            A downsampled list of the task's files.

        """
        for _ in range(scarcity):
            if int(len(Files) / 2) > 0:
                Files = self.rng.choice(
                    Files, 
                    int(len(Files) / 2), 
                    replace=False
                )
        return list(Files)

    def combine_cenfrac_acc(self, cen_fracs, accs):
        """Cartesian product between center fractions and accelerations for
        data augmentation.

        Examples:
        --------
        >>> cen_fracs, accs = [c_1, c_2], [a_1, a_2]
        >>> combine_cenfrac_acc(cen_fracs, accs)
        [c_1, c_2, c_1, c_2], [a_1, a_1, a_2, a_2]

        """
        accs_final = []
        for acc in accs:
            accs_final += [acc] * len(cen_fracs)
        cen_fracs_final = cen_fracs * len(accs)
        return cen_fracs_final, accs_final

    
    def _task_labels(self):
        # will be populated as [0, 0, 0, 1, 1, 1, 1, 1, ...]
        labels = np.empty(sum(self.ratios.values())).astype(int)

        # self.ratios keys are in insertion order, python >= 3.7
        start_idx = 0
        for idx_task, task_count in enumerate(self.ratios.values()):
            labels[start_idx : start_idx + task_count] = np.full(
                task_count, idx_task
                )
            start_idx += task_count
        return labels




class balancedSampler(Sampler):
    """Abstraction over data sampler to create stratified sampler
    on unbalanced classes.

    Parameters
    ----------
    labels : ndarray
        Each element is the task of an MR slice in the dataset
    method : str
        Must be one of ['downsample', 'upsample']
    
    Attributes
    -------
    self.samples_per_class : Dict
        Each task is a key, and values are the number of slices corresponding
        to that task. Values are based on parameter `method`.
    self.lbl2idx : Dict
        Each task is a key, and values are indices where the task is 
        represented in parameter `labels`

    Yields
    -------
    iter(interleaved) : Iterator
        stratified sampler of `labels` indices.
        i.e. For two tasks A & B, the iterator indices will sample A B A B.
        Note that interleaving makes the sample selection randomized
        within tasks.

    Raises
    ------
    'Method for stratification invalid'
        If method is not one of ['downsample', 'upsample']

    References
    -------
    https://www.kaggle.com/shonenkov/class-balance-with-pytorch-xla

    """

    def __init__(
        self, 
        labels: np.ndarray, 
        method: str = 'upsample'
        ):

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
            # every nth will be a slice from the same task (n tasks total)
            interleaved[idx_start::len(np.unique(self.labels))] = label_indices
        return iter(interleaved)

    def __len__(self) -> int:
        return self.length



def genDataLoader(
    roots, scarcities,
    center_fractions, accelerations,
    shuffle, num_workers, seed=333,
    stratified = False, method = 'upsample',
    use_same_mask = False,
):
    """Official Torch DataLoader building off of MRIDataset.

    Parameters
    ----------
    roots : List
        Directories of MR data to comprise the dataloader.
    scarcities : List[int]
        Each scarcity represents the downsampling factor for dataset size.
    center_fractions : List
        Option to provide multiple center_fractions for data augmentation.
    accelerations : List
        Option to provide multiple accelerations for data augmentation.
    shuffle : bool
        Shuffling should be off for evaluation and stratified.
    seed : int, default 333
        RNG seed. Keep it constant between runs.
    stratified : bool, default False
        Whether to use stratified dataloader to balance task sizes.
    method : str, default 'upsample'
        If stratified is True, what method to stratify datasets.
    use_same_mask : bool, default False
        Typically used for evaluation, where it may be good to have the 
        same pseudorandom masks for all evaluation samples.

    Returns
    -------
    DataLoader : Torch Dataloader
        Dataloader with batch size 1 and approriate sampler (i.e.
        stratified or not) / shuffle setting.
    MRIdataset.ratios : Dict
    MRIdataset.slices : List
    
    See Also
    ---------
    dloader.MRIDataset, dloader.balancedSampler for detailed descriptions of
    parameters and yields. This function feeds arguments to those modules.

    Notes
    -----
    If shuffle = True, but stratified = True, 
    then shuffle will be overriden w/ shuffle = False.
    In general, this allows us to say shuffle = True
    for all train dataloaders, even if by accident.

    The second yielded element is always the ratio
    between scarce / abundant without stratification.
    Stratification counts are taken care of in wrappers.py
    
    """

    dset = MRIDataset(
        roots = roots, scarcities = scarcities, seed = seed, 
        center_fractions = center_fractions, accelerations = accelerations,
        use_same_mask = use_same_mask,
        )
    # only for beginning of training
    if stratified:
        sampler = balancedSampler(
            labels = dset._task_labels(),
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