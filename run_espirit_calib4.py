import os, sys
import numpy as np
import h5py
import sigpy as sp
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)
import sigpy.plot as pl
import pathlib
import bart
import tqdm

base_dir = pathlib.Path('/mnt/dense/kanghyun/summer_dset')

for h5file in base_dir.glob('*axial_t2*/*/*.h5'):
    print('Processing', h5file)
    hf = h5py.File(h5file,'a')
    esp_maps = []
    nslice = len(hf['kspace'])
    for sl in range(nslice):
        ksp_slice = hf['kspace'][sl]
        ksp_slice = np.moveaxis(ksp_slice,0,-1)
        maps = bart.bart(1,'ecalib -m 1 -d 0 -a -g -r 24',ksp_slice[None])
        esp_maps.append(np.moveaxis(maps[0],-1,0))
    esp_maps = np.stack(esp_maps,0)
    print('Saving the espirit_maps of size:', esp_maps.shape)
    hf.create_dataset('esp_maps', data=esp_maps)
    hf.close()
