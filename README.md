# Multi-Task Learning for Accelerated MR Reconstruction

This repository contains the source code of multi-task and baseline variational networks from our manuscript, "Multi-Task Accelerated MR Reconstruction Schemes for Jointly Training Multiple Contrasts", introduced by [Victoria Liu](https://liuvictoria.github.io/), [Kanghyun Ryu](https://scholar.google.co.kr/citations?user=w5SLrr6uQq0C&hl=ko), [Cagan Alkan](https://openreview.net/profile?id=~Cagan_Alkan1), [John Pauly](https://web.stanford.edu/~pauly/), and [Shreyas Vasanawala](http://bodymri.stanford.edu/shreyasvasanawala). The manuscript is currently under review for the 2021 NeuIPS Deep Inverse Workshop and will be publicly available after October 15, 2021; please check back soon. In the meantime, here is our abstract and an MTL network visualization.

## Abstract

Model-based accelerated MRI reconstruction methods leverage large datasets to reconstruct diagnostic-quality images from undersampled k-space. These networks require matching training and test time distributions to achieve high quality recon- structions. However, there is inherent variability in MR datasets, including different contrasts, orientations, anatomies, and institution-specific protocols. The current paradigm is to train separate models for each dataset. However, this is a demanding process and cannot exploit information that may be shared amongst datasets. To address this issue, we propose multi-task learning (MTL) schemes that can jointly reconstruct multiple datasets. We test multiple MTL architectures and weighted loss functions against single task learning (STL) baselines. Our quantitative and qualitative results suggest that MTL can outperform STL across a range of dataset ratios for two knee contrasts.


<img width="1781" alt="MTL network schemes" src="https://user-images.githubusercontent.com/66798771/134863166-53a01e08-02da-4081-b321-03a02bc5ca14.png">

Figure 1: Multi-task learning network architectures built on an unrolled variational network for (a) fully split and (b) shared-encoder-split-decoder blocks.

## Setup
- Clone this repo:

```
git clone https://github.com/liuvictoria/multitasklearning.git
```

- Create a virtual environment using `conda-env.txt`. Install packages in `requirements.txt`

- Download MRI data from [mridata.org](http://mridata.org/list?project=NYU%20machine%20learning%20data). We use the PDw and PDw-FS coronal volumes.

## Usage

In the multiTaskLearning folder, you'll be able to run STL, transfer learning, MTL, and evaluation experiments from the command line. There are required and optional user-defined values that are passed in as flags. Here are some examples:

#### STL Joint training

```
python3 stl.py --datasets div_coronal_pd_fs div_coronal_pd --epochs 120 --experimentname STL_joint --savefreq 10 --device cuda:2 --scarcities 1 2 3
```

#### STL Non-joint training
```
python3 stl.py --datasets div_coronal_pd --epochs 120 --experimentname STL_nojoint --savefreq 10 --device cuda:2 --scarcities 1 2 3 --mixeddata 0
```

#### MTL multi-head, uncertainty training
```
python3 mtl.py --datasets div_coronal_pd_fs div_coronal_pd --weighting uncert --blockstructures trueshare trueshare mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare mhushare --epochs 120 --experimentname MTL_uncert_mhushare --device cuda:2 --scarcities 1 2 3 --savefreq 10
```

#### MTL split, naive training
```
python3 mtl.py --datasets div_coronal_pd_fs div_coronal_pd --weighting naive --blockstructures trueshare trueshare split split split split split split split split split split --epochs 120 --experimentname MTL_naive_split --device cuda:1 --scarcities 1 2 3 --savefreq 10
```

#### Transfer learning
```
python3 stl.py --datasets div_coronal_pd_fs --epochs 120 --experimentname STL_transfer5e-4 --savefreq 10 --device cuda:1 --scarcities 1 2 3 --lr 0.0005 --weightsdir /mnt/dense/vliu/summer_runs_models/models/STL_baselines/STL_nojoint_varnet_div_coronal_pd/N=481_l1.pt --mixeddata 0
```

#### Evaluation for split, naive network
```
python3 evaluate.py --datasets div_coronal_pd_fs div_coronal_pd --stratified 0 --network varnet --shareetas 1 --mixeddata 1 --scarcemax 497 --device cuda:1 --plotdir neurips/MTL_naive_split_varnetIIVVVVVVVVVV_div_coronal_pd_fs_div_coronal_pd --experimentnames MTL_naive_split --blockstructures IIVVVVVVVVVV --showbaselines 1
```

In addition, there is a script to delete aborted runs (usually due to GPU issue) and a script to visualize the undersampled images before any training:

```
python3 deleteDuplicateRuns.py --runnames MTL_uncert_gradacc_varnetIIYYYYYYYYYY_div_coronal_pd_fs_div_coronal_pd --rundir ../../../mnt/dense/vliu/runs/one_eta_pdfs_scarce
```

```
python3 getusamp.py --datasets div_coronal_pd_fs div_coronal_pd --device cuda:2 --plotdir usamp
```

## Flags

In the following tables, we list _some_ flags that may be of interest to you. The set of required flags differ amongst scripts, and there may be default values for some of the flags. Please consult the extensive documentation within the code for this additional information.

#### Hyperparameters

| Flag Name        | Usage  |  Relevant in |
| ------------- |-------------| -----|
| `epochs` | number of epochs to run | mtl, stl |
| `lr` | learning rate | mtl, stl |
| `gradaccumulation` | how many iterations per gradient accumulation | mtl, stl |

#### Network

| Flag Name        | Usage  |  Relevant in |
| ------------- |-------------| -----|
| `blockstructures` | explicit list of what each unrolled block's convolutional network will be | mtl, stl, evaluate |
| `shareetas` | whether to share data consistency term eta | mtl, stl, evaluate |
| `weighting` | naive, uncert, dwa | mtl, stl, evaluate |
| `device` | cuda:2 device default | mtl, stl, evaluate |


#### Dataset & undersampling

| Flag Name        | Usage  |  Relevant in |
| ------------- |-------------| -----|
| `datasets` | names of relevant tasks | mtl, stl, evaluate |
| `datadir` | data root directory; where are datasets contained | mtl, stl, evaluate |
| `scarcities` | number of samples in second task will be decreased by 1/2^N | mtl, stl |
| `accelerationse` | list of undersampling factor of k-space for training | mtl, stl, evaluate |
| `centerfracs` | list of center fractions sampled of k-space for training | mtl, stl, evaluate |





#### Data loader properties

| Flag Name        | Usage  |  Relevant in |
| ------------- |-------------| -----|
| `stratified` | whether to use a stratified dataloader | mtl, stl |
| `numworkers` | number of workers for PyTorch dataloader | mtl, stl, evaluate |


#### Save & display results

| Flag Name        | Usage  |  Relevant in |
| ------------- |-------------| -----|
| `experimentname` | experiment name for training only | mtl, stl |
| `verbose` | verbosity on terminal | mtl, stl |
| `tensorboard` | whether to create tensorboard | mtl, stl, evaluate |
| `savefreq` | how frequently to save recon image | mtl, stl, evaluate |


#### Evaluate.py specific flags

| Flag Name        | Usage  |
| ------------- |-------------|
| `colors` | https://docs.bokeh.org/en/latest/docs/reference/palettes.html#bokeh-palette |
| `createplots` | whether to create plots of metrics for different ratios of MRI |
| `showbaselines` | whether to show (pre-calculated) STL baselines in plot |
| `showbest` | whether to show best mtl run |

## Workflow samples

You can explore how MR images improve throughout training epochs. The top left is the reconstructed image, and the top right is the ground truth. Bottom row contains error maps.

![validation, view slices throughout epochs on tensorboard](https://user-images.githubusercontent.com/66798771/134854302-46e26acc-ec83-4555-8516-0f2c32988e8f.gif)

The Bokeh plots are interactive. You can zoom in, see exact values by hovering over a point, and declutter the plot by clicking the legend.

![plot, interactive bokeh](https://user-images.githubusercontent.com/66798771/134877253-07e8a265-f5b7-4649-8220-4d7e93572897.gif)

At the inference step, you can also scroll through entire MRI volumes to see how reconstruction affects different slices.

![evaluation, interactive view of MRI slices on tensorboard](https://user-images.githubusercontent.com/66798771/134854191-b7934fb6-0de0-4eea-a86c-dc3817b2fbd4.gif)

## Acknowledgments 
We thank the Stanford MRSRL group for their insightful comments during group meetings.

## Citation
If you found our repository or paper helpful, please cite:

{to be updated on October 15 when the preprint is released}

## Contact
If you have any questions or comments, please contact me at [vliu@caltech.edu](mailto:vliu@caltech.edu)