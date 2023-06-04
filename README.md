# MedARC: Algonauts Challenge 2023

Predicting human brain fMRI activity from the images participants perceived in the Natural Scenes Dataset.

This repository is for competition in the 2032 [Algonauts Challenge](http://algonauts.csail.mit.edu/). 

Please join the [MedARC Discord](https://discord.com/invite/CqsMthnauZ) to join our team as a volunteer contributor (join our weekly Monday meetings; see our [weekly notes](https://docs.google.com/document/d/1AckB0eowQOi7q173KzUH1Gny95Ddu5XenrDIho5daDk/edit#)).

## Installation instructions

1. Download this repository: ``git clone https://github.com/MedARC-AI/fMRI-Algonauts-Challenge-2023.git``

2. Get our webdataset implementation of the Algonauts Challenge training data. If you are on Stability HPC, it is located in ``/fsx/proj-medarc/fmri/natural-scenes-dataset/algonauts_data/wds``. Otherwise, you need to [download the NSD dataset yourself](https://cvnlab.slite.page/p/dC~rBTjqjb/How-to-get-the-data) and then use ``src/algo_webdataset_creation.ipynb`` to convert it to webdataset format.

3. Run ``setup.sh`` to create a conda environment that contains the packages necessary to run our scripts; activate the environment with ``conda activate mindeye``.

```bash
cd fMRI-Algonauts-Challenge-2023/src
. setup.sh
```

## Usage

See train_decoder.ipynb
