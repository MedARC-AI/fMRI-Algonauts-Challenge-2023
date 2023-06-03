#!/bin/bash
#SBATCH --account=fmri
#SBATCH --partition=g40
#SBATCH --job-name=algo_wds
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     # should = number of gpus
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)
#SBATCH --comment=fmri
#SBATCH -e slurms/%j.err
#SBATCH -o slurms/%j.out
#SBATCH --array=9

source /fsx/home-paulscotti/.bashrc
cd /fsx/proj-medarc/fmri/paulscotti/fMRI-Algonauts-Challenge-2023/src
srun python algo_webdataset_creation.py 0 $SLURM_ARRAY_TASK_ID