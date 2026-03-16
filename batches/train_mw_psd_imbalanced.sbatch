#!/bin/bash

#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --job-name=train_mw_psd_imbalanced
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=START,END,FAIL

source ~/.bashrc
conda activate roamm

cd /gpfs1/pi/djangraw/hsun11/roamm_ml/scripts

python3 train_mw_psd_imbalanced.py

