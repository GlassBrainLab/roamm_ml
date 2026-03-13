#!/bin/bash

#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
##SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=train_reve_roamm
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=START,END,FAIL

source ~/.bashrc
conda activate roamm

cd /gpfs1/pi/djangraw/hsun11/roamm_ml/eegfm

python3 train_reve_roamm.py

