#!/bin/bash

#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=train_decoding_nr
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=START,END,FAIL

source ~/.bashrc
conda activate roamm

cd /gpfs1/pi/djangraw/hsun11/roamm_ml/eeg2text

python3 train_decoding_roamm.py \
    -m BrainTranslator \
    -t all \
    -d nr \
    -1step \
    -pre \
    -no-load1 \
    -ne1 20 \
    -ne2 30 \
    -lr1 0.00005 \
    -lr2 0.0000005 \
    -b 32 \
    -s ./checkpoints/decoding \
    -cuda cuda:0

