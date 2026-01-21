#!/bin/bash

#SBATCH --partition=nvgpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --job-name=eval_decoding
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=START,END,FAIL

source ~/.bashrc
conda activate roamm

cd /gpfs1/pi/djangraw/hsun11/roamm_ml/eeg2text

python3 eval_decoding_roamm.py \
    --checkpoint_path ./checkpoints/decoding/best/all_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path ./config/decoding/all_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0 -tf True -n True

python3 eval_decoding_roamm.py \
    --checkpoint_path ./checkpoints/decoding/best/nr_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path ./config/decoding/nr_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0 -tf True -n False

python3 eval_decoding_roamm.py \
    --checkpoint_path ./checkpoints/decoding/best/mw_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt \
    --config_path ./config/decoding/mw_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.json \
    -cuda cuda:0 -tf False -n True

