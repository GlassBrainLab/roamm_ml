#!/bin/bash

# Bash script to run EEG-to-Text training
# Usage: ./train_decoding.sh

python3 train_decoding_roamm.py \
    -m BrainTranslator \
    -t all \
    -d all \
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
