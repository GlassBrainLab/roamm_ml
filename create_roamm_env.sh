#!/bin/bash

# Script to create the 'roamm' conda environment with Python 3.9.17 and install dependencies

# Create the conda environment
conda create -n roamm python=3.9.17 -y

# Activate the environment
conda activate roamm

# Install pip packages from requirements.txt, skipping the first line
pip install -r <(tail -n +2 requirements.txt)

echo "Environment 'roamm' created and dependencies installed."