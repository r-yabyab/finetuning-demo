#!/usr/bin/env bash
set -e  # exit if any command fails

# chmod +x unsloth-setup.sh

# Download and install Miniconda silently
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p "$HOME/miniconda3"
rm miniconda.sh

# Initialize Conda for current shell
"$HOME/miniconda3/bin/conda" init

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
# source ~/.bashrc

# Accept TOS for Anaconda repositories
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment non-interactively
conda create --yes --name unsloth_env \
  python=3.11 \
  pytorch-cuda=12.1 \
  pytorch cudatoolkit xformers \
  -c pytorch -c nvidia -c xformers

# Activate environment
conda activate unsloth_env

# Install Unsloth
pip install unsloth

# for logging
pip install tensorboard