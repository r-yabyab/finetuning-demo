#!/bin/bash

sudo apt-get update
sudo apt install -y build-essential

# miniconda venv
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create -n myenv python=3.12
conda activate myenv

# requirements
cd $HOME && git clone https://github.com/mistralai/mistral-finetune.git
cd mistral-finetune
pip install torch==2.2  
pip install -r requirements.txt # takes like 5 minutes on aws
pip install mistral-common==1.5.0 # downgrade needed

# download mistral 7B model
# takes like 10 minutes to dl
cd
mkdir -p ~/${HOME}/mistral_models
mkdir mistral_models
cd ${HOME} && wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-v0.3.tar
tar -xf mistral-7B-v0.3.tar -C mistral_models
sudo rm -r mistral-7B-v0.3.tar

# clone repo, then create /data with jsonl
git clone https://github.com/r-yabyab/finetuning-demo.git
cd finetuning-demo/src/
mkdir data
cd data
touch data_chunk_train.jsonl data_chunk_eval.jsonl # fill data, watch out for empty line at end of file
sudo vim data_chunk_train.jsonl
sudo vim data_chunk_eval.jsonl

# validate data and config yaml
cd $HOME/mistral-finetune/example
sudo vim 7B.yaml # put data
cd $HOME/mistral-finetune
python -m utils.reformat_data ../finetuning-demo/src/data/data_chunk_train.jsonl
python -m utils.validate_data --train_yaml example/7B.yaml

cd $HOME/finetuning-demo/src/
sudo vim yaml-create.py
# add mistral-finetune/example/7B.yaml to the block in .py
pip install pyyaml
python yaml-create.py

sudo vim example.yaml # Change seq_len to like 512 depending on data size
mv example.yaml ../../mistral-finetune/
cd $HOME/mistral-finetune

pip install "numpy<2.0"

# check if has nvidia drivers
# dl deps first then this or else wheels for xformers gets stuck
# https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/index.html
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers list --gpgpu
sudo ubuntu-drivers install --gpgpu nvidia:535-server
# sudo apt install nvidia-cuda-toolkit # skipped this
# nvcc --version    # and this skipped
sudo apt install nvidia-utils-535-server
sudo reboot # reboot to apply changes, best if you rebooted from console
nvidia-smi
# watch -n 5 nvidia-smi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

# start training
cd $HOME/mistral-finetune
torchrun --nproc-per-node 1 -m train example.yaml

# 16gb vram g4dn.xlarge gives this OOM error when run torch:
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 514.00 MiB. GPU 0 has a total capacity of 14.58 GiB of which 479.56 MiB is free. Including non-PyTorch memory, this process has 14.11 GiB memory in use. Of the allocated memory 13.58 GiB is allocated by PyTorch, and 115.49 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)


# g4dn.2xlarge, 32gb, $0.752
