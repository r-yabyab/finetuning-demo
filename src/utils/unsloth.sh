# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=GqwZAVbRtjWF

#18gb vol
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
conda activate unsloth_env

pip install unsloth


# check if has nvidia drivers
# dl deps first then this or else wheels for xformers gets stuck
# https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/index.html
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers list --gpgpu
sudo ubuntu-drivers install --gpgpu nvidia:535-server
sudo apt install nvidia-utils-535-server
sudo reboot # reboot to apply changes, best if you rebooted from console
nvidia-smi
# watch -n 5 nvidia-smi
conda activate unsloth_env

# clone repo, then create /data with jsonl
git clone https://github.com/r-yabyab/finetuning-demo.git
cd finetuning-demo/src/
mkdir data
cd data
touch data_chunk_train.jsonl data_chunk_eval.jsonl # fill data, watch out for empty line at end of file
sudo vim data_chunk_train.jsonl
sudo vim data_chunk_eval.jsonl


# for converting llama.cpp, takes 32gb vol 
sudo apt install -y cmake build-essential pkg-config
sudo apt install -y libcurl4-openssl-dev

# ollama dl
curl -fsSL https://ollama.com/install.sh | sh

# extra setup commands here...
# and here
ollama create m/mymistral