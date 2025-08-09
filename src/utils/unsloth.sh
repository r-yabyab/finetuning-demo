# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb#scrollTo=GqwZAVbRtjWF

sudo apt-get update

# check if has nvidia drivers
# dl deps first then this or else wheels for xformers gets stuck
# https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/index.html
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers list --gpgpu    # 24.04 LTS AMI should list 535 as first
# sudo ubuntu-drivers install --gpgpu nvidia:535-server
    sudo apt install -y nvidia-driver-535 nvidia-utils-535 # check uname -r if aws or generic, currently works on west-2 terraform... works for both aws/generic
# sudo apt install nvidia-utils-535-server
# sudo apt install nvidia-utils-535
sudo reboot # reboot to apply changes, could do from aws as well
nvidia-smi # should show a table like thing
# watch -n 5 nvidia-smi

# #18gb vol
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc
# # a, a, ENTER
# conda create --name unsloth_env \
#     python=3.11 \
#     pytorch-cuda=12.1 \
#     pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers
# conda activate unsloth_env


# Install Miniconda silently
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Initialize Conda for current shell
$HOME/miniconda3/bin/conda init
source ~/.bashrc

# Create conda environment non-interactively
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create --yes --name unsloth_env \
  python=3.11 \
  pytorch-cuda=12.1 \
  pytorch cudatoolkit xformers \
  -c pytorch -c nvidia -c xformers
conda activate unsloth_env

pip install unsloth


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