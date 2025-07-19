sudo apt-get update
sudo apt install -y build-essential

# need to downgrade
# also use python 3.12
pip install mistral-common==1.5.0

# set wandb.project: null in 7B.yaml





# cd $HOME/mistral-finetune


# pip install torch==2.2
# pip install -r requirements.txt

#then you can do
#cd $HOME/mistral-finetune
#python -m utils.validate_data --train_yaml example/7B.yaml

#########
# for aws
# sudo apt install python3-pip
# sudo apt install python3-venv -y

# use python3.10 on aws ubuntu 24.04, don't need to install torch just requirements.