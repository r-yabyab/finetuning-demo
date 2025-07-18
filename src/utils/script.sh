sudo apt-get update
sudo apt install -y build-essential


cd $HOME/mistral-finetune

pip install torch==2.2
pip install -r requirements.txt

#then you can do
#cd $HOME/mistral-finetune
#python -m utils.validate_data --train_yaml example/7B.yaml