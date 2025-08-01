# define training configuration
# for your own use cases, you might want to change the data paths, model path, run_dir, and other hyperparameters

config = """
# data
data:
  instruct_data: ""  # Fill
  data: ""  # Optionally fill with pretraining data 
  eval_instruct_data: ""  # Optionally fill

# model
model_id_or_path: "/mistral-models/mistral-7B-v0.3"  # Change to downloaded path
lora:
  rank: 64

# optim
seq_len: 32768
batch_size: 1
max_steps: 300
optim:
  lr: 6.e-5
  weight_decay: 0.1
  pct_start: 0.05

# other
seed: 0
log_freq: 1
eval_freq: 100
no_eval: False
ckpt_freq: 100

save_adapters: True  # save only trained LoRA adapters. Set to `False` to merge LoRA adapter into the base model and save full fine-tuned model

run_dir: "./runs/test_ultra"  # Fill

wandb:
  project: null # your wandb project name
  run_name: "" # your wandb run name
  key: "" # your wandb api key
  offline: False # Fill
"""

# save the same file locally into the example.yaml file
import yaml
with open('example.yaml', 'w') as file:
    yaml.dump(yaml.safe_load(config), file)