from unsloth import FastLanguageModel
import torch
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/mistral-7b-v0.3", # "unsloth/mistral-7b" for 16bit loading
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    # r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    # r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                    #   "embed_tokens", "lm_head",
                      ], # Add for continual pretraining
    lora_alpha = 32,
    # lora_dropout = 0, # Supports any, but = 0 is optimized
    lora_dropout = 0.05, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

from datasets import load_dataset
dataset = load_dataset("json", data_files="answers_only.jsonl", split = "train")
# Split dataset into train and eval (90% train, 10% eval)
dataset = dataset.train_test_split(test_size=0.1, seed=3407)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

EOS_TOKEN = tokenizer.eos_token
def formatting_prompts_func(examples):
    return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
train_dataset = train_dataset.map(formatting_prompts_func, batched = True,)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,)

for row in train_dataset[:5]["text"]:
    print("=========================")
    print(row)

    

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import UnslothTrainer, UnslothTrainingArguments

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        # per_device_train_batch_size = 2,
        # gradient_accumulation_steps = 8,
        gradient_accumulation_steps = 16,

        warmup_ratio = 0.1,
        num_train_epochs = 1,

        learning_rate = 5e-5,
        embedding_learning_rate = 5e-6,
        # fp16=True,
        # use_fp16 = True,
        logging_steps = 1,
        eval_steps = 15,
        eval_strategy = "steps",
        # save_strategy = "steps",
        # save_steps = 50,
        optim = "adamw_8bit",
        weight_decay = 0.00,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs-self",
        report_to = "tensorboard", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()