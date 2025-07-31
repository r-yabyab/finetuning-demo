from unsloth import FastLanguageModel
from transformers import TextStreamer
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
messages = [{
    "role": "user",
    "content": "Who are you?"
}]


model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit", # Using instruct version for better chat performance
    model_name = "outputs/checkpoint-216", # Using instruct version for better chat performance
    # model_name = "unsloth/mistral-7b-bnb-4bit", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    map_eos_token = True, # Maps <|im_end|> to </s> instead
)

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
).to("cuda")

text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(
  input_ids = inputs,
  streamer = text_streamer,
  temperature = 1.0,
  top_p = 0.95,
  top_k = 64,
  max_new_tokens = 100,
  use_cache = True
)