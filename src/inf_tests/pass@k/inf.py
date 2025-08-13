from unsloth import FastLanguageModel
max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# Read prompt from gen_prompt.txt
with open("gen_prompt.txt", "r") as f:
    prompt_content = f.read().strip()

messages = [{
    "role": "user",
    "content": prompt_content
}]

MODEL_PATH = "../outputs-full/checkpoint-110"  # Updated to use the outputs-full directory

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH, # Using instruct version for better chat performance
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    # chat_template = "mistral", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
    # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    # map_eos_token = True, # Maps <|im_end|> to </s> instead
)

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
    return_tensors = "pt",
    tokenize = True,
).to("cuda")

temperature = 1.0
max_new_tokens = 512  # Increased for full Java program generation

# Generate the response
outputs = model.generate(
    input_ids = inputs,
    temperature = temperature,
    top_p = 0.95,
    top_k = 64,
    max_new_tokens = max_new_tokens,
    use_cache = True,
    do_sample = True,
    pad_token_id = tokenizer.eos_token_id
)

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract only the assistant's response (after the prompt)
prompt_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
assistant_response = generated_text[len(prompt_text):].strip()

print("Generated Java code:")
print(assistant_response)

# Save to Calculator.java file
output_filename = "Calculator.java"
with open(output_filename, "w") as f:
    f.write(assistant_response)

print(f"\nJava code saved to {output_filename}")

import json
import os

config_path = f"{MODEL_PATH}/adapter_config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        adapter_config = json.load(f)
    
    base_model = adapter_config.get("base_model_name_or_path", "Not found")
    print(f"\nBase model: {base_model}")
else:
    print(f"\nAdapter config not found at {config_path}")

print(f"Temperature: {temperature}")
print(f"Max new tokens: {max_new_tokens}")