from unsloth import FastLanguageModel
from transformers import TextStreamer
import json
import os

max_seq_length = 512 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

MODEL_PATH = "outputs-full/checkpoint-926"

# Load model once at startup
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama",
)

# Configuration
temperature = 0.1
max_new_tokens = 200
text_streamer = TextStreamer(tokenizer, skip_prompt = True)

# Print model info once
config_path = f"{MODEL_PATH}/adapter_config.json"
if os.path.exists(config_path):
    with open(config_path, 'r') as f:
        adapter_config = json.load(f)
    
    base_model = adapter_config.get("base_model_name_or_path", "Not found")
    print(f"Base model: {base_model}")
else:
    print(f"Adapter config not found at {config_path}")

print(f"Temperature: {temperature}")
print(f"Max new tokens: {max_new_tokens}")
print("\nInteractive mode started. Type 'quit' or 'exit' to stop.\n")

# Interactive loop
while True:
    try:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Create fresh message for each prompt (no history)
        messages = [{
            "role": "user",
            "content": user_input
        }]
        
        # Process the input
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True,
            return_tensors = "pt",
            tokenize = True,
        ).to("cuda")
        
        print("Assistant: ", end="")
        
        # Generate response
        _ = model.generate(
            input_ids = inputs,
            streamer = text_streamer,
            temperature = temperature,
            top_p = 0.95,
            top_k = 64,
            max_new_tokens = max_new_tokens,
            use_cache = True
        )
        
        print("\n" + "-"*50 + "\n")  # Separator between conversations
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue