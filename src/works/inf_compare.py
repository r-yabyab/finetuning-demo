from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

max_seq_length = 512
dtype = None
load_in_4bit = True

messages = [{
    "role": "user",
    # "content": "In Spring Boot, implement a security filter"
}]

BASE_MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
LORA_MODEL_PATH = "outputs/checkpoint-125"  # Your fine-tuned LoRA checkpoint

print("Loading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("Loading LoRA fine-tuned model...")
lora_model, lora_tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

base_tokenizer = get_chat_template(base_tokenizer, chat_template="llama")
lora_tokenizer = get_chat_template(lora_tokenizer, chat_template="llama")

def generate_response(model, tokenizer, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    output_ids = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=300,
        use_cache=False
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("\n=== BASE MODEL OUTPUT ===")
base_output = generate_response(base_model, base_tokenizer, messages)
print("\n=== LORA MODEL OUTPUT ===")
lora_output = generate_response(lora_model, lora_tokenizer, messages)