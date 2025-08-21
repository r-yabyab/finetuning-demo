from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap

# Load the model and tokenizer first
max_seq_length = 2048
dtype = None
load_in_4bit = True

# Load your fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs-cpt/checkpoint-14",  # Path to your fine-tuned model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Prepare for inference
FastLanguageModel.for_inference(model)

# Create text streamer with the tokenizer
text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=False)
max_print_width = 100

# Prepare inputs
inputs = tokenizer(
    [
        "Summarize Cannabis use and dimensions of psychosis in a nonclinical population of female subjects."
    ], return_tensors="pt").to("cuda")

generation_kwargs = dict(
    **inputs,  # Unpack inputs properly
    streamer=text_streamer,
    max_new_tokens=256,
    use_cache=True,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id,
)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

print("Generated text:")
length = 0
for j, new_text in enumerate(text_streamer):
    if j == 0:
        wrapped_text = textwrap.wrap(new_text, width=max_print_width)
        if wrapped_text:
            length = len(wrapped_text[-1])
            wrapped_text = "\n".join(wrapped_text)
            print(wrapped_text, end="")
    else:
        length += len(new_text)
        if length >= max_print_width:
            length = 0
            print()
        print(new_text, end="")

thread.join()  # Wait for generation to complete
print("\n")  # Add final newline