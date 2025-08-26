from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap

# max_seq_length = 2048
max_seq_length = 512
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs-cpt/checkpoint-47",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Prepare for inference
FastLanguageModel.for_inference(model)

text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=False)
max_print_width = 100

inputs = tokenizer(
    [
        "Summarize Cannabis use and dimensions of psychosis in a nonclinical population of female subjects."
    ], return_tensors="pt").to("cuda")

generation_kwargs = dict(
    **inputs,  # Unpack inputs properly
    streamer=text_streamer,
    max_new_tokens=100,
    use_cache=True,
    do_sample=True,
    temperature=0.9,
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
print("\n")