from unsloth import FastLanguageModel
import torch
from transformers import TextIteratorStreamer
from threading import Thread
import textwrap

max_seq_length = 512
dtype = None
load_in_4bit = True

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs-self/checkpoint-476",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

FastLanguageModel.for_inference(model)
print("Model loaded successfully!\n")

max_print_width = 100

def generate_response(prompt):
    """Generate response for a given prompt"""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    text_streamer = TextIteratorStreamer(tokenizer, skip_prompt=False)
    
    generation_kwargs = dict(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=110,
        use_cache=True,
        do_sample=True,
        temperature=1.2,  # Increase from 1.0 for more creativity
        top_p=0.9,        # Add nucleus sampling
        top_k=50,         # Add top-k sampling
        repetition_penalty=1.1,  # Penalize repetition
        pad_token_id=tokenizer.eos_token_id,
    )
    
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # print("Generated text:")
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
    
    thread.join()
    print("\n")

# Interactive loop
print("Interactive Model Chat")
print("Type 'quit', 'exit', or 'q' to stop")
print("-" * 50)

while True:
    try:
        user_input = input("\nEnter your prompt: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter a valid prompt.")
            continue
        
        # print("-" * 50)
        generate_response(user_input)
        print("-" * 50)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        break
    except Exception as e:
        print(f"Error occurred: {e}")
        continue