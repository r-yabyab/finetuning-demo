from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

max_seq_length = 512
dtype = None
load_in_4bit = True

java_test_prompts = [
    {"role": "user", "content": "Write a Java class called `Person` with private fields `name` and `age`, a constructor, and getter methods."},
    {"role": "user", "content": "Create a Java class `Employee` that extends `Person` and adds a `salary` field with appropriate constructor and methods."},
    {"role": "user", "content": "Write a Java method that takes a `List<String>` and returns a new list with all strings converted to uppercase."},
    {"role": "user", "content": "Implement a Java try-catch block that handles `ArithmeticException` when dividing two integers."},
    {"role": "user", "content": "Write a Java program that starts two threads, each printing numbers from 1 to 5 with a short delay."},
    {"role": "user", "content": "Create a Java interface called `Shape` with a method `double area()`, and implement it in classes `Circle` and `Rectangle`."},
    {"role": "user", "content": "Write a Java program to read a text file line by line and print each line to the console."},
    {"role": "user", "content": "Implement a Java method that uses recursion to compute the factorial of a given integer."},
    {"role": "user", "content": "Write a Java method that sorts an array of integers using bubble sort."},
    {"role": "user", "content": "Create a Java enum `DayOfWeek` representing days Monday to Sunday."}
]

BASE_MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
LORA_PATH_1 = "output-30"  # First LoRA checkpoint path
LORA_PATH_2 = "output-60"  # Second LoRA checkpoint path

print("Loading base model...")
base_model, base_tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("Loading first LoRA fine-tuned model...")
lora_model_1, lora_tokenizer_1 = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH_1,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print("Loading second LoRA fine-tuned model...")
lora_model_2, lora_tokenizer_2 = FastLanguageModel.from_pretrained(
    model_name=LORA_PATH_2,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

base_tokenizer = get_chat_template(base_tokenizer, chat_template="llama")
lora_tokenizer_1 = get_chat_template(lora_tokenizer_1, chat_template="llama")
lora_tokenizer_2 = get_chat_template(lora_tokenizer_2, chat_template="llama")

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

for idx, prompt in enumerate(java_test_prompts, start=1):
    print(f"\n=== Running test prompt #{idx} ===")
    output_1 = generate_response(lora_model_1, lora_tokenizer_1, [prompt])
    output_2 = generate_response(lora_model_2, lora_tokenizer_2, [prompt])

    file_1 = f"lora_output_30_test{idx}.java"
    file_2 = f"lora_output_60_test{idx}.java"

    with open(file_1, "w", encoding="utf-8") as f1:
        f1.write(output_1)
    with open(file_2, "w", encoding="utf-8") as f2:
        f2.write(output_2)

    print(f"Saved outputs for test #{idx} to {file_1} and {file_2}")

print("\nAll test prompts processed and outputs saved.")