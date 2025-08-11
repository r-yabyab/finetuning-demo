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

# Configure your model path here - change this to your desired model
MODEL_PATH = "output-30"  # Change this to your preferred model path
OUTPUT_PREFIX = "single_model"  # Prefix for output files

print(f"Loading model from {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

tokenizer = get_chat_template(tokenizer, chat_template="llama")

def generate_response(model, tokenizer, messages):
    """Generate a response from the model given a list of messages."""
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

print("Starting inference on test prompts...")

for idx, prompt in enumerate(java_test_prompts, start=1):
    print(f"\n=== Running test prompt #{idx} ===")
    print(f"Prompt: {prompt['content']}")
    
    output = generate_response(model, tokenizer, [prompt])
    
    # Save output to file
    output_file = f"{OUTPUT_PREFIX}_test{idx}.java"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output)
    
    print(f"Saved output for test #{idx} to {output_file}")

print(f"\nAll test prompts processed and outputs saved with prefix '{OUTPUT_PREFIX}'.")
