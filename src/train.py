from datasets import load_dataset

# dataset = load_dataset("data", split="train")
dataset = load_dataset("json", data_files="data/instruction.json", split="train")
print(str(dataset[2]))