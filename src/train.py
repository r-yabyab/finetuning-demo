from datasets import load_dataset

dataset = load_dataset("pookie3000/donald_trump_interviews", split="train")
print(str(dataset[:10]))