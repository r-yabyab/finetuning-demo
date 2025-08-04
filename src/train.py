# from datasets import load_dataset

# # dataset = load_dataset("data", split="train")
# # dataset = load_dataset("json", data_files="data/instruction.json", split="train")
# # dataset = load_dataset("pookie3000/donald_trump_interviews", split="train")
# dataset = load_dataset("json", data_files="/home/ubuntu/finetuning-demo/src/data/instruction-roles-grouped.json", split = "train")

# print(str(dataset[:10]))

from datasets import load_dataset
squad = load_dataset("squad", split="train[:1]")
# squad = squad.train_test_split(test_size=0.2)

print(str(squad[:2000]))