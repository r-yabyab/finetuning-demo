import pandas as pd

df = pd.read_csv('./data/data.txt')

print(df.head())

df_train=df.sample(frac=0.80,random_state=200)
df_eval=df.drop(df_train.index)

df_train.to_json("./data/data_chunk_train.jsonl", orient="records", lines=True)
df_eval.to_json("./data/data_chunk_eval.jsonl", orient="records", lines=True)