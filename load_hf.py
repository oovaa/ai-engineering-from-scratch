import os

from datasets import load_dataset

#
# os.environ["HF_TOKEN"] = "hf_..."  # set via env or .env
# dataset = load_dataset(
#     "wikimedia/wikipedia", "20220301.en", split="train", streaming=True
# )
#
# for i, example in enumerate(dataset):
#     print(example["title"])
#     if i >= 4:
#         break


dataset = load_dataset("stanfordnlp/imdb", split="train")

dataset.to_csv("imdb_train.csv")
dataset.to_json("imdb_train.json")
dataset.to_parquet("imdb_train.parquet")
