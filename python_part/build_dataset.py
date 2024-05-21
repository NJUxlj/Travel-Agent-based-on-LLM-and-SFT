from datasets import load_dataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer

import os
# 查看代理设置
print(os.environ.get('http_proxy'))
print(os.environ.get('https_proxy'))
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
print(os.environ.get('http_proxy'))
print(os.environ.get('https_proxy'))

# dataset_id =  "NLPC-UOM/Travel-Dataset-5000"

dataset_id ="Binaryy/travel_sample_extended"

dataset = load_dataset(dataset_id)


print(dataset.keys())

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")




# def encode(examples):
#     return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")


# dataset = dataset.map(encode, batched=True)

# print(dataset[0])

# dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

# print(dataset[0])

