from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

# # how dataset was generated

# # load imdb data
# imdb_dataset = load_dataset("imdb")

# # define subsample size
# N = 1000 
# # generate indexes for random subsample
# rand_idx = np.random.randint(24999, size=N) 

# # extract train and test data
# x_train = imdb_dataset['train'][rand_idx]['text']
# y_train = imdb_dataset['train'][rand_idx]['label']

# x_test = imdb_dataset['test'][rand_idx]['text']
# y_test = imdb_dataset['test'][rand_idx]['label']

# # create new dataset
# dataset = DatasetDict({'train':Dataset.from_dict({'label':y_train,'text':x_train}),
#                              'validation':Dataset.from_dict({'label':y_test,'text':x_test})})

import os

# 代理地址
# os.environ["OPENAI_API_BASE"] = "http://{your proxy url}/v1"
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# load dataset
dataset = load_dataset('shawhin/imdb-truncated')


print(dataset)


model_checkpoint = 'distilbert-base-uncased'

id2label  ={0:"Negative", 1:"Positive"}
label2id  ={"Negative":0, "Positive":1}

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels = 2, id2label=id2label, label2id = label2id
)

print(model)


# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    
    
    # create tokenize function
def tokenize_function(examples):
    '''
    examples 就是数据集的一个样本
    '''
    # extract text
    text = examples["text"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs


# tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(tokenized_dataset)

# create data collator
# 数据整理器， 将输入的数据进行动态填充，使得每个输入的长度都一样
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")


# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}



# Apply untrained model to text

# define list of examples
text_list = ["It was good.", "Not a fan, don't recommend.", "Better than the first one.", "This is not worth watching even once.", "This one is a pass."]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # compute logits
    logits = model(inputs).logits
    # convert logits to label
    # 找到logits中概率最高的标签
    predictions = torch.argmax(logits)

    print(text + " - " + id2label[predictions.tolist()])
    
    
 
# Train Model
peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4, # rank
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['q_lin'])

print(peft_config)

model = get_peft_model(model, peft_config)

model.print_trainable_parameters()


# hyperparameters
lr = 1e-3
batch_size = 4
num_epochs = 10

# define training arguments
training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator, # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# train model
trainer.train()


# Generate prediction

model.to('cpu') # moving to mps for Mac (can alternatively do 'cpu')

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("cpu") # moving to mps for Mac (can alternatively do 'cpu')

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])
    
    
    
    
# 保存模型
model_path = "C:/Users/hzy/Desktop/CPT412/ft-models/bert"
model.save_pretrained(model_path)

# 保存tokenizer
tokenizer.save_pretrained(model_path)