import transformers

from datasets import load_dataset_builder

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GemmaTokenizer
from transformers import pipeline
from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training


from trl import SFTTrainer



import torch
import os

# 查看代理设置
print(os.environ.get('http_proxy'))
print(os.environ.get('https_proxy'))
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
print(os.environ.get('http_proxy'))
print(os.environ.get('https_proxy'))

os.environ['HF_TOKEN']="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"


print("Is GPU available? ",torch.cuda.is_available())


model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo


tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) # 使用快速分词器

model.eval() # model in evaluation mode

# 写提示词
comment = "User: Help me plan a path from Beijing to New York"
# INST: 指令
prompt = f'''[INST]{comment}[INST]'''




'''
测试模型
'''
# # 分词
# inputs = tokenizer(prompt, return_tensors="pt")


# # input_ids: 分词过后的文本对应的数字序列
# outputs = model.generate(input_ids = inputs["input_ids"].to("cuda"), max_new_tokens=300)

# # 将数字序列解码回文本， [0]表示取第一个解码结果
# print(tokenizer.batch_decode(outputs)[0])






# 提示工程
intstructions_string = f"""
    Act as an tour guide from travel agencies, and you also have a 10 years' working experience.
    You should base on the given context and reply to the questions of the customer who talks to you and want to seek advice for traveling suggestions.
    Reply to the questions considering to the traveling consulting history that I provided, and try to mimic the speaking style of the traveling agency, as well as your occupation and your talents.

Please respond to the following comment.
"""

prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

prompt = prompt_template(comment)
print(prompt)




# tokenize input
inputs = tokenizer(prompt, return_tensors="pt")

# generate output
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)


print(tokenizer.batch_decode(outputs)[0])

     


'''
Prepare Model for Training
'''
model.train() # model in training mode (dropout modules are activated)

# enable gradient check pointing
model.gradient_checkpointing_enable()

# enable quantized training
model = prepare_model_for_kbit_training(model)


# LoRA config
config = LoraConfig(
    r=8,
    lora_alpha=32, 
    target_modules=["q_proj"], # query projection, 一个线性层
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# LoRA trainable version of model
model = get_peft_model(model, config)

# trainable parameter count
model.print_trainable_parameters()




# Preparing Training Dataset
# load dataset

# dataset_id = "Binaryy/travel_sample_extended"
'''
data 是一个 datasets.DatasetDict 类型的变量。
load_dataset 函数从 Hugging Face 的数据集库中加载指定的数据集，
并返回一个 DatasetDict 对象。

DatasetDict 是一个字典，它的键通常是数据集的分割名称（如 'train'，'validation'，'test' 等），
值是对应分割的 datasets.Dataset 对象。每个 Dataset 对象包含了该分割的所有样本。
'''

# load dataset
data = load_dataset("shawhin/shawgpt-youtube-comments")



# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["example"]

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
tokenized_data = data.map(tokenize_function, batched=True)


# setting pad token
tokenizer.pad_token = tokenizer.eos_token
# data collator
data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)



# hyperparameters
lr = 2e-4
batch_size = 4
num_epochs = 10

# define training arguments
training_args = transformers.TrainingArguments(
    output_dir= "mistral-7b-qlora-ft",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    fp16=True,
    optim="paged_adamw_8bit",
)


# configure trainer
trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data["train"][20:],
    eval_dataset=tokenized_data["train"][:20],
    args=training_args,
    data_collator=data_collator
)

# train model
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# renable warnings
model.config.use_cache = True


# push model to hub
from huggingface_hub import notebook_login
notebook_login()

# # option 2: key login
# from huggingface_hub import login
# write_key = 'hf_' # paste token here
# login(write_key)

hf_name = 'zhongyah' # your hf username or org name
model_id = hf_name + "/" + "mistral-7b-qlora-ft"


model.push_to_hub(model_id)
trainer.push_to_hub(model_id)





# Load Fine-tuned Model

# # load model from hub
# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM

# model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
# model = AutoModelForCausalLM.from_pretrained(model_name,
#                                              device_map="auto",
#                                              trust_remote_code=False,
#                                              revision="main")

# config = PeftConfig.from_pretrained("shawhin/shawgpt-ft")
# model = PeftModel.from_pretrained(model, "shawhin/shawgpt-ft")

# # load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)




'''
Use Fine-tuned Model
'''

# intstructions_string = f"""ShawGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
# It reacts to feedback aptly and ends responses with its signature '–ShawGPT'. \
# ShawGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
# thus keeping the interaction natural and engaging.

# Please respond to the following comment.
# """
# prompt_template = lambda comment: f'''[INST] {intstructions_string} \n{comment} \n[/INST]'''

# comment = "Great content, thank you!"

# prompt = prompt_template(comment)
# print(prompt)
     


# model.eval()

# inputs = tokenizer(prompt, return_tensors="pt")
# outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)

# print(tokenizer.batch_decode(outputs)[0])


# comment = "What is fat-tailedness?"
# prompt = prompt_template(comment)

# model.eval()
# inputs = tokenizer(prompt, return_tensors="pt")

# outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=280)
# print(tokenizer.batch_decode(outputs)[0])