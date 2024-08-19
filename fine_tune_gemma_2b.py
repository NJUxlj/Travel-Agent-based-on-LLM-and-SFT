import transformers

from datasets import load_dataset_builder

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig


from trl import SFTTrainer


# modelscope
from modelscope.pipelines import pipeline
# word_segmentation = pipeline('word-segmentation',model='damo/nlp_structbert_word-segmentation_chinese-base')



import torch
import os

# huggingface的代理
# 查看代理设置
# print(os.environ.get('http_proxy'))
# print(os.environ.get('https_proxy'))
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
# print(os.environ.get('http_proxy'))
# print(os.environ.get('https_proxy'))

# os.environ['HF_TOKEN']="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# 我们这里用modelscope

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Is GPU available? ",torch.cuda.is_available())




# 加载数据集
from modelscope.msdatasets import MsDataset
from datasets import load_dataset



train_split = load_dataset("json",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv", split="train")

eval_split = load_dataset("json",data_files="datasets/bitext-travel-llm-chatbot-training-dataset.csv", split="validation")

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)




model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer_id = 'distilbert-base-uncased'

# tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, token=os.environ['HF_TOKEN'])
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])








# text = "Quote: Imagination is more"
# device = "cuda:0"
# inputs = tokenizer(text, return_tensors="pt").to(device)

# outputs = model.generate(**inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# data = load_dataset("Abirate/english_quotes")

data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)



def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}<eos>"
    return [text]



trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
            # 每个设备上的训练批次大小为1
            per_device_train_batch_size=1,
            # 梯度累积步数=4：每进行4次前向传播累积一次梯度后再进行一次反向传播，有助于处理内存限制
            gradient_accumulation_steps=4,
            # 在训练初期，学习率会逐渐增加至设定值，这里的预热步数是2
            warmup_steps=2,
            # 最多执行10步训练
            max_steps=10,
            # 
            learning_rate=2e-4,
            # 使用半精度16位浮点数进行训练
            fp16=True,
            logging_steps=1,
            output_dir="outputs",
            optim="paged_adamw_8bit"
        ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()

# 保存模型
output_path = "saves/gemma-2b/lora/sft"
model.save_pretrained(output_path)

# 保存tokenizer
tokenizer.save_pretrained(output_path)


text = "Can you plan me a path from Beijing to Los Angles?"
# device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))






# # push model to hub
# from huggingface_hub import notebook_login
# notebook_login()

# # # option 2: key login
# # from huggingface_hub import login
# # write_key = 'hf_' # paste token here
# # login(write_key)

# hf_name = 'zhongyah' # your hf username or org name
# model_id = hf_name + "/" + "gemma-2b-lora-ft"


# model.push_to_hub(model_id)
# trainer.push_to_hub(model_id)
