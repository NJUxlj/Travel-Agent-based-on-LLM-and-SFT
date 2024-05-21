import transformers

from datasets import load_dataset_builder

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GemmaTokenizer
from transformers import pipline
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

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)




model_id = "google/gemma-7b"


# model_name = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto", # automatically figures out how to best use CPU + GPU for loading model
                                             trust_remote_code=False, # prevents running custom model files on your machine
                                             revision="main") # which version of model to use in repo