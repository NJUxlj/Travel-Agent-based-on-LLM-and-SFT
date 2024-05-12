import transformers

from datasets import load_dataset_builder

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig

import torch
import os

# 查看代理设置
print(os.environ.get('http_proxy'))
print(os.environ.get('https_proxy'))
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
print(os.environ.get('http_proxy'))
print(os.environ.get('https_proxy'))


print("Is GPU available? ",torch.cuda.is_available())



