import os
import torch
import logging
import evaluate
import random
import argparse
import numpy as np
from itertools import product


import datetime


from typing import List, Union, Optional  
import torch.nn as nn
import torch.distributed as dist  
from torch.utils.checkpoint import checkpoint
import torch.utils.checkpoint 
import torch.multiprocessing as mp  
from torch.nn.parallel import DistributedDataParallel as DDP 

from torch.utils.data import DataLoader, DistributedSampler

from src.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from datasets import (
    Dataset,
    load_dataset
)

from transformers import (
    AutoModel,
    AutoTokenizer,
    RobertaTokenizerFast,
    GPT2TokenizerFast,
    BertTokenizerFast,
    T5TokenizerFast,
    Qwen2TokenizerFast,
    AutoConfig,
    BertTokenizerFast,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    
    BertForSequenceClassification,
    Qwen2ForCausalLM,
    Qwen2ForSequenceClassification,
    RobertaForSequenceClassification,
    GPT2ForSequenceClassification,
    
    BitsAndBytesConfig,
)


from peft import (
    PeftModel,
    
)

from accelerate import (
    init_empty_weights, 
    load_checkpoint_and_dispatch  
)

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

import os  
import sys

import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
sys.path.append("../")
from src.configs.config import MODEL_CONFIG, MODEL_PATH


# 设置环境变量以启用显存优化  
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True" 

from dataclasses import dataclass
import psutil  
import pynvml

@dataclass
class SFTArguments:
    def __init__(self):
        self.model_name = MODEL_PATH
        self.output_dir = "output"
        self.device = "cuda"
        self.device_map = "auto"
        self.local_rank = -1


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Trainer Arguments")
    parser.add_argument("--model_name", type=str, required=True, help="基础模型名称", default = MODEL_PATH )
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录", default = "output" )
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument("--device_map", type=str, default="auto", help="设备映射策略")
    
    # 添加 DeepSpeed 所需的参数  
    parser.add_argument("--local_rank", type=int, default=-1)  
    parser.add_argument("--deepspeed", type=str, default=None) 
    
    return parser.parse_args()




def check_deepspeed_env():  
    """检查DeepSpeed环境"""  
    import pkg_resources  
    import torch  
    
    print("\n=== Environment Check ===")  
    print(f"PyTorch version: {torch.__version__}")  
    print(f"CUDA available: {torch.cuda.is_available()}")  
    if torch.cuda.is_available():  
        print(f"CUDA version: {torch.version.cuda}")  
        print(f"GPU count: {torch.cuda.device_count()}")  
    
    try:  
        ds_version = pkg_resources.get_distribution('deepspeed').version  
        print(f"DeepSpeed version: {ds_version}")  
    except pkg_resources.DistributionNotFound:  
        print("DeepSpeed not found!")  
        
    return True




def check_deepspeed_config(training_args):
    # 1. 检查环境变量  
    print("Environment DEEPSPEED_CONFIG:", os.environ.get('DEEPSPEED_CONFIG'))  
    
    # 2. 检查 TrainingArguments 中的配置  
    print("TrainingArguments deepspeed config:", training_args.deepspeed)


    # 如果想看具体内容  
    if training_args.deepspeed:  
        import json  
        try:  
            with open(training_args.deepspeed, 'r') as f:  
                ds_config = json.load(f)  
            print("DeepSpeed config content:", json.dumps(ds_config, indent=2))  
        except Exception as e:  
            print(f"Error reading deepspeed config: {e}") 





def setup_cuda_debug_environment():  
    """设置调试环境"""  
    import torch  
    
    torch.backends.cuda.matmul.allow_tf32 = False  # 禁用TF32以获得更精确的错误信息  
    torch.backends.cudnn.deterministic = True      # 使用确定性算法  
    torch.backends.cudnn.benchmark = False         # 禁用基准测试优化  
    
    import os  
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  
    os.environ['TORCH_USE_CUDA_DSA'] = '1' 
    
    print("=== Debug Environment Setup ===")  
    print(f"CUDA available: {torch.cuda.is_available()}")  
    print(f"CUDA version: {torch.version.cuda}")  
    print(f"PyTorch version: {torch.__version__}")  
    print(f"TORCH_USE_CUDA_DSA: {os.getenv('TORCH_USE_CUDA_DSA')}")  
    print(f"Current device: {torch.cuda.current_device()}")  
    print(f"Device name: {torch.cuda.get_device_name()}")  
    print("===========================")  
    
    



def load_split_model(model_name_or_path):  
    """  
    在加载前就将模型分割到多个GPU上  
    """  
    # 1. 首先获取模型配置  
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)  
    
    # 2. 计算每个GPU应该分配的最大内存  
    max_memory = {  
        0: "20GiB",  # GPU 0 最大使用15GB  
        1: "20GiB",  # GPU 1 最大使用15GB  
        "cpu": "15GB"  # CPU 内存预留30GB  
    }  
    
    # 3. 使用 device_map="auto" 让 Accelerate 自动决定最优分配  
    try:  
        # 方法1：直接加载并自动分配  
        model = AutoModelForCausalLM.from_pretrained(  
            model_name_or_path,  
            device_map="auto",  
            max_memory=max_memory,  
            torch_dtype=torch.bfloat16,  
            trust_remote_code=True,  
            use_flash_attention_2=True  
        )  
        
    except Exception as e:  
        print(f"Direct loading failed, trying alternative method: {str(e)}")  
        
        # 方法2：使用空权重初始化后再加载  
        with init_empty_weights():  
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)  
            
        model = load_checkpoint_and_dispatch(  
            model,  
            model_name_or_path,  
            device_map="auto",  
            max_memory=max_memory,  
            no_split_module_classes=["GPTBlock"],  # 适当设置不可分割的模块  
            dtype=torch.bfloat16,  
            offload_folder="offload"  # 设置权重卸载目录  
        )  
    
    return model  






def load_qwen_in_4bit(  
    model_name,  
    load_in_4bit=True,  
    use_flash_attention=False  
):  
    """  
    使用更激进的优化方案加载Qwen模型  
    
    Args:  
        model_name: 模型名称或路径  
        load_in_4bit: 是否使用4-bit量化  
        use_flash_attention: 是否使用Flash Attention 2  
    """  
    # 初始化tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(  
        model_name,  
        trust_remote_code=True  
    )  

    torch.cuda.empty_cache()  
    
    # 配置4-bit量化参数  
    quantization_config = BitsAndBytesConfig(  
        load_in_4bit=True,  
        bnb_4bit_compute_dtype=torch.bfloat16,    # torch.float16,  
        bnb_4bit_use_double_quant=True,  
        bnb_4bit_quant_type="nf4",  # 使用nested float 4 量化  
        bnb_4bit_quant_storage=torch.bfloat16,  # 存储时也使用4-bit 
    )  

    max_memory = {}  
    for i in range(torch.cuda.device_count()):  
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  
        # 预留2GB给系统  
        max_memory[i] = f"{int(total_mem - 2)}GiB"  
    max_memory["cpu"] = "15GB"  # CPU内存预留  

    print("Max memory configuration:", max_memory)
    
    # 设置模型加载配置  
    model_kwargs = {  
        "torch_dtype": torch.bfloat16,  
        "trust_remote_code": True,  
        # "device_map": "auto",  
        "quantization_config": quantization_config,  
        "max_memory": max_memory,  # 限制GPU显存使用  
        "offload_folder": "offload",  # 设置模型权重卸载目录  
        # 启用梯度检查点以节省显存  
        # "use_gradient_checkpointing": True,  
    }  
    
    if use_flash_attention:  
        model_kwargs["use_flash_attention_2"] = True  
    
    # 加载模型
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
        low_cpu_mem_usage=True,
    )

    torch.cuda.empty_cache()

    # 在模型加载后设置gradient checkpointing
    # gradient_checkpointing_enable() 已经正确处理了梯度检查点的设置
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    elif hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()

    # 禁用缓存
    model.config.use_cache = False

    # 强制进行垃圾回收
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return model



def load_qwen(
    model_name,  
    use_flash_attention=False  
):
    
     # 初始化tokenizer  
    tokenizer = AutoTokenizer.from_pretrained(  
        model_name,  
        trust_remote_code=True  
    )  

    torch.cuda.empty_cache()  
    

    max_memory = {}  
    for i in range(torch.cuda.device_count()):  
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  
        # 预留2GB给系统  
        max_memory[i] = f"{int(total_mem - 2)}GiB"  
    max_memory["cpu"] = "15GB"  # CPU内存预留  

    print("Max memory configuration:", max_memory)
    
    # 设置模型加载配置  
    model_kwargs = {  
        "torch_dtype": torch.float16,  # 防止和 deepspeed 配置冲突
        "trust_remote_code": True,  
        # "device_map": "auto",  
        "max_memory": max_memory,  # 限制GPU显存使用  
        # "offload_folder": "offload",  # 设置模型权重卸载目录  
        # 启用梯度检查点以节省显存  
        # "use_gradient_checkpointing": True,  
    }  
    
    if use_flash_attention:  
        model_kwargs["use_flash_attention_2"] = True  
    
    # 加载模型
    model = Qwen2ForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
        low_cpu_mem_usage=True,
    )

    torch.cuda.empty_cache()

    # 在模型加载后设置gradient checkpointing
    # gradient_checkpointing_enable() 已经正确处理了梯度检查点的设置
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    elif hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()

    # 禁用缓存
    model.config.use_cache = False

    # 强制进行垃圾回收
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return model
    
    
    
    
    
def monitor_memory():  
    """监控GPU和CPU内存使用"""  

    try:  
        # 初始化 NVML  
        pynvml.nvmlInit()  
        
        print("\nGPU Memory Usage:")  
        # 获取所有GPU的信息  
        for i in range(torch.cuda.device_count()):  
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)  
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)  
            
            print(f"\nGPU {i}:")  
            print(f"Total memory: {info.total / 1024**3:.2f} GB")  
            print(f"Used memory: {info.used / 1024**3:.2f} GB")  
            print(f"Free memory: {info.free / 1024**3:.2f} GB")  
            
            # 获取GPU利用率  
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)  
            print(f"GPU Utilization: {utilization.gpu}%")  
            print(f"Memory Utilization: {utilization.memory}%")  
            
        pynvml.nvmlShutdown()  
        
    except Exception as e:  
        print(f"Error monitoring GPU memory: {str(e)}")  



# 更简单的版本，只使用 torch  
def print_gpu_memory():  
    """使用 torch 打印 GPU 内存使用情况"""  
    for i in range(torch.cuda.device_count()):  
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")  
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1024**3:.2f} GB")  
        print(f"Memory Reserved: {torch.cuda.memory_reserved(i)/1024**3:.2f} GB")  
        print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i)/1024**3:.2f} GB")  

# 完整的监控函数（包括CPU内存）  
def monitor_system_resources():  
    """监控系统资源使用情况"""  
    import psutil  
    
    # CPU 使用情况  
    print("\nCPU Usage:")  
    print(f"CPU Usage: {psutil.cpu_percent()}%")  
    
    # 内存使用情况  
    memory = psutil.virtual_memory()  
    print("\nSystem Memory:")  
    print(f"Total: {memory.total/1024**3:.2f} GB")  
    print(f"Available: {memory.available/1024**3:.2f} GB")  
    print(f"Used: {memory.used/1024**3:.2f} GB")  
    print(f"Percentage: {memory.percent}%")  
    
    # GPU 使用情况  
    print_gpu_memory()  
    
    
    
def get_model_name_using_model(model):
    '''
    
    use the model object's config file to retrieve the model name, e.g. bert-base-uncased
    '''
    
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
        
    config = model.config  
    # 尝试直接获取模型的名称  
    if hasattr(config, 'name_or_path') and config.name_or_path is not None:  
        # 使用 os.path.basename 提取路径中的模型名称  
        model_name = os.path.basename(config.name_or_path)  
        return model_name  
    # 根据模型类型和隐藏层大小推断模型名称  
    if config.model_type == "bert":  
        if config.hidden_size == 768:  
            return "bert-base-uncased"  
        elif config.hidden_size == 1024:  
            return "bert-large-uncased"  
    elif config.model_type == "roberta":  
        if config.hidden_size == 768:  
            return "roberta-base"  
        elif config.hidden_size == 1024:  
            return "roberta-large"  
    elif config.model_type == "llama":  
        if config.hidden_size == 4096:  
            return "meta-llama/Llama-2-13b-hf"  
        elif config.hidden_size == 5120:  
            return "meta-llama/Llama-2-70b-hf"  
    elif config.model_type == "qwen2":  
        if config.hidden_size == 896:  
            return "Qwen2.5-0.5B"  
        elif config.hidden_size == 1536:  
            return "Qwen2.5-1.5B"  
        elif config.hidden_size == 2048:
            return "Qwen2.5-3B"
        elif config.hidden_size == 3584:
            return "Qwen2.5-7B"
    elif config.model_type == "gpt2":
        if config.n_embd == 768:
            return "gpt2"
        elif config.n_embd == 1024:
            return "gpt2-medium"
        elif config.n_embd == 1280:
            return "gpt2-large"
        elif config.n_embd== 1600:
            return "gpt2-xl"
    else:  
        # 无法匹配已知模型，返回未知模型提示  
        raise ValueError("unknown model, please check your config, it should be [bert | llama | qwen2]") 

def get_base_model_using_model(model):
    """
    获取模型包装器的底层的基座模型对象

    """
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
    
        # 获取模型类型  
    model_type = type(model)

    if hasattr(model, "config"):
        config = model.config
    else:
        raise RuntimeError("This model object does not have a config file, check again~~~")

    try:
        if isinstance(model, AutoModel):
            model = model
        elif isinstance(model, PeftModel):  
            print("Info: Model is a PeftModel, getting the base model")  
            model = model.get_base_model() 
        elif isinstance(model, AutoModelForSequenceClassification):
            model = model.base_model
        elif isinstance(model, BertForSequenceClassification):
            model = model.bert
        elif isinstance(model, RobertaForSequenceClassification):
            model = model.roberta
        elif isinstance(model, Qwen2ForSequenceClassification):
            model = model.model
        elif isinstance(model, GPT2ForSequenceClassification):
            model = model.transformer
         
        else:
            raise ValueError(f"the passed model object is not either SequenceClassification model or AutoModel \
                The current model type = {model_type}")

    except:
        raise ValueError(f"Extracting base model failed, your current model type is {model_type}")

    return model

def get_hidden_size_using_config():
    pass

def get_hidden_size_by_model_name(model_name:str):
    pass

def get_hidden_size_using_model(model):
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module
    
        # 获取模型类型  
    model_type = type(model)
    
    model_name = get_model_name_using_model(model)

    if hasattr(model, "config"):
        config = model.config
    else:
        raise RuntimeError("This model object does not have a config file, check again~~~")
    
    if hasattr(config,'hidden_size'):
        hidden_size = config.hidden_size
    elif hasattr(config, 'd_model'): # t5
        hidden_size = config.d_model
    elif hasattr(config, 'n_embd'): # gpt2
        hidden_size = config.n_embd
    else:
        raise ValueError(f"the passed model object does not have the attribute `hidden_size` \
            The current model type = {model_type}")
    print(f"model:{model_name}'s hidden_size = {hidden_size}")
    return hidden_size

def get_classifier_from_model(model)-> nn.Module:  
    """  
    获取预训练模型的分类器  
    
    Args:  
        model : AutoModelForSequenceClassification or BertForSequenceClassification
        num_labels (int): 分类标签数量  
    
    Returns:  
        nn.Module: 分类器模块  
    """  
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module

    # 获取分类器  
    if hasattr(model, 'classifier'):  
        # BERT、RoBERTa 等模型的分类器  
        classifier = model.classifier  
        print(f"分类器类型: {type(classifier).__name__}")
        
    elif hasattr(model, 'score'):   # qwen2, gpt2
        # 某些模型可能使用 score 作为分类器名称  
        classifier = model.score  
    else:  
        raise AttributeError("无法找到模型的分类器层")  
    
    # 打印分类器信息  
    print("分类器结构：")  
    print(classifier)  
    
    in_features=None
    out_features=None
    if hasattr(classifier, 'dense'):
        in_features = classifier.dense.in_features
        print("这是一个RobertaClassificationHead，需要通过dense层获取输入维度")
    else:
        in_features = classifier.in_features
        
    if hasattr(classifier, 'out_proj'):
        out_features = classifier.out_proj.out_features
        print("这是一个RobertaClassificationHead，需要通过out_proj层获取输出维度")
    else:
        out_features = classifier.out_features
        
        
    print(f"\n分类器输入维度: {in_features}")  
    print(f"分类器输出维度: {out_features}") 
    
    # 示例：直接使用分类器进行前向传播  
    # batch_size = 4  
    # hidden_size = classifier.in_features  
    
    # 模拟来自BERT的特征输出  
    # dummy_features = torch.randn(batch_size, hidden_size)  
    
    # # 直接使用分类器进行预测  
    # with torch.no_grad():  
    #     outputs = classifier(dummy_features)  
        
    # print(f"\n分类器输出形状: {outputs.shape}")  
    # print("分类器输出示例：")  
    # print(outputs)   
    
    
    print("\n分类器的可训练参数：")  
    for name, param in classifier.named_parameters():  
        print(f"{name}: {param.shape}")  
        
    return classifier 

def get_max_length_from_model(model):  
    """  
    获取模型的最大序列长度  
    model: 既可以base model， 也可以是特定任务model
    
    """  
    if isinstance(model,str):
        model = AutoModel.from_pretrained(model)
    
    # 处理被Accelerator(DDP)包装的模型  
    if hasattr(model, "module"):  
        print("This model is wrapped by Accelerator(DDP), we use model.module")  
        model = model.module  
        
    if hasattr(model, "config"):
        config = model.config  
    else:
        raise ValueError('your model object is not properly defined ... since we can not find a `config` attribute')
    
    # 首先尝试从config中直接获取max_position_embeddings  
    if hasattr(config, 'max_position_embeddings'):  
        return config.max_position_embeddings  
    
    # 如果没有max_position_embeddings，尝试获取max_sequence_length  
    elif hasattr(config, 'max_sequence_length'):  
        return config.max_sequence_length  
    
    elif hasattr(config, 'n_positions'):  
        return config.n_positions
    
    elif hasattr(config, 'n_ctx'):  
        return config.n_ctx
    
    else:
        raise ValueError("Error model object, please check your config, it should have either [max_position_embeddings | max_sequence_length]") 

def get_classifier(model:AutoModelForSequenceClassification):
    """
    获取预训练模型的分类器
    """
    # 处理被Accelerator(DDP)包装的模型
    if hasattr(model, "module"):
        print("This model is wrapped by Accelerator(DDP), we use model.module")
        model = model.module

    classifier = None
    # 获取分类器
    if hasattr(model, 'classifier'):
        # BERT、RoBERTa 等模型的分类器
        classifier = model.classifier
        print(f"分类器类型: {type(classifier).__name__}")
    elif hasattr(model, 'score'):
        # 某些模型可能使用 score 作为分类器名称
        classifier = model.score
        
    else:
        raise AttributeError("无法找到模型的分类器层")
    
    return classifier

def print_model_info(model:AutoModelForSequenceClassification):  
    """打印模型的详细信息"""  
    
    
    print("\n=== Model Classification Head Information ===")  
    
    # 1. 打印分类器的结构  
    print("\nClassifier Architecture:")  
    if hasattr(model,'classifier'):
        print(model.classifier)  
    elif hasattr(model,'score'):
        print(model.score)
    
    # 2. 打印分类器中dense层的权重形状 
    if hasattr(model,'classifier') and hasattr(model.classifier, 'dense'): 
        dense_weight = model.classifier.dense.weight  
        print("\nDense Layer Weight Shape:", dense_weight.shape)  
    
    # 3. 打印分类器中out_proj层的权重形状  
    if hasattr(model,'classifier') and hasattr(model.classifier, 'out_proj'):
        out_proj_weight = model.classifier.out_proj.weight  
        print("Output Projection Weight Shape:", out_proj_weight.shape)  
    
    # 4. 打印整个模型的参数数量  
    total_params = sum(p.numel() for p in model.parameters())  
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  
    print(f"\nTotal Parameters: {total_params:,}")  
    print(f"Trainable Parameters: {trainable_params:,}")  
    print(f"Percentage of Trainable Parameters: {100 * trainable_params / total_params:.2f}%") 

