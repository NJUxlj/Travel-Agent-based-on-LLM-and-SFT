from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
)

from src.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from src.configs.config import (
    REWARD_MODEL_PATH,
    MODEL_PATH,
    SFT_MODEL_PATH,
    PPO_MODEL_PATH,
    DPO_DATA_PATH,
    CACHED_DPO_DATA_PATH,
    DPO_MODEL_PATH
)


from peft import LoraConfig, get_peft_model
from src.finetune.base_trainer import (
    init_model_and_tokenizer,
    create_default_bnb_config,
    create_default_lora_config,
)
from peft.peft_model import PeftModel, PeftModelForCausalLM
from datasets import load_dataset, load_from_disk, DatasetDict
# from trl import DPOTrainer  
# import deepspeed  
from deepspeed import DeepSpeedEngine 



from dataclasses import dataclass

@dataclass
class DPOTrainingConfig:
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_grad_norm: float = 0.3
    num_train_epochs: int = 3


class DPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "chosen_labels": self.data["chosen_labels"][idx],  
            "rejected_labels": self.data["rejected_labels"][idx]  
        }  
        
        
class DPOTrainer(Trainer):
    def __init__(self, ref_model, beta=0.1, **kwargs):  
        super().__init__(**kwargs)  
        self.ref_model = ref_model  
        self.beta = beta  
        
        self.device = self.ref_model.device
        
        
    def compute_loss(self, model, inputs, num_items_in_batch=None):  
        """核心DPO损失计算"""  
        
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        # 前向传播获取logits  
        outputs = model(  
            input_ids=inputs["input_ids"],  
            attention_mask=inputs["attention_mask"]  
        )  
        
        # 获取chosen和rejected的log概率  
        chosen_log_probs = self._get_log_probs(outputs.logits, inputs["chosen_labels"])  
        rejected_log_probs = self._get_log_probs(outputs.logits, inputs["rejected_labels"])  
        
        # 计算参考模型的log概率  
        with torch.no_grad():  
            ref_outputs = self.ref_model(  
                input_ids=inputs["input_ids"],  
                attention_mask=inputs["attention_mask"]  
            )  
            ref_chosen_log_probs = self._get_log_probs(ref_outputs.logits, inputs["chosen_labels"])  # 计算 log(π(y|x))
            ref_rejected_log_probs = self._get_log_probs(ref_outputs.logits, inputs["rejected_labels"])  
        
        # 计算DPO损失  L = -log(σ(r(x,y_w) - r(x,y_l))) , where r(x,y) = β * log(π(y|x)/π_ref(y|x)) = β * [log(π(y|x)) - log(π_ref(y|x))]
        losses = -F.logsigmoid(  
            self.beta * (  
                (chosen_log_probs - ref_chosen_log_probs) -     # log(π(y_win|x)/π_ref(y_win|x))
                (rejected_log_probs - ref_rejected_log_probs)  
            )  
        )  
        
        loss = losses.mean()  
        return loss

    def _get_log_probs(self, logits, labels):  
        """
        计算每个token的对数概率
        
        ##Args:
        logits: x  shape = (batch_size, seq_len, vocab_size)
        labels: y  shape = (batch_size, seq_len)
        
        计算 log(π(y|x))
        
        为了以后计算 reward r(x,y) = β * log(π(y|x)/π_ref(y|x)) 
        
        
        ##Return
        返回值：(batch_size, seq_len)
            每个位置的值表示对应token的对数概率
                
        """  
        log_probs = F.log_softmax(logits, dim=-1)  
        # 在 log_probs 的最后一个维度（vocab_size维度）上，根据 labels 的索引收集对应的对数概率
        return torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)  




class DPOTrainerWrapper:  
    def __init__(  
        self,  
        output_dir: str = DPO_MODEL_PATH,  
        dataset_name_or_path: str = DPO_DATA_PATH,  
        cached_dataset_name_or_path:str = CACHED_DPO_DATA_PATH,
        model_name: str = MODEL_PATH,  
        is_ds: bool = True,  
        ds_config_path: Optional[str] = None,  
        is_peft: bool = False,  
        peft_config: Optional[LoraConfig] = None,  
        is_quantized: bool = False,  
        bnb_config: Optional[BitsAndBytesConfig] = None,  
        max_seq_length: int = 1024,  
        beta: float = 0.1,
        dpo_training_config = None
    ):  
        self.output_dir = output_dir  
        self.dataset_name_or_path = dataset_name_or_path  
        self.cached_dataset_name_or_path = cached_dataset_name_or_path
        self.beta = beta  
        self.max_seq_length = max_seq_length  
        self.is_quantized = is_quantized
        
        
        self.dpo_training_config = dpo_training_config or DPOTrainingConfig()
        

        # 初始化模型和tokenizer  
        self.model, self.tokenizer = self._init_model_and_tokenizer(  
            model_name, is_quantized, bnb_config  
        )  
        
        self.device = self.model.device
        
        
        self.ref_model = self._get_ref_model()
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备数据集  
        self.dataset, self.eval_dataset = self._load_cached_dataset(self.cached_dataset_name_or_path)
        # 配置训练参数  
        self.training_args = TrainingArguments(  
            label_names= ["chosen_labels", "rejected_labels"], # 防止 peft 报错
            output_dir=output_dir,  
            deepspeed=ds_config_path if is_ds else None,  
            per_device_train_batch_size= self.dpo_training_config.batch_size,  
            gradient_accumulation_steps=2,  
            learning_rate=2e-5,  
            bf16=True,  
            logging_steps=10,  
            save_steps=500,  
            remove_unused_columns=False,  
            optim="adamw_torch",  
            max_grad_norm=0.3,  
            num_train_epochs=3,
            report_to = 'none'
        )  

        # 初始化自定义Trainer  
        self.trainer = DPOTrainer(  
            ref_model = self.ref_model,
            beta = self.beta,
            model=self.model,  
            args=self.training_args,  
            train_dataset=self.dataset,  
            data_collator=self.dpo_collator,  
            compute_metrics=self._compute_metrics,  # 计算模型选择正确偏好的比例
        )  
        
        
        
    def _get_ref_model(self):
        """创建并返回参考模型"""
        ref_model = Qwen2ForCausalLM.from_pretrained(MODEL_PATH)
        ref_model.load_state_dict(self.model.base_model.state_dict())
        ref_model = ref_model.to(self.device)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        return ref_model

    def _init_model_and_tokenizer(self, model_name, is_quantized, bnb_config):
        """Initialize model and tokenizer using common utilities."""
        return init_model_and_tokenizer(
            model_name,
            is_quantized,
            bnb_config
        )

    def _default_lora_config(self):
        """Get default LoRA configuration using common utilities."""
        return create_default_lora_config()  

    def _load_cached_dataset(self, dataset_path=CACHED_DPO_DATA_PATH):
        if not os.path.exists(dataset_path):
            os.makedirs(CACHED_DPO_DATA_PATH, exist_ok=True)
        try:
            tokenized_data = load_from_disk(dataset_path)
            print("从缓存加载DPO数据集成功~~~")
            tokenized_data.set_format(type="torch")  # 确保在执行 dpo_collator 之前，数据格式被转换为torch张量
            train_data = tokenized_data['train']
            eval_data = tokenized_data['validation']
            return DPODataset(train_data), DPODataset(eval_data)
        except Exception as e:
            print(f"加载缓存的DPO数据集（tokenized）失败: {e}, 将重新预处理数据")
            ppo_train_data, ppo_eval_data = self._prepare_dataset(self.dataset_name_or_path)
            return ppo_train_data, ppo_eval_data
    
    
    def _prepare_dataset(self, dataset_path):  
        """数据预处理"""  
        dataset = load_dataset(dataset_path)
        train_dataset = load_dataset(dataset_path, split='train').select(range(500))  
        
        if "validation" in dataset:
            eval_dataset = load_dataset(dataset_path, split='validation').select(range(500)) 
        else:
            eval_dataset = load_dataset(dataset_path, split='train').select(range(500, 1000))  
        
        train_dataset = train_dataset.filter(self._data_filter)  
        eval_dataset = eval_dataset.filter(self._data_filter)  
        
        
        train_data = train_dataset.map(  
            self._tokenize_function,  
            batched=True,  
            num_proc=1,  
            remove_columns=train_dataset.column_names  
        )  
        
        val_data = eval_dataset.map(
            self._tokenize_function,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names
        )
        
        tokenized_data = DatasetDict({
            'train': train_data,
            'validation': val_data
        })
        
        
        # 保存预处理好的数据集到本地
        if not os.path.exists(CACHED_DPO_DATA_PATH):
            os.makedirs(CACHED_DPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_DPO_DATA_PATH)
        
        
        train_data.set_format(type="torch")
        val_data.set_format(type="torch")
        
        
        
        return DPODataset(train_data), DPODataset(val_data)
        

    def _data_filter(self, sample):  
        """数据过滤逻辑"""  
        return all([sample["prompt"], sample["chosen"], sample["rejected"]]) and \
               len(sample["prompt"]) <= 512 and \
               len(sample["chosen"]) <= 1024 and \
               len(sample["rejected"]) <= 1024  

    def _tokenize_function(self, samples):  
        """DPO专用tokenize处理"""  
        batch = {"input_ids": [], "attention_mask": [],   
                "chosen_labels": [], "rejected_labels": []}  
        
        for prompt, chosen, rejected in zip(samples["prompt"], samples["chosen"], samples["rejected"]):  
            # 生成prompt模板  
            full_prompt = f"Instruction: {prompt}\nResponse: "  
            
            # Tokenize chosen响应  
            chosen_tokens = self.tokenizer(  
                full_prompt + chosen,  
                max_length=self.max_seq_length,  
                padding="max_length",  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            # Tokenize rejected响应  
            rejected_tokens = self.tokenizer(  
                full_prompt + rejected,  
                max_length=self.max_seq_length,  
                padding="max_length",  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            batch["input_ids"].append(chosen_tokens["input_ids"][0])  
            batch["attention_mask"].append(chosen_tokens["attention_mask"][0])  
            batch["chosen_labels"].append(chosen_tokens["input_ids"][0])  
            batch["rejected_labels"].append(rejected_tokens["input_ids"][0])  
            
        return batch  

    def dpo_collator(self, features):  
        """自定义数据整理函数"""  
        batch = {  
            "input_ids": torch.stack([f["input_ids"] for f in features]),  
            "attention_mask": torch.stack([f["attention_mask"] for f in features]),  
            "chosen_labels": torch.stack([f["chosen_labels"] for f in features]),  
            "rejected_labels": torch.stack([f["rejected_labels"] for f in features])  
        }  
        return batch  

    def _compute_metrics(self, eval_pred):  
        """自定义评估指标
        
        eval_pred.predictions  == (logits_chosen, logits_rejected)
        
        eval_pred.predictions包含两个部分：
            模型对优选回答的预测logits(logits_chosen)和对拒绝回答的预测logits(logits_rejected)
         
        ### Return
        accuracy = (logits_chosen > logits_rejected).mean()
            比较这两个logits的大小关系，计算模型正确偏好(即优选回答得分高于拒绝回答)的比例
            
        ### 注意：  
            1. 这里的logits实际上是模型对完整序列(提示+回答)的预测值
            2. 比较的是整个序列的综合评分，而不仅仅是单个token
            3. 指标值在0-1之间，1表示完美偏好学习
        
        """  
        logits_chosen, logits_rejected = eval_pred.predictions  
        accuracy = (logits_chosen > logits_rejected).mean()  
        return {"dpo_accuracy": accuracy}  

    def train(self):  
        """启动训练流程"""  
        self.trainer.train()  
        
        
        
        
    

    def save_model(self):  
        """模型保存逻辑"""  
        if not os.path.exists(DPO_MODEL_PATH):
            os.makedirs(DPO_MODEL_PATH, exist_ok=True)
        self.model.save_pretrained(DPO_MODEL_PATH)  












class DPOCallback(TrainerCallback):  
    def on_train_begin(self, args, state, control, **kwargs):  
        """训练开始前的初始化"""  
        self.model = kwargs.pop("model")  
        if isinstance(self.model, DeepSpeedEngine):  
            self.model = self.model.module  
            
        if isinstance(self.model, PeftModel) or isinstance(self.model, PeftModelForCausalLM):
            self.model = self.model.base_model

    def on_step_begin(self, args, state, control, **kwargs):  
        # """梯度累积期间冻结参考模型"""  
        # if state.global_step == 0:  
        #     # 初始化参考模型参数  
        #     self.ref_model = self._clone_model(self.model)  
        #     # self.ref_model.requires_grad_(False)
        #     self._frozen_model(self.ref_model)
        
        """不再需要在这里初始化ref_model"""
        pass
            
            
    def _frozen_model(self, model:nn.Module):
        """冻结模型参数"""
        if hasattr(model, 'named_parameters'):
            for param in model.parameters():
                param.requires_grad = False
        else:
            raise ValueError("传入的对象不是有效的PyTorch模型")
        

    def _clone_model(self, model):  
        """
        创建参考模型的深拷贝
        
        
        type(model)：获取模型的类类型（如 Qwen2ForCausalLM）
        **model.config.to_dict()：将模型的配置转换为字典并解包为关键字参数
        type(model)(**model.config.to_dict())：使用原始模型的配置创建一个新的模型实例
        """  
        cloned_model = Qwen2ForCausalLM.from_pretrained(MODEL_PATH)
        cloned_model.load_state_dict(model.state_dict())
        return cloned_model

    
    
    
    
        '''
        log_probs = torch.tensor([
            [[-0.5, -1.0, -2.0],  # 第一个样本，第一个token的对数概率
            [-0.8, -1.2, -1.5]], # 第一个样本，第二个token的对数概率
            [[-0.6, -1.1, -2.1],  # 第二个样本，第一个token的对数概率
            [-0.9, -1.3, -1.6]]  # 第二个样本，第二个token的对数概率
        ])  # shape: (2, 2, 3)  (batch_size=2, seq_len=2, vocab_size=3)

        labels = torch.tensor([
            [0, 2],  # 第一个样本的标签
            [1, 0]   # 第二个样本的标签
        ])  # shape: (2, 2)  (batch_size=2, seq_len=2)
        
        
        执行过程：
        labels.unsqueeze(-1)：

        在 labels 的最后一个维度上增加一个维度
        结果：

        python
        Apply
        tensor([
            [[0], [2]],  # 第一个样本
            [[1], [0]]   # 第二个样本
        ])  # shape: (2, 2, 1)
        torch.gather(log_probs, -1, labels.unsqueeze(-1))：

        在 log_probs 的最后一个维度（vocab_size维度）上，根据 labels 的索引收集对应的对数概率
        结果：

        python
        Apply
        tensor([
            [[-0.5], [-1.5]],  # 第一个样本
            [[-1.1], [-0.9]]   # 第二个样本
        ])  # shape: (2, 2, 1)
        .squeeze(-1)：

        移除最后一个维度
        最终结果：

        python
        Apply
        tensor([
            [-0.5, -1.5],  # 第一个样本
            [-1.1, -0.9]   # 第二个样本
        ])  # shape: (2, 2)
        解释：
        对于第一个样本的第一个token，labels[0,0]=0，所以收集 log_probs[0,0,0]=-0.5
        对于第一个样本的第二个token，labels[0,1]=2，所以收集 log_probs[0,1,2]=-1.5
        对于第二个样本的第一个token，labels[1,0]=1，所以收集 log_probs[1,0,1]=-1.1
        对于第二个样本的第二个token，labels[1,1]=0，所以收集 log_probs[1,1,0]=-0.9
        
        最终得到的矩阵表示每个样本中每个token对应的对数概率值。
        '''
    
    
    
    



if __name__ == "__main__":
    trainer = DPOTrainerWrapper()
    
    trainer.train()
    