import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    Qwen2ForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict
from typing import Optional, Dict, List, Union, Tuple
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np
import copy
from torch.nn.utils.rnn import pad_sequence

from src.configs.config import (
    MODEL_PATH,
    DATA_PATH,
    DPO_DATA_PATH,
    CACHED_DPO_DATA_PATH,
    GRPO_MODEL_PATH,
    CACHED_GRPO_DATA_PATH,
)


from src.evaluation.qa_evaluate import QAEvaluator
from src.finetune.base_trainer import (
    init_model_and_tokenizer,
    create_default_lora_config,
)


'''

GRPO算法的核心优势在于：

    无需critic模型，通过组采样方式估计advantage
    在每个prompt上生成多个输出，使用组内响应的相对得分计算advantage
    使用KL惩罚项保持与参考模型的接近度
    
    
GRPO核心思想：
    与PPO不同，GRPO不使用critic网络来估计值函数。
    相反，它使用同一prompt生成的多个响应（一个组）来计算相对优势。
    组内的每个响应与该组的平均得分进行比较，得到相对优势。

优势归一化：
    优势通过组内的均值和标准差进行归一化（可选），这有助于稳定训练，防止高方差。
    公式：(r_i - mean(r)) / std(r)

KL散度惩罚：
    为防止策略偏离太远，添加了KL散度惩罚项，这与PPO中的信任区域约束相似。

数据处理：
    代码改进了数据处理方式，将prompt和response分开，这有助于更精确地计算响应生成的概率。

'''


class QADataset(Dataset):  
    """  
    用于问答评估的数据集类  
    """  
    def __init__(self, dataset):  
        self.dataset = dataset  
        self.input_ids = dataset["input_ids"]  
        self.attention_mask = dataset["attention_mask"]  
        self.labels = dataset["labels"]  
    
    def __len__(self):  
        return len(self.input_ids)  
    
    def __getitem__(self, idx):  
        item = {  
            "input_ids": self.input_ids[idx],  
            "attention_mask": self.attention_mask[idx],  
            "labels": self.labels[idx]  
        }  
        return item  




class CustomGRPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        self.input_ids = tokenized_data["input_ids"]  
        self.attention_mask = tokenized_data["attention_mask"]  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx]  
        }  


@dataclass  
class GRPOConfig:  
    """Configuration for GRPO training."""  
    beta: float = 0.1  # KL divergence weight  
    group_size: int = 4  # Number of responses per prompt  
    mu: float = 1.0  # Clipping coefficient (1.0 means no clipping)  
    kl_coef: float = 0.1  # KL coefficient  
    scale_rewards: bool = True  # Whether to normalize advantages by std  


class GRPOTrainer(Trainer):  
    def __init__(  
        self,  
        ref_model,
        tokenizer,
        max_seq_length,
        grpo_config: Optional[GRPOConfig] = None,
        **kwargs  
    ):  
        super().__init__(**kwargs)
        self.ref_model = ref_model
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.device = self.ref_model.device
        
        # GRPO配置参数
        self.grpo_config = grpo_config or GRPOConfig()
        self.beta = self.grpo_config.beta
        self.group_size = self.grpo_config.group_size
        self.mu = self.grpo_config.mu
        self.kl_coef = self.grpo_config.kl_coef
        self.scale_rewards = self.grpo_config.scale_rewards
        
    def compute_metrics(self, eval_preds):  
        """计算评估指标  
        
        Args:  
            eval_preds: 评估预测结果，包含predictions和label_ids  
            
        Returns:  
            包含多个评估指标的字典  
        """  
        
        evaluator = QAEvaluator(self.model, self.tokenizer, self.max_seq_length)
        
        evaluator.compute_metrics(eval_preds)


    def compute_loss(self, model, inputs, num_items_in_batch=None):
        """计算GRPO损失函数  
        
        GRPO的核心是计算组内样本的相对优势，然后使用这些优势计算损失  
        """  
        input_ids = inputs["input_ids"]  
        attention_mask = inputs["attention_mask"]  
        batch_size = input_ids.shape[0]  
        
        # 获取group_ids（如果存在）
        group_ids = inputs.get("group_ids", None)
        if group_ids is None:
            # 如果没有group_ids，假设每2个样本组成一个组
            group_ids = torch.arange(0, batch_size // 2, dtype=torch.long).repeat_interleave(2)
            if len(group_ids) < batch_size:
                group_ids = torch.cat([group_ids, group_ids[-1:] * (batch_size - len(group_ids))])
        
        # 前向传播获取当前策略的logits  
        outputs = model(  
            input_ids=input_ids,  
            attention_mask=attention_mask  
        )  
        logits = outputs.logits  
        
        # 获取labels（输入的下一个token）  
        labels = input_ids.clone()  
        
        # 创建因果mask以确保只考虑自回归条件概率  
        # 找出所有非pad的位置  
        non_padding_mask = labels != self.tokenizer.pad_token_id  
        
        # 计算当前策略的对数概率  
        log_probs = self._get_batch_logprobs(logits, labels, non_padding_mask)  
        
        # 计算参考策略的对数概率  
        with torch.no_grad():  
            ref_outputs = self.ref_model(  
                input_ids=input_ids,  
                attention_mask=attention_mask  
            )  
            ref_logits = ref_outputs.logits  
            ref_log_probs = self._get_batch_logprobs(ref_logits, labels, non_padding_mask)  
        
        # 计算KL散度（log_p - log_ref_p）  
        kl_divs = log_probs - ref_log_probs  
        
        # 计算组内优势
        advantages = []
        unique_groups = torch.unique(group_ids)
        
        for group_id in unique_groups:
            group_mask = group_ids == group_id
            group_kl_divs = kl_divs[group_mask]
            
            if len(group_kl_divs) > 1:
                # 计算组内相对优势
                group_mean = group_kl_divs.mean()
                if self.scale_rewards:
                    group_std = group_kl_divs.std() + 1e-8
                    group_advantages = (group_kl_divs - group_mean) / group_std
                else:
                    group_advantages = group_kl_divs - group_mean
            else:
                # 如果组内只有一个样本，优势为0
                group_advantages = torch.zeros_like(group_kl_divs)
            
            advantages.append(group_advantages)
        
        # 将所有优势合并回一个张量  
        all_advantages = torch.cat(advantages)  
        
        # 计算GRPO的surrogate loss（最大化期望奖励）  
        policy_ratio = torch.exp(log_probs - ref_log_probs.detach())  
        
        # 应用比率裁剪（如果启用）  
        if self.mu < 1.0:  
            clipped_ratio = torch.clamp(policy_ratio, 1 - self.mu, 1 + self.mu)  
            surrogate1 = policy_ratio * all_advantages  
            surrogate2 = clipped_ratio * all_advantages  
            surrogate_loss = -torch.min(surrogate1, surrogate2).mean()  
        else:  
            surrogate_loss = -(policy_ratio * all_advantages).mean()  
        
        # 计算KL散度惩罚项  
        kl_loss = (log_probs - ref_log_probs.detach()).mean()  
        
        # 总损失  
        total_loss = surrogate_loss + self.kl_coef * kl_loss  
        
        return total_loss  

    def _get_batch_logprobs(self, logits, labels, non_padding_mask):  
        """计算批次的对数概率  
        
        对于每个序列，计算所有(非pad)token的条件对数概率之和  
        """  
        # 对logits应用log_softmax  
        log_probs = F.log_softmax(logits, dim=-1)  
        
        # 收集label对应的对数概率  
        # 将labels移动到与log_probs相同的设备  
        labels = labels.to(log_probs.device)  
        
        # 获取每个位置上标签对应的对数概率  
        token_log_probs = torch.gather(log_probs[:, :-1], -1, labels[:, 1:].unsqueeze(-1)).squeeze(-1)  
        
        # 应用非填充掩码，计算每个序列的对数概率之和  
        # 确保掩码尺寸正确  
        seq_log_probs = (token_log_probs * non_padding_mask[:, 1:]).sum(dim=1)  
        
        return seq_log_probs  



        

class GRPOTrainerWrapper:  
    def __init__(  
        self,  
        output_dir: str = GRPO_MODEL_PATH,  
        dataset_name_or_path: str = DPO_DATA_PATH,  
        sft_dataset_name_or_path: str = DATA_PATH,
        cached_dataset_name_or_path:str = CACHED_GRPO_DATA_PATH,
        model_name: str = MODEL_PATH,  
        is_ds: bool = True,  
        ds_config_path: Optional[str] = None,  
        is_peft: bool = False,  
        peft_config: Optional[LoraConfig] = None,  
        is_quantized: bool = False,  
        bnb_config: Optional[BitsAndBytesConfig] = None,  
        max_seq_length: int = 1024,  
        grpo_config: Optional[GRPOConfig] = None  
    ):  
        self.output_dir = output_dir
        self.dataset_name_or_path = dataset_name_or_path  
        self.sft_dataset_name_or_path = sft_dataset_name_or_path
        self.cached_dataset_name_or_path = cached_dataset_name_or_path
        self.max_seq_length = max_seq_length  
        self.is_quantized = is_quantized
        
        # GRPO配置  
        self.grpo_config = grpo_config or GRPOConfig()  
        self.beta = self.grpo_config.beta  
        self.group_size = self.grpo_config.group_size  
        self.mu = self.grpo_config.mu  
        self.kl_coef = self.grpo_config.kl_coef  
        self.scale_rewards = self.grpo_config.scale_rewards  

        # 初始化模型和tokenizer  
        self.model, self.tokenizer = self._init_model_and_tokenizer(  
            model_name, is_quantized, bnb_config  
        )  
        
        # 创建参考模型（使用深拷贝确保参数独立）  
        self.ref_model = self._clone_model(self.model)  
        self.ref_model.eval()  # 设置为评估模式  
        self.ref_model.requires_grad_(False)  # 冻结参数  
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备数据集  
        self.dataset = self._prepare_dataset()  

        # 配置训练参数  
        self.training_args = TrainingArguments(  
            output_dir=output_dir,  
            deepspeed=ds_config_path if is_ds else None,  
            per_device_train_batch_size=4,  
            gradient_accumulation_steps=2,  
            learning_rate=2e-5,  
            bf16=True,  
            logging_steps=10,  
            save_steps=500,  
            remove_unused_columns=False,  
            optim="adamw_torch",  
            max_grad_norm=0.3,  
            num_train_epochs=3  
        )  

        self.trainer = GRPOTrainer(  
            ref_model = self.ref_model,
            tokenizer = self.tokenizer,
            max_seq_length= self.max_seq_length,
            grpo_config=self.grpo_config,
            model=self.model,  
            args=self.training_args,  
            train_dataset=self.dataset,  
            eval_dataset=self.eval_dataset,
            data_collator=self.grpo_collator,  
            # compute_metrics=self._compute_metrics,
        )  

    def _init_model_and_tokenizer(self, model_name, is_quantized, bnb_config):
        """Initialize model and tokenizer using common utilities."""
        return init_model_and_tokenizer(
            model_name,
            is_quantized,
            bnb_config
        )

    def _clone_model(self, model):
        """Create a deep copy of the model."""
        return copy.deepcopy(model)

    def _default_lora_config(self):
        """Get default LoRA configuration using common utilities."""
        return create_default_lora_config()

    def _prepare_dataset(self, train_size = 1000, eval_size = 500):  
        '''
        从头开始预处理数据
        
        train数据使用的是DPO数据集, 每个样本包含 (prompt, chosen, rejected) 3个字段
        eval数据使用的是 SFT 数据集 （travel-qa），每个样本包含 (Question, Answer) 两个字段
        '''
        train_dataset = load_dataset(self.dataset_name_or_path, split='train').select(range(train_size))  
        eval_dataset = load_dataset(self.sft_dataset_name_or_path)
        
        if "validation" in eval_dataset:
            eval_dataset = load_dataset(self.sft_dataset_name_or_path, split='validation').select(range(eval_size)) 
        elif "test" in eval_dataset:
            eval_dataset = load_dataset(self.sft_dataset_name_or_path, split='test').select(range(eval_size))
        else:
            eval_dataset = load_dataset(self.sft_dataset_name_or_path, split='train').select(range(eval_size))
            
        train_dataset = train_dataset.filter(self._data_filter)  
        eval_dataset = eval_dataset.filter(self._data_filter)  

        
        train_data = train_dataset.map(  
            self._tokenize_train_function,  
            batched=True,  
            num_proc=1,  
            remove_columns=train_dataset.column_names  
        )  
        
        val_data = eval_dataset.map(
            self._tokenize_eval_function,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names
        )
        
        tokenized_data = DatasetDict({
            'train': train_data,
            'validation': val_data
        })
        
        # 保存预处理好的数据集到本地
        if not os.path.exists(CACHED_GRPO_DATA_PATH):
            os.makedirs(CACHED_GRPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_GRPO_DATA_PATH)
        
        
        train_data.set_format(type="torch")
        val_data.set_format(type="torch")
        
        self.eval_dataset = QADataset(val_data)
        return CustomGRPODataset(train_data)

    def _data_filter(self, sample):  
        # 检查训练数据（DPO格式）
        if "chosen" in sample and "rejected" in sample:
            return all([sample["prompt"], sample["chosen"], sample["rejected"]]) and \
                   len(sample["prompt"]) <= 512 and \
                   len(sample["chosen"]) <= 1024 and \
                   len(sample["rejected"]) <= 1024
        # 检查评估数据（SFT格式）
        elif "Question" in sample and ("Answer" in sample or "Response" in sample):
            answer_field = "Answer" if "Answer" in sample else "Response"
            return all([sample["Question"], sample[answer_field]]) and \
                   len(sample["Question"]) <= 512 and \
                   len(sample[answer_field]) <= 1024
        else:
            return False  
               
               

    def _tokenize_train_function(self, samples):  
        """  
        处理训练数据集（DPO格式，包含prompt, chosen, rejected）  
        返回适用于GRPO训练的格式  
        
        GRPO需要组内多个响应来计算相对优势，这里我们为每个prompt生成多个响应
        """  
        
        batch = {  
            "input_ids": [],            # 所有响应的完整输入  
            "attention_mask": [],       # 所有响应的注意力掩码  
        }  
        
        # 为每个prompt生成多个响应（chosen + rejected + 额外的生成）
        for prompt, chosen, rejected in zip(samples["prompt"], samples["chosen"], samples["rejected"]):  
            # 为GRPO生成组内多个响应
            responses = [chosen, rejected]
            
            # 可以添加更多响应生成逻辑，这里先用chosen和rejected
            # 在实际应用中，可以调用模型生成更多响应
            
            for response in responses:
                # 处理prompt
                prompt_tokens = self.tokenizer(  
                    f"Question: {prompt}\nAnswer:",   
                    max_length=self.max_seq_length // 2,  
                    truncation=True,  
                    return_tensors="pt"  
                )  
                
                # 处理response
                response_tokens = self.tokenizer(  
                    response,  
                    max_length=self.max_seq_length // 2,  
                    truncation=True,  
                    return_tensors="pt"  
                )  
                
                # 合并输入ID和attention mask
                input_ids = torch.cat([  
                    prompt_tokens["input_ids"][0],   
                    response_tokens["input_ids"][0][1:]  # 去掉response的BOS token  
                ])  
                attention_mask = torch.cat([  
                    prompt_tokens["attention_mask"][0],   
                    response_tokens["attention_mask"][0][1:]  
                ])  
                
                # 裁剪到最大长度  
                if len(input_ids) > self.max_seq_length:  
                    input_ids = input_ids[:self.max_seq_length]  
                    attention_mask = attention_mask[:self.max_seq_length]  
                
                # 添加到批次  
                batch["input_ids"].append(input_ids)  
                batch["attention_mask"].append(attention_mask)  
        
        # 使用PyTorch内置的padding  
        from torch.nn.utils.rnn import pad_sequence  
        batch["input_ids"] = pad_sequence(batch["input_ids"], batch_first=True, padding_value=self.tokenizer.pad_token_id)  
        batch["attention_mask"] = pad_sequence(batch["attention_mask"], batch_first=True, padding_value=0)  
        
        return batch  
        
    
    def _tokenize_eval_function(self, samples):  
        """  
        处理评估数据集（travel_qa格式，包含Question和Response/Answer）  
        返回适用于评估的格式  
        """  
        batch = {  
            "input_ids": [],  
            "attention_mask": [],  
            "labels": []  
        }  
        
        # 确定问题和答案字段名  
        question_field = "Question" if "Question" in samples else "question"  
        answer_field = "Response" if "Response" in samples else "Answer" if "Answer" in samples else "answer"  
        
        for question, answer in zip(samples[question_field], samples[answer_field]):  
            # 对问题进行编码  
            question_tokens = self.tokenizer(  
                f"Question: {question}\nAnswer:",  
                max_length=self.max_seq_length // 2,  # 预留一半长度给回答  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            # 对回答进行编码  
            answer_tokens = self.tokenizer(  
                answer,  
                max_length=self.max_seq_length // 2,  
                truncation=True,  
                return_tensors="pt"  
            )  
            
            # 合并输入ID和attention mask  
            input_ids = torch.cat([  
                question_tokens["input_ids"][0],  
                answer_tokens["input_ids"][0][1:]  # 去掉answer的BOS token  
            ])  
            attention_mask = torch.cat([  
                question_tokens["attention_mask"][0],  
                answer_tokens["attention_mask"][0][1:]  
            ])  
            
            # 创建标签 (-100表示不计算损失的token，即提问部分)  
            labels = torch.cat([  
                torch.full_like(question_tokens["input_ids"][0], -100),  
                answer_tokens["input_ids"][0][1:]  # 去掉answer的BOS token  
            ])  
            
            # 裁剪到最大长度  
            if len(input_ids) > self.max_seq_length:  
                input_ids = input_ids[:self.max_seq_length]  
                attention_mask = attention_mask[:self.max_seq_length]  
                labels = labels[:self.max_seq_length]  
            
            batch["input_ids"].append(input_ids)  
            batch["attention_mask"].append(attention_mask)  
            batch["labels"].append(labels)  
        
        # 填充成相同长度的张量  
        batch["input_ids"] = self._pad_sequences(batch["input_ids"], max_seq_length = self.max_seq_length)  
        batch["attention_mask"] = self._pad_sequences(batch["attention_mask"], max_seq_length = self.max_seq_length)  
        batch["labels"] = self._pad_sequences(batch["labels"], padding_value=-100, max_seq_length = self.max_seq_length)  
        
        return batch  
    
    
    def _pad_sequences(self, sequences, padding_value=0, max_seq_length = 1024):  
        """  
        将不同长度的序列填充到相同长度  
        """  
        # max_length = max(len(seq) for seq in sequences)  
        max_length = max_seq_length
        padded_sequences = []  
        
        for seq in sequences:  
            padding_length = max_length - len(seq)  
            padded_seq = torch.cat([  
                seq,   
                torch.full((padding_length,), padding_value, dtype=seq.dtype)  
            ])  
            padded_sequences.append(padded_seq)  
        
        return torch.stack(padded_sequences)  
    
    

    def grpo_collator(self, features):  
        """  
        将多个样本组合成一个批次，用于GRPO训练  
        
        Args:  
            features: 样本列表，每个样本包含input_ids、attention_mask等  
            
        Returns:  
            包含批次数据的字典  
        """  
        # 初始化结果字典  
        batch = {}  
        
        # 检查是否为训练数据（GRPO格式）  
        is_train = "input_ids" in features[0] and "attention_mask" in features[0]  
        
        if is_train:  
            # 处理训练批次（GRPO格式）  
            batch = {  
                "input_ids": torch.stack([f["input_ids"] for f in features]),  
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),  
            }  
            
            # 生成group_ids - 用于GRPO中区分不同问题的回答组
            batch_size = len(features)
            # 验证数据有效性：batch_size必须是偶数（每组包含chosen和rejected）
            if batch_size % 2 != 0:
                raise ValueError(f"Batch size must be even for GRPO training, got {batch_size}. "
                                 f"Each group requires a chosen and rejected sample pair.")
            # 每2个样本组成一个组（chosen和rejected）
            num_groups = batch_size // 2
            group_ids = torch.arange(0, num_groups, dtype=torch.long).repeat_interleave(2)
            batch["group_ids"] = group_ids  
            
            # 确保张量格式正确  
            batch["input_ids"] = batch["input_ids"].to(torch.long)  
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)  
            
        else:  
            # 处理评估批次  
            batch = {  
                "input_ids": torch.stack([f["input_ids"] for f in features]),  
                "attention_mask": torch.stack([f["attention_mask"] for f in features]),  
            }  
            
            # 如果存在labels字段，也添加到批次中  
            if "labels" in features[0]:  
                batch["labels"] = torch.stack([f["labels"] for f in features])  
                batch["labels"] = batch["labels"].to(torch.long)  
            
            # 确保张量格式正确  
            batch["input_ids"] = batch["input_ids"].to(torch.long)  
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)  
        
        return batch  


    def train(self):  
        self.trainer.train()  

    def save_model(self):  
        save_path = os.path.join(self.output_dir, "qwen2_grpo")  
        self.trainer.save_model(save_path)  
        self.tokenizer.save_pretrained(save_path)  