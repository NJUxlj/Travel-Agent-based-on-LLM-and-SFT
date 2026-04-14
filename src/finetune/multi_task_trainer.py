# 分别在 MCQ, QA, Text-to-Text数据集上进行微调

import sys
import os
import json
import random
import torch
import numpy as np
import swanlab
from typing import List, Any, Dict, Optional, Tuple
from collections import defaultdict

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    default_data_collator,
    TrainerCallback
)

import evaluate
import deepspeed

# 添加项目路径
sys.path.append("../../")

from src.configs.config import (
    MODEL_CONFIG, 
    BATCH_SIZE, 
    DEEPSPEED_CONFIG_PATH,
    SFT_MODEL_PATH
)
from src.models.model import TravelAgent
from src.utils.utils import (
    parse_args,
    get_max_length_from_model,
    check_deepspeed_env,
    check_deepspeed_config,
    SFTArguments,
    monitor_memory
)
from src.data.data_processor import DataProcessor


class MemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        monitor_memory()


class CustomTrainer(Trainer):
    def __init__(self, task_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_weights = task_weights or {}
        self.current_task = None
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # 从inputs中提取任务类型（如果存在）
        task_type = inputs.pop('task_type', 'default')
        self.current_task = task_type
        
        # 计算基本损失
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # 应用任务权重
        if task_type in self.task_weights:
            loss = loss * self.task_weights[task_type]
        
        return (loss, outputs) if return_outputs else loss


class MultiTaskTrainer:
    """
    多任务微调训练器，支持同时在MCQ、QA和Text-to-Text数据集上进行微调
    实现了多种多任务学习策略：
    1. 混合采样 (mixed_sampling): 将不同任务的数据混合后采样
    2. 交替训练 (alternating_training): 按周期交替训练不同任务
    3. 梯度累积 (gradient_accumulation): 累积不同任务的梯度后更新
    """
    
    def __init__(
        self,
        travel_agent: TravelAgent = None,
        output_dir: str = SFT_MODEL_PATH,
        training_args: Optional[TrainingArguments] = None,
        lora_config: Optional[Dict] = None,
        use_lora: bool = False,
        max_length: int = 512,
        local_rank: int = -1,
        args: SFTArguments = None,
        multi_task_strategy: str = "mixed_sampling",  # mixed_sampling, alternating_training, gradient_accumulation
        task_weights: Optional[Dict[str, float]] = None
    ):
        """
        初始化多任务训练器
        
        Args:
            travel_agent: TravelAgent实例，用于获取模型和分词器
            output_dir: 输出目录
            training_args: 训练参数
            lora_config: LoRA配置
            use_lora: 是否使用LoRA
            max_length: 最大序列长度
            local_rank: 本地进程rank
            args: 命令行参数
            multi_task_strategy: 多任务学习策略
            task_weights: 各任务的权重字典，例如 {"mcq": 1.0, "qa": 1.0, "text2text": 1.0}
        """
        # 检查DeepSpeed环境
        if check_deepspeed_env():
            pass
        else:
            raise ValueError("DeepSpeed is not installed or not configured correctly.")
        
        # 初始化配置参数
        if travel_agent is None:
            self.model_name = args.model_name
            self.output_dir = args.output_dir
            self.device = args.device
            self.device_map = args.device_map
            self.local_rank = args.local_rank
            self.use_lora = use_lora
            self.lora_config = lora_config
        else:
            self.model_name = travel_agent.model_name
            self.output_dir = output_dir
            self.device = travel_agent.device
            self.device_map = travel_agent.device_map
            self.local_rank = local_rank
            self.use_lora = travel_agent.use_lora
            self.lora_config = travel_agent.lora_config
        
        # 加载模型和分词器
        self.agent = TravelAgent(
            model_name=self.model_name,
            device=self.device,
            device_map=self.device_map,
            lora_config=lora_config,
            use_lora=self.use_lora,
        ) if travel_agent is None else travel_agent
        
        self.max_length = max_length
        
        # 设置设备
        if self.local_rank != -1:
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = self.agent.model.device
        
        self.model = self.agent.model
        self.tokenizer = self.agent.tokenizer
        
        # 设置多任务策略和权重
        self.multi_task_strategy = multi_task_strategy
        self.task_weights = task_weights or {"mcq": 1.0, "qa": 1.0, "text2text": 1.0}
        
        # 设置默认训练参数
        default_training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=5,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=100,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            # DeepSpeed使用自己的bf16/fp16配置，不需要在TrainingArguments中设置
            bf16=False,
            fp16=False,
            logging_dir="./logs",
            logging_strategy="steps",
            logging_steps=100,
            logging_first_step=True,
            report_to="none",
            save_steps=100,
            eval_steps=100,
            save_total_limit=3,
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            deepspeed=DEEPSPEED_CONFIG_PATH,
            local_rank=int(os.getenv("LOCAL_RANK", -1)),
            ddp_find_unused_parameters=False,
            optim="paged_adamw_8bit",
        )
        
        # 更新训练参数
        if training_args:
            self.training_args = training_args
        else:
            self.training_args = default_training_args
        
        check_deepspeed_config(self.training_args)
    
    def _create_mixed_dataset(self, datasets: Dict[str, List[Dict]]) -> List[Dict]:
        """
        创建混合数据集，将不同任务的数据混合
        
        Args:
            datasets: 任务名称到数据集的映射
            
        Returns:
            混合后的数据集
        """
        mixed_data = []
        
        # 为每个数据添加任务类型标记
        for task_name, data_list in datasets.items():
            for data in data_list:
                data['task_type'] = task_name
                mixed_data.append(data)
        
        # 打乱数据集
        random.shuffle(mixed_data)
        return mixed_data
    
    def _create_task_collator(self, task_type: str) -> DataCollatorForSeq2Seq:
        """
        为不同任务创建数据整理器
        
        Args:
            task_type: 任务类型
            
        Returns:
            数据整理器实例
        """
        # 可以根据不同任务类型返回不同的数据整理器配置
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
    
    def train(
        self,
        train_datasets: Dict[str, List[Dict]],
        eval_datasets: Optional[Dict[str, List[Dict]]] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        开始多任务训练
        
        Args:
            train_datasets: 任务名称到训练数据集的映射
            eval_datasets: 任务名称到评估数据集的映射
            resume_from_checkpoint: 恢复训练的检查点路径
            
        Returns:
            训练器实例
        """
        # 创建SwanLab回调
        swanlab_callback = swanlab.integration.huggingface.SwanLabCallback(
            project="qwen2-multitask",
            log_dir="./swanlab_logs",
            experiment_name="Qwen2-Multitask",
            description="使用通义千问Qwen2模型在多任务上微调。",
            config={
                "model": self.model_name,
                "tasks": list(train_datasets.keys()),
                "strategy": self.multi_task_strategy
            }
        )
        
        # 根据策略处理数据集
        if self.multi_task_strategy == "mixed_sampling":
            # 混合采样策略
            train_dataset = self._create_mixed_dataset(train_datasets)
            eval_dataset = self._create_mixed_dataset(eval_datasets) if eval_datasets else None
            
            # 创建数据整理器
            data_collator = self._create_task_collator("default")
            
            # 创建训练器
            trainer = CustomTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                task_weights=self.task_weights,
                callbacks=[
                    TrainerCallback,
                    MemoryCallback(),
                    swanlab_callback
                ]
            )
            
            # 开始训练
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
        elif self.multi_task_strategy == "alternating_training":
            # 交替训练策略
            for epoch in range(self.training_args.num_train_epochs):
                for task_name, task_dataset in train_datasets.items():
                    print(f"Epoch {epoch+1}, Training on task: {task_name}")
                    
                    # 为当前任务创建数据集
                    task_data = [{'task_type': task_name, **data} for data in task_dataset]
                    
                    # 获取对应的评估数据集
                    task_eval_data = None
                    if eval_datasets and task_name in eval_datasets:
                        task_eval_data = [{'task_type': task_name, **data} for data in eval_datasets[task_name]]
                    
                    # 创建数据整理器
                    data_collator = self._create_task_collator(task_name)
                    
                    # 创建训练器
                    task_trainer = CustomTrainer(
                        model=self.model,
                        args=self.training_args,
                        train_dataset=task_data,
                        eval_dataset=task_eval_data,
                        tokenizer=self.tokenizer,
                        data_collator=data_collator,
                        compute_metrics=self.compute_metrics,
                        task_weights=self.task_weights,
                        callbacks=[
                            MemoryCallback(),
                            swanlab_callback
                        ]
                    )
                    
                    # 训练一个epoch
                    task_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
                    
                    # 更新checkpoint路径，确保后续任务从当前状态继续
                    if hasattr(task_trainer, 'state') and task_trainer.state.best_model_checkpoint:
                        resume_from_checkpoint = task_trainer.state.best_model_checkpoint
            
            # 最后使用主训练器
            trainer = CustomTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self._create_mixed_dataset(train_datasets),
                eval_dataset=self._create_mixed_dataset(eval_datasets) if eval_datasets else None,
                tokenizer=self.tokenizer,
                data_collator=self._create_task_collator("default"),
                compute_metrics=self.compute_metrics,
                task_weights=self.task_weights,
                callbacks=[MemoryCallback(), swanlab_callback]
            )
            
        elif self.multi_task_strategy == "gradient_accumulation":
            # 梯度累积策略（简化版，实际实现需要更复杂的梯度处理）
            train_dataset = self._create_mixed_dataset(train_datasets)
            eval_dataset = self._create_mixed_dataset(eval_datasets) if eval_datasets else None
            
            # 创建数据整理器
            data_collator = self._create_task_collator("default")
            
            # 创建训练器
            trainer = CustomTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                task_weights=self.task_weights,
                callbacks=[
                    TrainerCallback,
                    MemoryCallback(),
                    swanlab_callback
                ]
            )
            
            # 开始训练
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        else:
            raise ValueError(f"Unsupported multi-task strategy: {self.multi_task_strategy}")
        
        # 只在主进程保存模型
        if self.local_rank in [-1, 0]:
            # 检查输出目录是否存在
            os.makedirs(self.output_dir, exist_ok=True)
            trainer.save_model(self.output_dir)
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """
        计算评估指标
        
        Args:
            eval_pred: 评估预测结果
            
        Returns:
            指标字典
        """
        # 确保获取tokenizer实例
        tokenizer = self.tokenizer
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 分离预测和标签
        predictions, labels = eval_pred  # predictions.shape =  
        
        # 处理预测结果
        pred_ids = np.argmax(predictions, axis=-1)
        
        # 解码预测和标签
        decoded_preds = tokenizer.batch_decode(
            pred_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 处理标签（过滤填充值-100）
        decoded_labels = []
        for label_seq in labels:
            valid_label_ids = np.where(label_seq != -100, label_seq, tokenizer.pad_token_id)
            decoded_label = tokenizer.decode(
                valid_label_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded_labels.append(decoded_label)
        
        # 计算ROUGE指标
        rouge = evaluate.load("rouge")
        results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
            use_aggregator=False
        )
        
        # 添加BLEU指标
        bleu = evaluate.load("bleu")
        bleu_results = bleu.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels]
        )
        
        # 返回平均指标
        return {
            "rouge1": np.mean(results["rouge1"]),
            "rouge2": np.mean(results["rouge2"]),
            "rougeL": np.mean(results["rougeL"]),
            "bleu": bleu_results["bleu"]
        }
    
    @staticmethod
    def load_trained_model(
        base_model_name: str,
        adapter_path: str = None,
        device_map: str = "auto"
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        加载训练好的模型
        
        Args:
            base_model_name: 基础模型名称
            adapter_path: LoRA权重路径
            device_map: 设备映射策略
            
        Returns:
            tuple: (model, tokenizer)
        """
        # 加载基础模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device_map
        )
        
        if adapter_path is not None:
            # 加载LoRA权重
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                device_map=device_map
            )
        
        return model, tokenizer


def main():
    """
    主函数，用于示例如何使用多任务训练器
    """
    # 解析命令行参数
    args = SFTArguments()
    
    # 初始化多任务训练器
    multi_task_trainer = MultiTaskTrainer(
        args=args,
        multi_task_strategy="mixed_sampling",  # 可以选择: mixed_sampling, alternating_training, gradient_accumulation
        task_weights={"mcq": 1.0, "qa": 1.0, "text2text": 1.0},  # 可以调整任务权重
        max_length=512
    )
    
    # 这里需要加载不同任务的数据集
    # 注意：以下是示例代码，实际使用时需要根据具体数据格式实现
    train_datasets = {
        "mcq": [],  # 多选题数据集
        "qa": [],  # 问答数据集
        "text2text": []  # 文本到文本数据集
    }
    
    eval_datasets = {
        "mcq": [],
        "qa": [],
        "text2text": []
    }
    
    # 开始训练
    multi_task_trainer.train(
        train_datasets=train_datasets,
        eval_datasets=eval_datasets
    )


if __name__ == "__main__":
    main()