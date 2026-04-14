from typing import Dict, Optional
import os
import torch
import swanlab
from swanlab.integration.huggingface import SwanLabCallback
import numpy as np
import evaluate
import deepspeed
import transformers
from transformers import (
    Trainer, 
    TrainingArguments,
    default_data_collator,
    DataCollatorForLanguageModeling,  
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
    AutoTokenizer
)

from transformers import TrainerCallback  
from peft import PeftModel


import sys
sys.path.append("../../")  # 添加上级目录的上级目录到sys.path
sys.path.append("../")
from src.configs.config import (
    MODEL_CONFIG, 
    BATCH_SIZE, 
    DEEPSPEED_CONFIG_PATH,
    SFT_MODEL_PATH,
    SFT_DPO_MODEL_PATH
)
from src.models.model import TravelAgent
from src.utils.utils import (
    parse_args,
    get_max_length_from_model,
    check_deepspeed_env,
    check_deepspeed_config,
    load_qwen_in_4bit,
    SFTArguments,
    monitor_memory
)

from src.data.data_processor import DataProcessor, CrossWOZProcessor
from contextlib import contextmanager


# MODEL_PATH = "/root/autodl-tmp/models/Qwen2.5-1.5B"


'''
python sft_trainer.py \
--model_name "/root/autodl-tmp/models/Qwen2.5-1.5B" \
--output_dir "output" \
--device "cuda" \
--device_map "auto"


deepspeed --num_gpus=2 sft_trainer.py \
--deepspeed ds_config.json \
--model_name "/root/autodl-tmp/models/Qwen2.5-1.5B" \
--output_dir "output" \
--device "cuda" \
--device_map "auto"

deepspeed --num_gpus=2 sft_trainer.py \
--deepspeed ds_config.json


deepspeed --num_gpus 2 sft_trainer.py \
    --deepspeed ds_config.json

'''



class MemoryCallback(TrainerCallback):  
    def on_step_end(self, args, state, control, **kwargs):  
        monitor_memory() 

class CustomTrainer(Trainer):  
    @contextmanager  
    def compute_loss_context_manager(self):  
        """  
        重写这个方法以禁用 no_sync 上下文管理器  
        """  
        if self.args.gradient_accumulation_steps > 1:  
            if self.deepspeed:  
                # 对于 deepspeed，我们直接返回一个空的上下文管理器  
                yield  
            else:  
                # 对于非 deepspeed，保持原有行为  
                if self.model.is_gradient_checkpointing:  
                    # 如果使用了梯度检查点，不要使用 no_sync  
                    yield  
                else:  
                    with self.model.no_sync():  
                        yield  
        else:  
            yield  

class SFTTrainer:
    """
    监督微调训练器
    """
    def __init__(
        self,
        travel_agent: TravelAgent = None,
        # model_name: str,
        output_dir: str = SFT_MODEL_PATH,
        training_args: Optional[TrainingArguments] = None,
        # device = "auto",
        # device_map = 'auto',
        lora_config: Optional[Dict] = None,
        use_lora = False,
        max_length = 50,
        local_rank = -1,
        args: SFTArguments = None
    ):
        """
        初始化训练器
        
        Args:
            model_name: 基础模型名称
            output_dir: 输出目录
            training_args: 训练参数
        """
        if check_deepspeed_env():
            pass
        else:
            raise ValueError("DeepSpeed is not installed or not configured correctly.")
        
        
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
            self.lora_config  = travel_agent.lora_config
            
            
            
        # 加载模型和分词器 # 添加LoRA
        self.agent=TravelAgent(
            model_name=self.model_name,
            device=self.device,
            device_map=self.device_map,
            lora_config=lora_config,
            use_lora = self.use_lora,
        ) if travel_agent is None else travel_agent
        
        

        self.max_length= max_length
        
        if self.local_rank!=-1:
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = self.agent.model.device
        
        
        self.model = self.agent.model
        # self.max_length = get_max_length_from_model(self.model)
        self.tokenizer = self.agent.tokenizer
        
        '''
        无论选择哪种方案，确保：
            DeepSpeed的train_batch_size等于实际的总batch size
            DeepSpeed的train_micro_batch_size_per_gpu与TrainingArguments的per_device_train_batch_size相等
            所有数值满足：total_batch = micro_batch * num_gpus * grad_accum
        '''

        # 设置默认训练参数
        default_training_args = TrainingArguments(  
            output_dir=self.output_dir,  
            num_train_epochs=5,  
            per_device_train_batch_size=1,  # 每个GPU上的batch size
            per_device_eval_batch_size=1,  
            gradient_accumulation_steps=4,  
            learning_rate=2e-4,  
            weight_decay=0.01,  
            warmup_steps=100,
            warmup_ratio=0.03,  
            lr_scheduler_type="cosine",
            # DeepSpeed使用自己的bf16/fp16配置，不需要在TrainingArguments中设置
            # 这样可以避免冲突，DeepSpeed会在ds_config.json中配置 bf16 enabled: true
            bf16=False,
            fp16=False,  
            logging_dir="./logs",  # 指定日志目录  
            logging_strategy="steps",  
            logging_steps=100,  
            logging_first_step=True,  
            report_to="none",  # 已经在回调函数中配置了 SwanLab
            save_steps=100,  
            eval_steps=100,  
            save_total_limit=3,  
            evaluation_strategy="steps",  
            load_best_model_at_end=True,  
            # report_to="tensorboard",  
            # DeepSpeed配置  
            deepspeed=DEEPSPEED_CONFIG_PATH,  
            # 分布式训练配置  
            local_rank=int(os.getenv("LOCAL_RANK", -1)),  
            ddp_find_unused_parameters=False,  
            # 添加以下参数来启用 8-bit 优化器  
            optim="paged_adamw_8bit",  
        )  
        
        # 更新训练参数
        if training_args:
            self.training_args = training_args
        else:
            self.training_args = default_training_args
        
        check_deepspeed_config(self.training_args)
    
 
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """
        开始训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            resume_from_checkpoint: 恢复训练的检查点路径
        """
        swanlab_callback = SwanLabCallback(
            project="qwen2-sft",
            log_dir = "./swanlab_logs",
            experiment_name="Qwen2-0.5B",
            description="使用通义千问Qwen2-0.5B模型在travel_qa数据集上微调。",
            config={
                "model": "qwen/Qwen2-0.5B",
                "dataset": "travel_qa",
            }
        )
        
        # 数据整理器  
        data_collator = DataCollatorForSeq2Seq(  
            tokenizer=self.tokenizer,  
            model=self.model,
            max_length=self.max_length, # self.max_length 参数指的是 input + output 的总长度。
            padding="max_length",
            return_tensors="pt",
            pad_to_multiple_of=8,  # 提升计算效率  
            # padding="longest",      # 动态填充 
            # mlm=False  
        )  
        
        
        
        sample = next(iter(train_dataset))  
        print("Input[0] :", sample["input_ids"])  # 应该类似 (seq_len,)  
        # print("Label type:", type(sample["labels"]))      # 应该为torch.Tensor  
        # print("Label shape:", sample["labels"].shape)     # 应该与input_ids一致 
        
        # 创建训练器
        trainer = CustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[  
                transformers.EarlyStoppingCallback(  
                    early_stopping_patience=3,  
                    early_stopping_threshold=0.01  
                ),
                MemoryCallback(),
                swanlab_callback
            ]  
        )
        
        monitor_memory()
        torch.cuda.empty_cache()  
        # 开始训练
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # 只在主进程保存模型  
        if self.local_rank in [-1, 0]: 
            # 检查输出目录是否存在  
            os.makedirs(self.output_dir, exist_ok=True)   
            trainer.save_model(self.output_dir)
            # self.tokenizer.save_pretrained(self.output_dir)
            
            # 方案2
            # self.model.save_pretrained(  
            #     self.output_dir,  
            #     safe_serialization=True  # 使用安全序列化  
            # )  

        
        return trainer
    
    @staticmethod
    def load_trained_model(
        base_model_name: str,
        adapter_path: str = None,
        device_map: str = "auto"
    ) -> tuple:
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
            model = PeftModel.from_pretrained(
                model,
                adapter_path,
                device_map=device_map
            )
        
        return model, tokenizer
    
    # def compute_metrics(self, eval_pred):  
    #     # 计算评估指标  
    #     metric = evaluate.load("perplexity")  
        
    #     predictions, labels = eval_pred  
    #     # 去除padding的影响  
    #     mask = labels != -100  
    #     predictions = predictions[mask]  
    #     labels = labels[mask]  
        
    #     return metric.compute(predictions=predictions, references=labels)  
    
    
    def compute_metrics(self, eval_pred):
        # 计算评估指标
        # 确保获取tokenizer实例  
        tokenizer = self.tokenizer
        
        tokenizer.pad_token_id = tokenizer.eos_token_id  
        
        # 分离预测和标签  
        predictions, labels = eval_pred  # labels.shape = (batch_size, max_length)
        
        # 处理预测结果  
        pred_ids = np.argmax(predictions, axis=-1)  
        
        print("pred_ids = ")
        
        unique_tokens, counts = np.unique(pred_ids, return_counts=True)  
        print("预测token分布:", list(zip(unique_tokens[:5], counts[:5])))  # 显示前5个高频token  


        # 检查是否全为pad_token  
        if (pred_ids == tokenizer.pad_token_id).all():  
            print("警告：所有预测都是pad token！")  
            
        decoded_preds = tokenizer.batch_decode(  
            pred_ids,  
            skip_special_tokens=True,  
            clean_up_tokenization_spaces=True  
        )  
        
        print("labels.shape = ", labels.shape) # (50, 512)
        
        label_ids = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)  
        print("标签样例 label_ids:", label_ids)  # 应该显示完整答案  
        
        
        # 处理标签（过滤填充值-100）  
        decoded_labels = []  
        for label_seq in labels:  
            # 将-100替换为pad_token_id  
            valid_label_ids = np.where(label_seq != -100, label_seq, tokenizer.pad_token_id)  
            decoded_label = tokenizer.decode(  
                valid_label_ids,  
                skip_special_tokens=True,  
                clean_up_tokenization_spaces=True  
            )  
            
            decoded_labels.append(decoded_label)  
            
            
        # 打印前3个样本的输入输出  
        print("\n===== 调试样例 =====")  
        for i in range(3):  
            print(f"样本 {i+1}:")  
            print(f"预测: {decoded_preds[i]}")  
            print(f"参考: {decoded_labels[i]}")  
            print("-------------------")  
        
        # 计算ROUGE-L分数（示例）  
        rouge = evaluate.load("rouge")  
        results = rouge.compute(  
            predictions=decoded_preds,  
            references=decoded_labels,  
            use_stemmer=True,  
            use_aggregator=False  # 获取每个样本的分数  
        )  
        
        # 添加BLEU指标  
        bleu = evaluate.load("bleu")  
        bleu_results = bleu.compute(  
            predictions=decoded_preds,  
            references=[[l] for l in decoded_labels] ,
            # max_order=4,  
            # smooth=True  # 启用平滑处理  
        )  
        
        # 返回平均指标  
        return {  
            "rouge1": np.mean(results["rouge1"]),  
            "rouge2": np.mean(results["rouge2"]),  
            "rougeL": np.mean(results["rougeL"]),
            "bleu": bleu_results["bleu"]  
        }  


if __name__ == "__main__":
    args = SFTArguments()  # 使用parse_args获取参数
    trainer = SFTTrainer(args = args)
    
    processor = CrossWOZProcessor(
        tokenizer=trainer.tokenizer,
        max_length = trainer.max_length,
        system_prompt=None  
    )
    
    
    data_path = "/root/autodl-tmp/Travel-Agent-based-on-LLM-and-SFT/data/processed/crosswoz_sft"
    processed_data = processor.process_conversation_data_huggingface(data_path)
    
    
    trainer.train(
        train_dataset=processed_data["train"],
        eval_dataset=processed_data["test"]
    )
    
    
