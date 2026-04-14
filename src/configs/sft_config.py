from pathlib import Path  
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from pydantic import BaseModel
import os, sys
from pathlib import Path


@dataclass
class TrainingConfig:
    """训练配置"""
    # 输出目录
    output_dir: str = "./outputs"
    # 是否覆盖输出目录
    overwrite_output_dir: bool = True
    # 训练轮数
    num_train_epochs: int = 3
    # 每个设备的批次大小
    per_device_train_batch_size: int = 4
    # 每个设备的评估批次大小
    per_device_eval_batch_size: int = 4
    # 梯度累积步数
    gradient_accumulation_steps: int = 4
    # 学习率
    learning_rate: float = 2e-5
    # 权重衰减
    weight_decay: float = 0.01
    # 学习率调度器
    lr_scheduler_type: str = "cosine"
    # 预热步数比例
    warmup_ratio: float = 0.1
    # 日志记录步数
    logging_steps: int = 10
    # 评估步数
    eval_steps: int = 100
    # 保存步数
    save_steps: int = 100
    # 最大保存检查点数量
    save_total_limit: int = 3


    """优化器配置"""
    # 优化器类型
    optimizer_type: str = "adamw"
    # 是否使用8位优化器
    use_8bit_optimizer: bool = False
    # 梯度裁剪
    max_grad_norm: float = 1.0


    mixed_precision = True

    mixed_precision_dtype: str = "float16"



@dataclass
class DataConfig:
    """数据配置"""
    # 训练数据路径
    train_file: str = "data/processed/train.json"
    # 验证数据路径
    validation_file: str = "data/processed/validation.json"
    # 最大训练样本数
    max_train_samples: int = None
    # 最大验证样本数
    max_eval_samples: int = None
    """数据预处理配置"""
    # 最大输入序列长度
    max_input_length: int = 512
    # 最大输出序列长度
    max_output_length: int = 512
    # 填充到最大长度
    pad_to_max_length: bool = True
    # 数据加载时的并行进程数
    num_workers: int = 4






@dataclass
class SFTConfig:
    """SFT训练总配置"""
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)




if __name__ == "__main__":
    config = SFTConfig()
    import json
    print(json.dumps(asdict(config), indent=4))   