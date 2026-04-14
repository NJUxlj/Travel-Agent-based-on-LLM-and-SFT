import torch  
import torch.nn as nn
from torch.utils.data import Dataset  
import torch.nn.functional as F  
from transformers import (  
    AutoTokenizer,  
    TrainingArguments,  
    Trainer,  
    TrainerCallback,  
    AutoModelForSequenceClassification,
    DebertaV2Model,
    DebertaV2Config
)  

from datasets import DatasetDict
from tqdm import tqdm
from pathlib import Path
import os, sys
sys.path.append(Path(__file__).parent.parent)
from models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

from configs.config import (
    REWARD_MODEL_PATH, 
    MODEL_PATH, 
    SFT_MODEL_PATH, 
    PPO_MODEL_PATH, 
    DPO_DATA_PATH,
    CACHED_DPO_DATA_PATH,
    CACHED_PPO_DATA_PATH,
)


from peft import LoraConfig, get_peft_model  
import os  
import numpy as np  
import copy
from datasets import load_dataset, load_from_disk  

class PPODataset(Dataset):  
    def __init__(self, tokenized_data):  
        self.data = tokenized_data  
        
    def __len__(self):  
        return len(self.data["input_ids"])  
    
    def __getitem__(self, idx):  
        # 返回单个数据样本，不再包含预先计算的advantages和returns  
        # 这些值将在训练过程中动态计算  
        return {  
            "input_ids": self.data["input_ids"][idx],  
            "attention_mask": self.data["attention_mask"][idx],  
            "response_ids": self.data["response_ids"][idx],  
            "old_log_probs": self.data["old_log_probs"][idx],  
        }  
        
        
        
        
class Critic(nn.Module):
    '''
    # 价值（评论家）模型：
    #   - 用于预测每一步（生成token）的动作产生的收益，使用演员模型进行初始化，并外加一个回归头，输出shape为：(batch_size, seq_len， 1)
    #   - 通过策略模型初始化得到
    
    '''
    def __init__(self, base_model):
        super().__init__()
        # 创建base_model的副本而不是直接使用原模型，避免共享参数
        self.base_model = copy.deepcopy(base_model)
        # 不冻结骨干部分，让critic可以正常学习
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)
        
    def forward(
            self, 
            input_ids = None, 
            inputs_embeds = None,
            attention_mask = None,
            past_key_values = None,
            use_cache = False
            ):
        '''
        return values: shape = (batch_size, seq_len)
        '''
        
        assert input_ids is not None or inputs_embeds is not None, f"one of the input_ids and inputs_embeds has to be not None"
        
        
        if input_ids is not None:
            hidden_states = self.base_model.forward(
                input_ids = input_ids,
                attention_mask = attention_mask,
                past_key_values = past_key_values,
                use_cache = use_cache
            ).last_hidden_state  # shape = (batch_size, seq_len, hidden_size)
        else:
            hidden_states = self.base_model.forward(
                inputs_embeds = inputs_embeds,
                attention_mask = attention_mask,
                past_key_values = past_key_values,
                use_cache = use_cache
            ).last_hidden_state
        
        value_model_output = self.value_head(hidden_states)  # shape = (batch_size, seq_len, 1)
        values = value_model_output.squeeze(-1)  # shape = (batch_size, seq_len)
        
        return values
    
        

class PPOTrainer:  
    """  
    PPO训练器类  
    实现PPO算法的核心训练逻辑  
    """ 
    def __init__(  
        self,  
        output_dir: str = PPO_MODEL_PATH,  
        dataset_path: str = DPO_DATA_PATH,  
        cached_data_path:str = CACHED_PPO_DATA_PATH,
        model_name: str = MODEL_PATH,  # 基础模型名称
        reward_model_name:str = REWARD_MODEL_PATH,
        reference_model_name: str = SFT_MODEL_PATH,  # 参考模型应该使用SFT模型
        clip_epsilon: float = 0.2,     # PPO裁剪参数，限制策略更新幅度
        gamma: float = 0.99,           # 折扣因子，控制未来奖励的重要性  
        gae_lambda: float = 0.95,      # GAE λ参数，平衡偏差和方差 
        ppo_epochs: int = 3,            # 每批数据的PPO更新次数 
        lr_actor: float = 5e-5,         # Actor学习率
        lr_critic: float = 5e-5,        # Critic学习率
        max_seq_length: int = 1024,  
        is_peft: bool = True,  
        peft_config: LoraConfig = None,
        kl_coef: float = 0.1,          # KL散度系数
        vf_coef: float = 0.5,          # 价值损失系数
        entropy_coef: float = 0.01     # 熵奖励系数
    ):  
        # 导入copy模块
        import copy
        
        self.output_dir = output_dir  
        
        self.dataset_path = dataset_path
        self.cached_data_path = cached_data_path
        self.max_seq_length = max_seq_length
        
        self.clip_epsilon = clip_epsilon  
        self.gamma = gamma  
        self.gae_lambda = gae_lambda  
        self.ppo_epochs = ppo_epochs
        self.kl_coef = kl_coef
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef

        # 初始化模型和tokenizer  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        self.reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        
        # 创建主策略模型（Actor）  - 这是我们要优化的模型 
        self.model = Qwen2ForCausalLM.from_pretrained(  
            model_name,  
            device_map="auto",  
            trust_remote_code=True  
        )  
        
        self.device  = self.model.device
        
        # 创建参考策略模型 - 用于KL散度计算，防止策略偏离太远  
        # 应该使用SFT模型作为参考模型，而不是与当前模型相同
        self.ref_model = Qwen2ForCausalLM.from_pretrained(  
            reference_model_name if reference_model_name else model_name,  
            device_map="auto",
            trust_remote_code=True  
        ).to(self.device)
        # 固定参考模型参数
        for param in self.ref_model.parameters():
            param.requires_grad = False

            
            
            
        # 创建奖励模型并设置为评估模式
        self.reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            trust_remote_code=True
        ).to(self.device)
        # 设置为评估模式，因为我们只使用它来计算奖励而不训练它
        self.reward_model.eval()
        
        
        # 创建价值头网络（Critic）- 用于预测状态值函数V(s)  
        # 在PPO中，Actor-Critic架构是标准做法   
        # self.value_head = torch.nn.Linear(  
        #     self.model.config.hidden_size,    # 输入维度：模型隐藏状态大小 
        #     1                                 # 输出维度：单一值，表示状态价值  
        # ).to(self.model.device)  
        
        self.critic_model = Critic(self.model).to(self.device)
        
        # 应用LoRA  
        if is_peft:  
            self.peft_config = peft_config or self._default_lora_config()  
            self.model = get_peft_model(self.model, self.peft_config)  

        # 准备训练数据集  
        self.dataset, self.eval_dataset = self._load_cached_dataset(self.cached_data_path)  
        

        # 配置优化器 - 同时优化策略模型和价值头   
        
        # [方案1， 可以用， 但是有点奇怪]
        # self.optimizer = torch.optim.AdamW(  
        #     list(self.model.parameters()) + list(self.value_head.parameters()),   
        #     lr=lr,  
        #     weight_decay=0.01   # 权重衰减，防止过拟合  
        # )  
        
        # 为actor和critic分别创建优化器
        self.optimizer_actor = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr_actor,
            weight_decay=0.01
        )
        self.optimizer_critic = torch.optim.AdamW(
            self.critic_model.parameters(), 
            lr=lr_critic,
            weight_decay=0.01
        )
        

    def _default_lora_config(self):  
        return LoraConfig(  
            r=32,  
            lora_alpha=16,  
            lora_dropout=0.05,  
            target_modules=["q_proj", "k_proj" ,"v_proj"],  
            bias="none",  
            task_type="CAUSAL_LM"  
        )  
        
        
    def _load_cached_dataset(self, dataset_path=CACHED_PPO_DATA_PATH):
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path, exist_ok=True)
        try:
            tokenized_data = load_from_disk(dataset_path)
            print("从缓存加载PPO数据集成功")
            train_data = tokenized_data['train']
            eval_data = tokenized_data['validation']
            # 确保数据格式正确
            train_data.set_format(type="torch")
            eval_data.set_format(type="torch")
            return PPODataset(train_data), PPODataset(eval_data)
        except Exception as e:
            print(f"加载缓存的PPO数据集失败: {e}, 将重新预处理数据")
            ppo_train_data, ppo_eval_data = self._prepare_dataset(self.dataset_path, self.max_seq_length)
            return ppo_train_data, ppo_eval_data

    def _prepare_dataset(self, dataset_path, max_length):  
        """数据预处理流程"""  
        dataset = load_dataset(dataset_path)
        train_dataset = load_dataset(dataset_path, split='train').select(range(500))  
        
        if "validation" in dataset:
            eval_dataset = load_dataset(dataset_path, split='validation').select(range(500)) 
        else:
            eval_dataset = load_dataset(dataset_path, split='train').select(range(500, 1000))  
            
        def process_fn(samples):  
            batch = {"input_ids": [], "attention_mask": [],  
                    "response_ids": [], "old_log_probs": []}  
            
            for prompt, chosen in zip(samples["prompt"], samples["chosen"]):  
                # 生成完整prompt  
                full_prompt = f"Instruction: {prompt}\nResponse: {chosen}"  
                
                # Tokenize输入  
                tokens = self.tokenizer(  
                    full_prompt,  
                    max_length=max_length,  
                    padding="max_length",  
                    truncation=True,  
                    return_tensors="pt"  
                ).to(self.device)
                
                # 计算旧策略概率  (使用多项式采样替代argmax)
                with torch.no_grad():  
                    logits = self.model.forward(**tokens).logits.to(self.device)
                    log_probs = F.log_softmax(logits, dim=-1)  
                    
                    # 多项式采样 - 向量化实现，提高效率
                    probs = torch.softmax(logits, dim=-1)  # shape: (batch_size, seq_len, vocab_size)
                    # 使用multinomial一次采样所有位置，避免循环
                    response_ids = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)  # (batch_size * seq_len, 1)
                    response_ids = response_ids.view(probs.size(0), probs.size(1))  # (batch_size, seq_len)
                    
                    # 只收集生成部分的log_prob
                    old_log_probs = torch.gather(  
                        log_probs, -1, response_ids.unsqueeze(-1)    # 把整个序列的logprobs都收集起来了，而不是仅仅收集action的部分
                    ).squeeze(-1)    # shape = (batch_size, seq_len)
                
                batch["input_ids"].append(tokens["input_ids"][0])  
                batch["attention_mask"].append(tokens["attention_mask"][0])  
                batch["response_ids"].append(response_ids[0])  
                batch["old_log_probs"].append(old_log_probs[0])  
            
            return batch  
        
        train_data = train_dataset.map(  
            process_fn,  
            batched=True,  
            num_proc=1,  
            remove_columns=train_dataset.column_names  
        )  
        
        val_data = eval_dataset.map(
            process_fn,
            batched=True,
            num_proc=1,
            remove_columns=eval_dataset.column_names
        )
        
        tokenized_data = DatasetDict({
            'train': train_data,
            'validation': val_data
        })
        
        
        # 保存预处理好的数据集到本地
        if not os.path.exists(CACHED_PPO_DATA_PATH):
            os.makedirs(CACHED_PPO_DATA_PATH, exist_ok=True)
        tokenized_data.save_to_disk(CACHED_PPO_DATA_PATH)
        
        
        train_data.set_format(type="torch")
        val_data.set_format(type="torch")
        
        
        
        return PPODataset(train_data), PPODataset(val_data)

    def ppo_collator(self, batch):
        """自定义数据整理函数，用于处理PPO训练的数据批次"""
        # 预先转换所有字段为张量，避免循环中重复转换
        input_ids_list = [item["input_ids"] for item in batch]
        attention_mask_list = [item["attention_mask"] for item in batch]
        response_ids_list = [item["response_ids"] for item in batch]
        old_log_probs_list = [item["old_log_probs"] for item in batch]

        # 获取批次中的最大长度
        max_len = min(max(len(ids) for ids in input_ids_list), self.max_seq_length)

        batch_size = len(batch)

        # 预先转换列表为张量，使用stack一次完成
        input_ids_tensor = torch.stack([
            torch.as_tensor(ids[:max_len], dtype=torch.long) for ids in input_ids_list
        ])
        attention_mask_tensor = torch.stack([
            torch.as_tensor(mask[:max_len], dtype=torch.long) for mask in attention_mask_list
        ])
        response_ids_tensor = torch.stack([
            torch.as_tensor(ids[:max_len], dtype=torch.long) for ids in response_ids_list
        ])
        old_log_probs_tensor = torch.stack([
            torch.as_tensor(lp[:max_len], dtype=torch.float) for lp in old_log_probs_list
        ])

        # 创建填充后的张量
        padded_input_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_response_ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_old_log_probs = torch.zeros((batch_size, max_len), dtype=torch.float)

        # 计算每个序列的实际长度
        lengths = torch.tensor([min(len(ids), max_len) for ids in input_ids_list], dtype=torch.long)

        # 使用scatter将数据填充到正确位置
        for i in range(batch_size):
            seq_len = lengths[i].item()
            padded_input_ids[i, :seq_len] = input_ids_tensor[i, :seq_len]
            padded_attention_mask[i, :seq_len] = attention_mask_tensor[i, :seq_len]
            padded_response_ids[i, :seq_len] = response_ids_tensor[i, :seq_len]
            padded_old_log_probs[i, :seq_len] = old_log_probs_tensor[i, :seq_len]

        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "response_ids": padded_response_ids,
            "old_log_probs": padded_old_log_probs,
        }
        
    def compute_rewards(self, model_outputs, response_ids, attention_mask, input_ids):  
        """计算奖励，包括外部奖励和KL惩罚
        
        Args:
            model_outputs: 模型输出
            response_ids: 生成的响应ID
            attention_mask: 注意力掩码
            input_ids: 输入ID，用于区分prompt和response部分
            
        Returns:
            rewards: 每个token位置的奖励 (bsz, seq_len)
            new_log_probs: 新策略的log概率 (bsz, seq_len)
            kl: KL散度 (bsz, seq_len)
        """
        # 提取prompt部分和response部分的边界
        batch_size, seq_len = response_ids.shape
        
        # 计算KL散度惩罚
        logits = model_outputs.logits   # # 获取当前策略模型的logits输出，形状为(batch_size, seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)   # 对logits应用log_softmax得到标准化的log概率分布
        new_log_probs = torch.gather(     # 从概率分布中提取实际生成token对应的log概率
            log_probs, -1, response_ids.unsqueeze(-1)   # response_ids是模型实际生成的token序列 
        ).squeeze(-1)    # 压缩维度，最终得到形状为(batch_size, seq_len)的log概率矩阵
        
        # 计算参考模型的概率（用于KL散度）  
        with torch.no_grad():  
            ref_outputs = self.ref_model(    # 使用参考模型（通常是SFT模型）进行前向推理
                input_ids=input_ids,  
                attention_mask=attention_mask  
            ) 
            ref_logits = ref_outputs.logits  
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)  
            ref_log_probs = torch.gather(    # 提取相同生成token在参考模型中的log概率
                ref_log_probs, -1, response_ids.unsqueeze(-1)  
            ).squeeze(-1)   # 压缩维度，形状为(batch_size, seq_len)
            
        # 计算KL散度  
        kl = (new_log_probs - ref_log_probs) * attention_mask  # shape = (bsz, seq_len)
        
        # 初始化奖励矩阵
        rewards = torch.zeros_like(new_log_probs)  
        
        # 使用reward model计算外部奖励
        # 首先确定response部分（假设prompt以特定token或格式结束）
        # 这里简单处理：找到第一个非padding token作为response开始
        response_text = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        reward_model_inputs = self.reward_tokenizer(response_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        reward_model_inputs = {k: v.to(self.reward_model.device) for k, v in reward_model_inputs.items()}
        
        with torch.no_grad():
            reward_outputs = self.reward_model(**reward_model_inputs)
            external_rewards = reward_outputs.logits.squeeze(-1)  # shape = (bsz,)
        
        # 组合奖励：KL惩罚应用于所有token，外部奖励只应用于response的最后一个token
        # 1. 应用KL惩罚
        rewards = -self.kl_coef * kl
        
        # 2. 找到每个序列的最后一个非padding token位置
        for i in range(batch_size):
            # 找到最后一个为1的位置
            last_valid_idx = torch.where(attention_mask[i] == 1)[0][-1] if torch.sum(attention_mask[i]) > 0 else seq_len - 1
            rewards[i, last_valid_idx] += external_rewards[i]
        
        return rewards, new_log_probs, kl  
        
    def compute_gae(self, rewards, values, masks, gamma=0.99, lam=0.95):  
        """计算广义优势估计(GAE)
        
        Args:
            rewards: 奖励 (bsz, seq_len)
            values: 价值估计 (bsz, seq_len)
            masks: 掩码 (bsz, seq_len)
            gamma: 折扣因子
            lam: GAE lambda参数
            
        Returns:
            advantages: 优势估计 (bsz, seq_len)
            returns: 回报 (bsz, seq_len)
        """  
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)  
        
        # 计算最后一个时间步的优势
        last_gae_lam = 0   # A_{T+1}==0
        
        # 反向遍历序列以计算GAE  
        for t in reversed(range(seq_len)):  
            if t == seq_len - 1:  
                next_value = 0   # V(S_{t+1}) = 0
            else:  
                next_value = values[:, t + 1]  
                
            # 计算TD误差
            delta = rewards[:, t] + gamma * next_value * masks[:, t] - values[:, t]  

            '''
            masks[:, t] （第t个时间步的掩码）有两个重要作用：

            1. 在TD误差计算中 
                - 这里 masks[:, t] 用于确保只有有效token（非padding）的未来价值被考虑。
                - 对于padding token， masks[:, t]=0 ，这样就会将 gamma * next_value * masks[:, t] 项置零，只保留当前奖励减去当前价值估计。

            2. 在递归计算GAE时 ：
                - last_gae_lam = delta + gamma * lam * masks[:, t] * last_gae_lam
                - 等价于 A(S_t) = delta + gamma * lam * mask[:, t] * A(S_{t+1})
                这里 masks[:, t] 用于控制是否将后续时间步的优势估计回传。对于padding token，后续时间步的优势不应影响当前步，因此将其置零。

            '''
            # 递归计算GAE
            last_gae_lam = delta + gamma * lam * masks[:, t] * last_gae_lam  
            advantages[:, t] = last_gae_lam  
        
        # 计算回报（returns = advantages + values）  
        returns = advantages + values  
        
        # 标准化优势 - 对每个样本单独标准化
        for i in range(batch_size):
            valid_advantages = advantages[i][masks[i] > 0]
            if len(valid_advantages) > 0:
                mean = valid_advantages.mean()
                std = valid_advantages.std() + 1e-8
                advantages[i][masks[i] > 0] = (valid_advantages - mean) / std
        
        return advantages, returns  

    def compute_loss(self, actor_model:Qwen2ForCausalLM, critic_model:Critic, inputs):  
        # 前向传播获取logits  
        outputs = actor_model(  
            input_ids=inputs["input_ids"],  
            attention_mask=inputs["attention_mask"] ,
            output_hidden_states=True
        )  
        
        # 计算奖励和KL散度  
        rewards, new_log_probs, kl = self.compute_rewards(  
            outputs,   
            inputs["response_ids"],  
            inputs["attention_mask"],
            inputs["input_ids"]  # 传递input_ids以帮助区分prompt和response
        )  
        
        # 使用critic模型计算每个token的价值  
        values = critic_model(  
            input_ids=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]  
        )  # shape = (bsz, seq_len)
        
        # 计算蒙版（用于忽略padding）  
        masks = inputs["attention_mask"]  
        
        # 计算GAE优势和回报  
        advantages, returns = self.compute_gae(  
            rewards, values, masks,   
            gamma=self.gamma, lam=self.gae_lambda  
        )  
        
        # 计算概率比率  
        ratio = torch.exp(new_log_probs - inputs["old_log_probs"])  
        
        # 计算策略损失 (clipped PPO objective)  
        # 只考虑有效的token位置（使用mask）
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2) * masks  # 应用mask
        policy_loss = policy_loss.sum() / masks.sum()  # 按有效token数量归一化
        
        # 计算价值函数损失  
        value_loss = F.mse_loss(values * masks, returns * masks, reduction='sum') / masks.sum()
        
        # 计算熵奖励（增加探索性）  
        logits = outputs.logits
        entropy = -torch.sum(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1), dim=-1)
        entropy = (entropy * masks).sum() / masks.sum()  # 按有效token数量归一化
        
        # 总损失  
        total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy  
        
        return total_loss  

    def train(self):  
        """自定义训练循环"""
        train_args = TrainingArguments(  
            output_dir=self.output_dir,  
            per_device_train_batch_size=2,  
            gradient_accumulation_steps=2,  
            learning_rate=5e-5,  
            logging_steps=10,  
            save_steps=500,  
            remove_unused_columns=False,  
            optim="adamw_torch",  
            max_grad_norm=0.5  
        )  
        
        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=train_args.per_device_train_batch_size,
            collate_fn=self.ppo_collator,
            shuffle=True
        )
        
        # 自定义的PPO算法无法使用huggingface官方的Trainer 
        
        # trainer = Trainer(  
        #     model=self.model,  
        #     args=train_args,  
        #     train_dataset=self.dataset,  
        #     data_collator=self.ppo_collator,  
        #     compute_metrics=None,  
        #     compute_loss=self.compute_loss  
        # )  
        
        best_eval_loss = float('inf')
        
        # PPO多轮优化  
        for epoch in range(self.ppo_epochs):
            self.model.train()
            self.critic_model.train()
            
            progress_bar = tqdm(train_dataloader, desc=f"PPO Epoch {epoch+1}/{self.ppo_epochs}")
            
            for step, batch in enumerate(progress_bar):
                
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # 梯度清零
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                
                # 计算损失
                loss = self.compute_loss(self.model, self.critic_model, batch)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), train_args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic_model.parameters(), train_args.max_grad_norm)
                
                # 参数更新
                self.optimizer_actor.step()
                self.optimizer_critic.step()
                
                # 更新进度条
                progress_bar.set_postfix(loss=loss.item())
                
                # 记录日志
                if (step + 1) % train_args.logging_steps == 0:
                    print(f"Epoch {epoch+1}, Step {step+1}: Loss = {loss.item():.4f}")
            
            eval_loss = self.evaluate()
            
            # 保存最佳模型
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                # self.save_model()
                print(f"New best eval loss: {best_eval_loss:.4f}")
            
            
        self.save_model() 
        
        
        
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        self.critic_model.eval()
        
        eval_dataloader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=4,  # 评估batch_size可以小一些
            collate_fn=self.ppo_collator,
            shuffle=False
        )
        
        total_loss = 0
        total_valid_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 计算损失
                loss = self.compute_loss(self.model, self.critic_model, batch)
                
                # 统计有效token数量
                valid_tokens = batch["attention_mask"].sum().item()
                total_loss += loss.item() * valid_tokens
                total_valid_tokens += valid_tokens
        
        avg_loss = total_loss / total_valid_tokens if total_valid_tokens > 0 else 0
        print(f"\nEvaluation - Average Loss: {avg_loss:.4f}")
        
        return avg_loss
            
    def save_model(self):
        """保存训练好的模型"""
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存策略模型（主要模型）
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # 保存价值头模型（Critic）
        critic_output_dir = os.path.join(self.output_dir, "critic")
        os.makedirs(critic_output_dir, exist_ok=True)
        torch.save(self.critic_model.state_dict(), os.path.join(critic_output_dir, "critic_model.bin"))
        
        # 保存训练配置
        config = {
            "clip_epsilon": self.clip_epsilon,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "kl_coef": self.kl_coef,
            "vf_coef": self.vf_coef,
            "entropy_coef": self.entropy_coef
        }
        torch.save(config, os.path.join(self.output_dir, "training_config.pt"))
        
        print(f"模型已保存至 {self.output_dir}")
        
        
        
        
        
        




if __name__ == "__main__":
    
    
    trainer = PPOTrainer()
    
    trainer.train()